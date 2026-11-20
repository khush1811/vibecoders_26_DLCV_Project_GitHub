"""Train supervised-only / Semi-UIR baseline / UA-Bank (ours) from a YAML config.

    python train.py --config configs/ours.yaml
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from models.unet import build_model
from semiuir.bank import ReliableBank
from semiuir.data import Labeled, Paired, ScenePairs, Unlabeled, list_png
from semiuir.fidelity import FidelityBank, FidelityRisk
from semiuir.losses import ContrastLoss, VGGFeatures, gradient, supervised_loss, weighted_l1
from semiuir.uncertainty import teacher_tta


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def cycle(loader):
    while True:
        yield from loader


@torch.no_grad()
def ema_update(teacher, student, it, keep=0.996):
    a = min(1 - 1 / (it + 1), keep)
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(a).add_(s.detach(), alpha=1 - a)


def rampup(it, length):
    """Sigmoid ramp-up of the consistency weight (official Semi-UIR schedule)."""
    if length == 0:
        return 1.0
    phase = 1.0 - min(it, length) / length
    return float(math.exp(-5.0 * phase * phase))


@torch.no_grad()
def quick_val(model, ds, dev, n=20):
    was_training = model.training
    model.eval()
    ps = []
    for i in range(min(n, len(ds))):
        a, b, _ = ds[i]
        mse = (model(a[None].to(dev))[0].clamp(0, 1).cpu() - b).square().mean().item()
        ps.append(-10 * math.log10(mse) if mse else float("inf"))
    model.train(was_training)
    return float(np.mean(ps))


def scene_loss(out, target, exposure_weight=0.0, log_mse_weight=0.0):
    error = out - target
    dx, dy = gradient(error)
    loss = (F.l1_loss(out, target) + 2 * F.mse_loss(out, target)
            + 0.25 * F.l1_loss(F.avg_pool2d(out, 8), F.avg_pool2d(target, 8))
            + 0.1 * (dx.abs().mean() + dy.abs().mean()))
    if exposure_weight:
        loss = loss + exposure_weight * error.mean(dim=(-2, -1)).abs().mean()
    if log_mse_weight:
        # Mean log MSE gives each scene a contribution on the same scale as mean PSNR.
        mse_by_scene = error.square().mean(dim=(1, 2, 3))
        loss = loss + log_mse_weight * torch.log1p(mse_by_scene / 0.0025).mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", type=int, help="override total iterations (e.g. smoke test)")
    ap.add_argument("--name", help="output name for a new run")
    ap.add_argument("--supervised", action="store_true", help="disable semi-supervision for a matched control")
    ap.add_argument("--data-root", help="override the config's dataset path")
    ap.add_argument("--seed", type=int, help="override the training seed")
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"))
    holdout_args = ap.add_mutually_exclusive_group()
    holdout_args.add_argument("--holdout", type=int, default=0, help="number of labeled scenes held out for development")
    holdout_args.add_argument("--holdout-names", help="text file with one held-out labeled filename per line")
    ap.add_argument("--init", help="fine-tune the student weights from a saved checkpoint")
    args = ap.parse_args()
    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)
    if args.iters is not None:
        cfg["iters"] = args.iters
    if args.name:
        cfg["name"] = args.name
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.supervised:
        if not args.name:
            ap.error("--supervised requires a distinct --name")
        cfg["semi"] = False
    if (args.holdout or args.holdout_names) and not args.name:
        ap.error("--holdout requires a distinct --name")
    if cfg["iters"] < 1:
        ap.error("iterations must be positive")
    seed_all(cfg["seed"])
    dev = torch.device(args.device) if args.device else get_device()
    if dev.type == "cpu" and "cpu_threads" in cfg:
        torch.set_num_threads(cfg["cpu_threads"])

    root = cfg["data_root"]
    scene = cfg.get("architecture") == "scene_color_unet"
    lab_root = os.path.join(root, "labeled")
    val = None
    held_out = set()
    if scene:
        names = list_png(os.path.join(lab_root, "input"))
        if args.holdout_names:
            with open(args.holdout_names, encoding="utf-8") as name_file:
                chosen = [line.strip() for line in name_file if line.strip()]
            held_out = set(chosen)
            if not chosen or len(chosen) != len(held_out) or not held_out.issubset(names):
                ap.error("the holdout file must list unique filenames from the labeled split")
        elif 0 <= args.holdout <= len(names) - cfg["batch"]:
            held_out = set(random.Random(0).sample(names, args.holdout))
        else:
            ap.error("holdout must leave at least one full training batch")
        if held_out:
            val = Paired(lab_root)
            val.names = sorted(held_out)
            cfg["holdout_names"] = val.names
        scene_size = cfg.get("scene_size", cfg["crop"])
        labeled = ScenePairs(lab_root, [n for n in names if n not in held_out], scene_size,
                             tone_augmentation=cfg.get("tone_augmentation", 0.0),
                             cast_mix_probability=cfg.get("cast_mix_probability", 0.0))
    else:
        if args.holdout or args.holdout_names:
            ap.error("holdout options are supported for scene_color_unet runs")
        labeled = Labeled(lab_root, cfg["crop"])
        val = Paired(os.path.join(root, "test"))
    if len(labeled) < cfg["batch"]:
        ap.error("the labeled split must contain at least one full batch")

    student = build_model(cfg).to(dev)
    if args.init:
        source = torch.load(args.init, map_location="cpu", weights_only=True)
        if held_out and not held_out.issubset(source["cfg"].get("holdout_names", [])):
            ap.error("the initialization checkpoint must have excluded every development scene")
        student.load_state_dict(source["student"])
        cfg["initial_checkpoint"] = args.init

    out_dir = os.path.join("checkpoints", cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    log_path = os.path.join("results/logs", f"{cfg['name']}.jsonl")
    log = open(log_path, "w")
    print(json.dumps(cfg))

    # set `workers: 2` in a config to overlap CPU augmentation with GPU compute (faster, more CPU load)
    dl = dict(shuffle=True, drop_last=True, num_workers=cfg.get("workers", 0), persistent_workers=cfg.get("workers", 0) > 0)
    lab_rng = torch.Generator().manual_seed(cfg["seed"]) if scene else None
    lab = DataLoader(labeled, cfg["batch"], generator=lab_rng, **dl)
    lab_it = cycle(lab)
    semi = cfg["semi"]
    fidelity = semi and cfg["bank_mode"] == "fidelity"
    if fidelity and (not scene or cfg["bank_size"] != cfg["crop"] or cfg["crop"] % 8):
        ap.error("fidelity bank requires scene_color_unet and a full scene size divisible by 8")
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if semi:
        uds = Unlabeled(os.path.join(root, "unlabeled"), cfg["bank_size"], cfg["crop"], mild=fidelity)
        if len(uds) < cfg["batch"]:
            ap.error("the unlabeled split must contain at least one full batch")
        unl_rng = torch.Generator().manual_seed(cfg["seed"] + 1) if scene else None
        unl_it = cycle(DataLoader(uds, cfg["batch"], generator=unl_rng, **dl))
        if fidelity:
            risk = FidelityRisk().to(dev)
            risk_opt = torch.optim.Adam(risk.parameters(), lr=0.003)
            bank = FidelityBank(len(uds), cfg["bank_size"], risk, dev)
        else:
            bank = ReliableBank(len(uds), cfg["bank_size"], cfg["bank_mode"], cfg.get("reject_q", 0.2), dev)
        # hidden GT of the unlabeled images: used ONLY to log pseudo-label quality, never in a loss
        gt_dir = os.path.join(root, "unlabeled_gt_analysis")
        bank_gt = None
        if cfg.get("bank_diagnostics", not fidelity) and os.path.isdir(gt_dir):
            if uds.names != list_png(gt_dir):
                ap.error("unlabeled diagnostic references must match unlabeled input names")
            bank_gt = torch.stack(Unlabeled(root, cfg["bank_size"], sub="unlabeled_gt_analysis").inp)

    if not scene or (semi and not fidelity):
        vgg = VGGFeatures().to(dev)
    if semi and not fidelity:
        contrast = ContrastLoss(vgg)
    opt = torch.optim.Adam(student.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"], eta_min=1e-6)
    ramp_len = int(cfg["rampup_frac"] * cfg["iters"])
    warmup = int(cfg.get("warmup_frac", 0.25) * cfg["iters"])
    exposure_weight = cfg.get("exposure_weight", 0.0)
    log_mse_weight = cfg.get("log_mse_weight", 0.0)

    t0 = time.time()
    stats = {}
    stat_steps = 0
    for it in range(1, cfg["iters"] + 1):
        x_l, y_l = [t.to(dev) for t in next(lab_it)]
        out_l = student(x_l)
        if scene:
            loss_sup = scene_loss(out_l, y_l, exposure_weight, log_mse_weight)
        else:
            loss_sup = supervised_loss(vgg, out_l, y_l)
        loss = loss_sup
        rec = {"sup": loss_sup.item()}

        if fidelity:
            with torch.no_grad():
                pred_l, unc_l = teacher_tta(teacher, x_l, k=2, mode="color", strength=0.1, pseudo="plain")
            risk_loss = risk.loss(x_l, pred_l.clamp(0, 1), unc_l, y_l)
            risk_opt.zero_grad(set_to_none=True)
            risk_loss.backward()
            risk_opt.step()
            rec["risk"] = risk_loss.item()
            if it > warmup:
                x_w, x_s, idx, yx = next(unl_it)
                x_w, x_s = x_w.to(dev), x_s.to(dev)
                pred, uncertainty = teacher_tta(teacher, x_w, k=2, mode="color", strength=0.1, pseudo="plain")
                pseudo, confidence, accepted = bank.update(idx, yx, x_w, pred, uncertainty)
                error = student(x_s) - pseudo
                loss_unsup = (confidence * (error.abs() + error.square())).mean()
                lam = cfg["consistency"] * rampup(it - warmup, ramp_len)
                loss = loss + lam * loss_unsup
                rec.update(unsup=loss_unsup.item(), lam=lam, w_mean=confidence.mean().item(), accepted=accepted.item())
        elif semi:
            x_w, x_s, idx, yx = next(unl_it)
            x_w, x_s = x_w.to(dev), x_s.to(dev)
            teacher.eval()
            if cfg["tta"]:  # [UA-Bank] TTA pseudo-label + per-pixel uncertainty
                p_t, u_t = teacher_tta(teacher, x_w, cfg["tta_k"], cfg.get("tta_mode", "flip"),
                                       cfg.get("tta_strength", 0.15), cfg.get("tta_pseudo", "mean"))
            else:  # original: single teacher prediction
                with torch.no_grad():
                    p_t = teacher(x_w)
                u_t = torch.zeros_like(p_t[:, :1])
            out_u = student(x_s)
            p_bank, u_bank, filled = bank.update(idx, yx, p_t, u_t, out_u.detach())

            w = filled.float()
            if cfg["weighted"] and filled.any():  # [UA-Bank] confidence weight w = exp(-U / tau)
                u = torch.where(filled, u_bank, torch.zeros_like(u_bank))
                tau = u[filled].mean().clamp_min(1e-6)
                w = w * torch.exp(-u / tau)
            if filled.any():
                loss_unsup = weighted_l1(out_u, p_bank, w) + contrast(out_u, p_bank, x_s, w)
                lam = cfg["consistency"] * rampup(it, ramp_len)
                loss = loss + lam * loss_unsup
                rec.update(unsup=loss_unsup.item(), lam=lam, filled=filled.float().mean().item(),
                           w_mean=w[filled].mean().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema_update(teacher, student, it, cfg["ema"])

        for k, v in rec.items():
            stats[k] = stats.get(k, 0) + v
        stat_steps += 1
        if it % cfg["log_every"] == 0 or it == cfg["iters"]:
            row = {"iter": it, "time_min": (time.time() - t0) / 60, "lr": sched.get_last_lr()[0]}
            row.update({k: v / stat_steps for k, v in stats.items()})
            if it % cfg["val_every"] == 0 or it == cfg["iters"]:
                if val is not None:
                    if scene:
                        row["dev_psnr"] = quick_val(student, val, dev, len(val))
                        row["dev_ema_psnr"] = quick_val(teacher, val, dev, len(val))
                    else:
                        row["val_psnr20"] = quick_val(student, val, dev)
                if semi and bank_gt is not None:
                    row["bank_psnr"], row["bank_coverage"] = bank.quality(bank_gt)
            stats = {}
            stat_steps = 0
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if dev.type == "mps":  # release cached GPU memory to keep the machine responsive
                torch.mps.empty_cache()

    torch.save({"cfg": cfg, "student": student.state_dict(), "teacher": teacher.state_dict(),
                "train_minutes": (time.time() - t0) / 60}, os.path.join(out_dir, "final.pth"))
    log.close()
    print("saved", os.path.join(out_dir, "final.pth"))


if __name__ == "__main__":
    main()
