"""Figures and diagnostics for the report.

    python analysis.py curves                       # training curves (val PSNR, bank PSNR)
    python analysis.py uncertainty --ckpt checkpoints/ours/final.pth   # U vs. error on test set
    python analysis.py qualitative --runs supervised baseline ours     # needs eval.py --save_dir
    python analysis.py stats                         # paired Wilcoxon tests -> results/significance.txt
    python analysis.py uncsource --ckpt checkpoints/ours/final.pth    # flip vs colour U (train set)
    python analysis.py gate --runs supervised_l50 baseline_l50 supervised baseline ours ours_color

Outputs go to results/figures/ and results/uncertainty.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from models.unet import UNet
from semiuir.data import Paired
from semiuir.metrics import peak_signal_noise_ratio, psnr_ssim
from semiuir.uncertainty import teacher_tta
from train import get_device

FIG = "results/figures"
# fixed colour per method (never re-assigned by rank)
COLORS = {"baseline": "#2a78d6", "ours": "#eb6834", "supervised": "#1baf7a", "ours_color": "#e87ba4"}
LABELS = {"baseline": "Semi-UIR", "ours": "UA-Bank (flip U)", "supervised": "Supervised only",
          "ours_color": "UA-Bank (colour U)"}
for _k in ("baseline", "ours", "supervised"):  # 50-label runs share the colour of their method
    COLORS[f"{_k}_l50"] = COLORS[_k]
LABELS.update({"baseline_l50": "Semi-UIR", "ours_l50": "UA-Bank (ours)", "supervised_l50": "Supervised only"})
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "lines.linewidth": 2})


def read_log(name):
    p = f"results/logs/{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def curves(args):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2))
    for name in args.runs:
        rows = [r for r in read_log(name) if "val_psnr20" in r]
        if not rows:
            continue
        it = [r["iter"] for r in rows]
        axes[0].plot(it, [r["val_psnr20"] for r in rows], color=COLORS[name], label=LABELS[name])
        b = [(r["iter"], r["bank_psnr"]) for r in rows if "bank_psnr" in r]
        if b:
            axes[1].plot(*zip(*b), color=COLORS[name], label=LABELS[name])
    axes[0].set(xlabel="iteration", ylabel="PSNR (dB)", title="Student PSNR (20 test imgs)")
    axes[1].set(xlabel="iteration", ylabel="PSNR vs hidden GT (dB)", title="Reliable-bank pseudo-label quality")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = f"{FIG}/curves_{args.tag}.png"
    fig.savefig(out, dpi=250)
    print("saved", out)


@torch.no_grad()
def uncertainty(args):
    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    teacher = UNet(ck["cfg"]["width"]).to(dev).eval()
    teacher.load_state_dict(ck["teacher"])
    ds = Paired("data/UIEB/test")
    rng = np.random.default_rng(0)
    rho_img, all_u, all_e, gain = [], [], [], []
    for a, b, _ in ds:
        x = a[None].to(dev)
        p_tta, u = teacher_tta(teacher, x)
        p_one = teacher(x)
        err = (p_tta.clamp(0, 1).cpu()[0] - b).abs().mean(0).flatten().numpy()
        uu = u.cpu()[0, 0].flatten().numpy()
        rho_img.append(spearmanr(uu, err)[0])
        sel = rng.choice(len(uu), 2000, replace=False)
        all_u.append(uu[sel]); all_e.append(err[sel])
        gain.append(psnr_ssim(p_tta[0].cpu(), b)[0] - psnr_ssim(p_one[0].cpu(), b)[0])
    all_u, all_e = np.concatenate(all_u), np.concatenate(all_e)
    # error of the most / least confident pixels: what the confidence weight down-weights
    q = np.quantile(all_u, np.linspace(0, 1, 11))
    bins = np.clip(np.digitize(all_u, q[1:-1]), 0, 9)
    err_bin = [float(all_e[bins == k].mean()) for k in range(10)]
    res = {"spearman_per_image_mean": float(np.nanmean(rho_img)),
           "spearman_pooled": float(spearmanr(all_u, all_e)[0]),
           "err_by_uncertainty_decile": err_bin,
           "tta_minus_single_psnr_db": float(np.mean(gain))}
    json.dump(res, open("results/uncertainty.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.bar(range(1, 11), err_bin, color=COLORS["ours"], width=0.8)
    ax.set(xlabel="teacher-uncertainty decile (low → high)", ylabel="mean |pseudo-label − GT|",
           title=f"Uncertainty predicts error (Spearman ρ={res['spearman_pooled']:.2f})")
    fig.tight_layout()
    fig.savefig(f"{FIG}/uncertainty_vs_error.png", dpi=250)
    print("saved", f"{FIG}/uncertainty_vs_error.png")


@torch.no_grad()
def qualitative(args):
    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    teacher = UNet(ck["cfg"]["width"]).to(dev).eval()
    teacher.load_state_dict(ck["teacher"])
    names = args.images
    cols = ["input"] + args.runs + ["gt", "uncertainty"]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(1.25 * len(cols), 1.0 * len(names)))
    for r, n in enumerate(names):
        inp = np.array(Image.open(f"data/UIEB/test/input/{n}"))
        gt = np.array(Image.open(f"data/UIEB/test/gt/{n}"))
        x = torch.from_numpy(inp).permute(2, 0, 1)[None].float().div(255).to(dev)
        _, u = teacher_tta(teacher, x)
        ims = [inp] + [np.array(Image.open(f"results/outputs/{k}/{n}")) for k in args.runs] + [gt, u[0, 0].cpu().numpy()]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im, cmap="magma" if im.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(LABELS.get(cols[c], cols[c]).replace(" (", "\n("), fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/qualitative.png", dpi=250)
    print("saved", f"{FIG}/qualitative.png")


def stats(_):
    """Paired per-image PSNR comparisons (Wilcoxon signed-rank) from the saved test outputs."""
    from scipy.stats import wilcoxon
    gt_dir = "data/UIEB/test/gt"
    names = sorted(os.listdir(gt_dir))

    def per_image(run):
        return np.array([peak_signal_noise_ratio(np.array(Image.open(f"{gt_dir}/{n}")),
                                                 np.array(Image.open(f"results/outputs/{run}/{n}")),
                                                 data_range=255) for n in names])

    pairs = [("baseline", "supervised"), ("ours", "baseline"), ("ours_color", "baseline"),
             ("baseline_l50", "supervised_l50"), ("ours_l50", "baseline_l50"), ("ours_l50", "supervised_l50")]
    cache, lines = {}, []
    for a, b in pairs:
        for r in (a, b):
            cache.setdefault(r, per_image(r))
        d = cache[a] - cache[b]
        lines.append(f"{a:12s} vs {b:15s} {d.mean():+.3f} dB  win {np.mean(d > 0) * 100:3.0f}%  "
                     f"Wilcoxon p={wilcoxon(d).pvalue:.3f}")
    open("results/significance.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


@torch.no_grad()
def uncsource(args):
    """Which perturbation family gives an uncertainty that predicts pseudo-label error?
    Evaluated on LABELED TRAINING images (not the test set) with the given teacher."""
    from semiuir.data import Labeled
    torch.manual_seed(0)
    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    t = UNet(ck["cfg"]["width"]).to(dev).eval()
    t.load_state_dict(ck["teacher"])
    d = Labeled("data/UIEB/labeled")
    res = {"flip": [], "color": []}
    for i in range(0, len(d), 2):
        x = d.inp[i].float()[None].div(255).to(dev)
        y = d.gt[i].float()[None].div(255).to(dev)
        for mode in res:
            p, u = teacher_tta(t, x, 4, mode)
            e = (p - y).abs().mean(1).flatten().cpu().numpy()
            uu = u.flatten().cpu().numpy()
            sel = np.random.default_rng(i).choice(len(e), 3000, replace=False)
            res[mode].append(spearmanr(uu[sel], e[sel])[0])
    out = {m: float(np.nanmean(v)) for m, v in res.items()}
    json.dump(out, open("results/uncertainty_source.json", "w"), indent=2)
    print("mean per-image Spearman(U, |P-GT|) on labeled train images:", out)


@torch.no_grad()
def gate(args):
    """How often does a bank gate pick the output that is truly closer to GT? Pairs of outputs from
    different trained teachers on the 50-label unlabeled pool (hidden GT, training split only)."""
    import itertools
    from semiuir.data import Unlabeled
    from semiuir.metrics import uiqm
    torch.manual_seed(0)
    dev = get_device()
    x_all = Unlabeled("data/UIEB/L50/unlabeled", 256).inp
    y_all = Unlabeled("data/UIEB/L50", 256, sub="unlabeled_gt_analysis").inp
    idx = list(range(0, len(x_all), 10))
    P, Q, U = {}, {}, {}
    for n in args.runs:
        ck = torch.load(f"checkpoints/{n}/final.pth", map_location="cpu")
        t = UNet(ck["cfg"]["width"]).to(dev).eval()
        t.load_state_dict(ck["teacher"])
        ps, qs, us = [], [], []
        for i in idx:
            x = x_all[i].float()[None].div(255).to(dev)
            y = y_all[i].float()[None].div(255).to(dev)
            p, u = teacher_tta(t, x, 4, "color", 0.15, "plain")
            ps.append((10 * torch.log10(1 / ((p.clamp(0, 1) - y) ** 2).mean())).item())
            qs.append(uiqm(p.clamp(0, 1)).item())
            us.append(u.mean().item())
        P[n], Q[n], U[n] = map(np.array, (ps, qs, us))
    acc_q, acc_u = [], []
    for a, b in itertools.combinations(args.runs, 2):
        better, m = P[a] > P[b], np.abs(P[a] - P[b]) > 0.1
        acc_q.append(((Q[a] > Q[b]) == better)[m])
        acc_u.append(((U[a] < U[b]) == better)[m])
    out = {"pairs": int(np.concatenate(acc_q).size),
           "uiqm_gate_accuracy": float(np.concatenate(acc_q).mean()),
           "colour_uncertainty_gate_accuracy": float(np.concatenate(acc_u).mean())}
    json.dump(out, open("results/gate_accuracy.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["curves", "uncertainty", "qualitative", "stats", "uncsource", "gate"])
    ap.add_argument("--ckpt", default="checkpoints/ours/final.pth")
    ap.add_argument("--runs", nargs="+", default=["supervised", "baseline", "ours"])
    ap.add_argument("--images", nargs="+", default=["0005.png", "0030.png", "0061.png"])
    ap.add_argument("--tag", default="l50", help="suffix of the curves figure")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    {"curves": curves, "uncertainty": uncertainty, "qualitative": qualitative,
     "stats": stats, "uncsource": uncsource, "gate": gate}[a.what](a)
