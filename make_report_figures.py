"""Generate every figure used in the multi-page report from the saved results.

    python make_report_figures.py          # needs results/outputs (eval.py --save_dir) and checkpoints/
Outputs: results/figures/report_*.png
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from models.unet import UNet
from semiuir.metrics import peak_signal_noise_ratio
from semiuir.uncertainty import teacher_tta
from train import get_device

FIG = "results/figures"
SUP, BASE, OURS, OURS2, GREY = "#1baf7a", "#2a78d6", "#eb6834", "#e87ba4", "#8a8f94"
COL = {"supervised": SUP, "baseline": BASE, "ours": OURS, "ours_color": OURS2,
       "supervised_l50": SUP, "baseline_l50": BASE, "ours_l50": OURS}
SHORT = {"supervised": "Supervised", "baseline": "Semi-UIR", "ours": "UA-Bank flip", "ours_color": "UA-Bank colour",
         "supervised_l50": "Supervised", "baseline_l50": "Semi-UIR", "ours_l50": "UA-Bank colour"}
LAB = {"supervised": "Supervised", "baseline": "Semi-UIR", "ours": "UA-Bank (flip U)",
       "ours_color": "UA-Bank (colour U)", "supervised_l50": "Supervised", "baseline_l50": "Semi-UIR",
       "ours_l50": "UA-Bank (colour U)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.edgecolor": "#52514e", "lines.linewidth": 2, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=220)
    plt.close(fig)
    print("saved", f"{FIG}/{name}.png")


def metrics():
    return {r["method"]: {k: float(v) for k, v in r.items() if k != "method"}
            for r in csv.DictReader(open("results/metrics.csv"))}


def log(name):
    return [json.loads(l) for l in open(f"results/logs/{name}.jsonl")]


def per_image(run):
    g = "data/UIEB/test/gt"
    return np.array([peak_signal_noise_ratio(np.array(Image.open(f"{g}/{n}")),
                                             np.array(Image.open(f"results/outputs/{run}/{n}")), data_range=255)
                     for n in sorted(os.listdir(g))])


def bars_main(m):
    """PSNR and SSIM bars for both label regimes, raw input as reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    groups = [("200 labels", ["supervised", "baseline", "ours", "ours_color"]),
              ("50 labels", ["supervised_l50", "baseline_l50", "ours_l50"])]
    for ax, metric, lo in zip(axes, ["PSNR", "SSIM"], [16, 0.72]):
        x, ticks, names = 0, [], []
        for gname, runs in groups:
            for r in runs:
                v = m[r][metric]
                ax.bar(x, v - lo, bottom=lo, color=COL[r], width=0.8)
                ax.text(x, v, f"{v:.2f}" if metric == "PSNR" else f"{v:.3f}", ha="center", va="bottom", fontsize=7)
                ticks.append(x); names.append(SHORT[r])
                x += 1
            x += 0.7
        ax.axhline(m["input (no restoration)"][metric], color=GREY, ls="--", lw=1)
        ax.text(ticks[-1] + 0.6, m["input (no restoration)"][metric], "raw input", color=GREY, fontsize=7, va="bottom", ha="right")
        ax.set_xticks(ticks, names, fontsize=7, rotation=35, ha="right")
        ax.set_ylim(lo, (20.2 if metric == "PSNR" else 0.9))
        ax.set_title(f"{metric}  —  200 labels (left) · 50 labels (right)", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    save(fig, "report_bars")


def gains_vs_input(m):
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    runs = ["supervised", "baseline", "ours", "ours_color", "supervised_l50", "baseline_l50", "ours_l50"]
    g = [m[r]["PSNR"] - m["input (no restoration)"]["PSNR"] for r in runs]
    y = np.arange(len(runs))[::-1]
    ax.barh(y, g, color=[COL[r] for r in runs], height=0.7)
    for yi, gi in zip(y, g):
        ax.text(gi + 0.03, yi, f"+{gi:.2f} dB", va="center", fontsize=7)
    ax.set_yticks(y, [("200 · " if "l50" not in r else "50 · ") + LAB[r] for r in runs], fontsize=7)
    ax.set_xlabel("PSNR gain over raw input (dB)")
    ax.set_xlim(0, 3.8)
    ax.grid(axis="y", visible=False)
    save(fig, "report_gain_vs_input")


def paired(ps):
    """Per-image ΔPSNR distributions for the key paired comparisons (sorted bars)."""
    pairs = [("baseline_l50", "supervised_l50", "Semi-UIR − Supervised\n(50 labels)", "p = 0.015"),
             ("ours_color", "baseline", "UA-Bank (colour) − Semi-UIR\n(200 labels)", "p = 0.185"),
             ("ours_l50", "baseline_l50", "UA-Bank − Semi-UIR\n(50 labels)", "p = 0.106")]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.4), sharey=True)
    for ax, (a, b, title, p) in zip(axes, pairs):
        d = np.sort(ps[a] - ps[b])
        ax.bar(np.arange(len(d)), d, color=[COL[a] if v > 0 else GREY for v in d], width=1.0)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.axhline(d.mean(), color="#0b0b0b", lw=1, ls="--")
        ax.set_title(f"{title}\nmean {d.mean():+.3f} dB · {np.mean(d > 0) * 100:.0f}% up · {p}", fontsize=6.6)
        ax.set_xticks([])
        ax.set_xlabel("90 test images (sorted)", fontsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("ΔPSNR per image (dB)")
    save(fig, "report_paired")


def bank_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
    for ax, runs, title in [(axes[0], ["baseline", "ours", "ours_color"], "200 labels"),
                            (axes[1], ["baseline_l50", "ours_l50"], "50 labels")]:
        for r in runs:
            rows = [x for x in log(r) if "bank_psnr" in x]
            ax.plot([x["iter"] for x in rows], [x["bank_psnr"] for x in rows], color=COL[r], label=LAB[r])
        if title == "50 labels":
            rows = [x for x in log("baseline_l50") if "bank_psnr" in x]
            pk = max(rows, key=lambda x: x["bank_psnr"])
            ax.annotate(f"peak {pk['bank_psnr']:.2f} dB", (pk["iter"], pk["bank_psnr"]), xytext=(4200, 19.5),
                        fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
            ax.annotate(f"final {rows[-1]['bank_psnr']:.2f} dB", (8000, rows[-1]["bank_psnr"]), xytext=(6000, 19.3),
                        fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
        ax.set_title(f"Bank pseudo-label PSNR vs hidden GT — {title}", fontsize=8)
        ax.set_xlabel("iteration")
        ax.legend(frameon=False, fontsize=7, loc="lower right" if title == "200 labels" else "lower left")
    axes[0].set_ylabel("PSNR (dB)")
    save(fig, "report_bank")


def student_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
    for ax, runs, title in [(axes[0], ["supervised", "baseline", "ours", "ours_color"], "200 labels"),
                            (axes[1], ["supervised_l50", "baseline_l50", "ours_l50"], "50 labels")]:
        for r in runs:
            rows = [x for x in log(r) if "val_psnr20" in x]
            ax.plot([x["iter"] for x in rows], [x["val_psnr20"] for x in rows], color=COL[r], label=LAB[r], lw=1.6)
        ax.set_title(f"Student PSNR during training (20 test imgs, monitoring only) — {title}", fontsize=7.5)
        ax.set_xlabel("iteration")
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("PSNR (dB)")
    save(fig, "report_student")


def training_dynamics():
    """Supervised loss (overfitting signal) and bank coverage, 50-label regime."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for r in ["supervised_l50", "baseline_l50", "ours_l50"]:
        rows = log(r)
        axes[0].plot([x["iter"] for x in rows], [x["sup"] for x in rows], color=COL[r], label=LAB[r], lw=1.5)
        cov = [x for x in rows if "bank_coverage" in x]
        if cov:
            axes[1].plot([x["iter"] for x in cov], [100 * x["bank_coverage"] for x in cov], color=COL[r], label=LAB[r])
    axes[0].set(title="Supervised loss on the 50 labeled pairs", xlabel="iteration", ylabel="loss")
    axes[1].set(title="Reliable-bank coverage (% of unlabeled pixels filled)", xlabel="iteration", ylabel="%")
    axes[0].legend(frameon=False, fontsize=7)
    for a in axes:
        a.title.set_fontsize(8)
    save(fig, "report_dynamics")


def uncertainty_charts():
    src = json.load(open("results/uncertainty_source.json"))
    gate = json.load(open("results/gate_accuracy.json"))
    unc = json.load(open("results/uncertainty.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.4))
    ax = axes[0]
    ax.bar([0, 1], [src["flip"], src["color"]], color=[OURS, OURS2], width=0.6)
    for i, v in enumerate([src["flip"], src["color"]]):
        ax.text(i, v, f"ρ = {v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1], ["flip views", "colour-cast\nperturbations"])
    ax.set_title("Spearman(U, pixel error)\nlabeled train images", fontsize=8)
    ax.set_ylim(0, 0.25)
    ax = axes[1]
    e = np.array(unc["err_by_uncertainty_decile"])
    ax.bar(np.arange(1, 11), e, color=[GREY] * 9 + [OURS], width=0.8)
    ax.set_ylim(0.08, 0.112)
    ax.set_xticks(range(1, 11))
    ax.set_title("Pseudo-label error by U decile\n(top decile +23%)", fontsize=8)
    ax.set_xlabel("uncertainty decile (low → high)", fontsize=7)
    ax = axes[2]
    vals = [100 * gate["uiqm_gate_accuracy"], 100 * gate["colour_uncertainty_gate_accuracy"]]
    ax.bar([0, 1], vals, color=[BASE, OURS2], width=0.6)
    ax.axhline(50, color=GREY, ls="--", lw=1)
    ax.text(1.45, 50.5, "chance", color=GREY, fontsize=7, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1], ["UIQM\n(Semi-UIR gate)", "colour-U"])
    ax.set_ylim(40, 75)
    ax.set_title(f"Gate accuracy: picks the truly\nbetter output ({gate['pairs']} pairs)", fontsize=8)
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "report_uncertainty")


@torch.no_grad()
def uncertainty_maps(images=("0005.png", "0030.png", "0061.png", "0012.png")):
    """Flip vs colour-cast uncertainty next to the true error of the teacher."""
    dev = get_device()
    ck = torch.load("checkpoints/ours_l50/final.pth", map_location="cpu")
    t = UNet(ck["cfg"]["width"]).to(dev).eval()
    t.load_state_dict(ck["teacher"])
    torch.manual_seed(0)
    cols = ["input", "teacher output", "|error| vs GT", "flip U", "colour-cast U"]
    fig, axes = plt.subplots(len(images), len(cols), figsize=(7.2, 1.15 * len(images)))
    for r, n in enumerate(images):
        x = torch.from_numpy(np.array(Image.open(f"data/UIEB/test/input/{n}"))).permute(2, 0, 1)[None].float().div(255).to(dev)
        y = torch.from_numpy(np.array(Image.open(f"data/UIEB/test/gt/{n}"))).permute(2, 0, 1)[None].float().div(255).to(dev)
        p = t(x).clamp(0, 1)
        _, uf = teacher_tta(t, x, 4, "flip")
        _, uc = teacher_tta(t, x, 4, "color", 0.15, "plain")
        ims = [x[0].permute(1, 2, 0).cpu(), p[0].permute(1, 2, 0).cpu(), (p - y).abs().mean(1)[0].cpu(),
               uf[0, 0].cpu(), uc[0, 0].cpu()]
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im.numpy(), cmap=None if im.ndim == 3 else "magma")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
    fig.tight_layout(pad=0.2)
    save(fig, "report_umaps")


def qualitative_full(images=("0005.png", "0030.png", "0061.png", "0012.png", "0047.png")):
    runs = ["supervised_l50", "baseline_l50", "ours_l50"]
    cols = ["Input"] + [LAB[r] for r in runs] + ["Ground truth"]
    fig, axes = plt.subplots(len(images), len(cols), figsize=(7.2, 1.2 * len(images)))
    for r, n in enumerate(images):
        ims = [Image.open(f"data/UIEB/test/input/{n}")] + [Image.open(f"results/outputs/{k}/{n}") for k in runs] + \
              [Image.open(f"data/UIEB/test/gt/{n}")]
        gt = np.array(ims[-1])
        for c, im in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if 0 < c < len(cols) - 1 or c == 0:
                ax.set_xlabel(f"{peak_signal_noise_ratio(gt, np.array(im), data_range=255):.2f} dB", fontsize=7)
            if r == 0:
                ax.set_title(cols[c], fontsize=8)
    fig.tight_layout(pad=0.3)
    save(fig, "report_qualitative")


def per_image_scatter(ps):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    a, b = ps["supervised_l50"], ps["baseline_l50"]
    ax.scatter(a, b, s=10, color=BASE, alpha=0.75, edgecolor="none")
    lo, hi = min(a.min(), b.min()) - 0.5, max(a.max(), b.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1, ls="--")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Supervised PSNR (dB)", ylabel="Semi-UIR PSNR (dB)")
    ax.set_title("Per-image PSNR, 50 labels\n(points above diagonal: Semi-UIR wins)", fontsize=8)
    save(fig, "report_scatter")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    m = metrics()
    ps = {r: per_image(r) for r in ["supervised", "baseline", "ours", "ours_color",
                                     "supervised_l50", "baseline_l50", "ours_l50"]}
    bars_main(m)
    gains_vs_input(m)
    paired(ps)
    per_image_scatter(ps)
    bank_curves()
    student_curves()
    training_dynamics()
    uncertainty_charts()
    uncertainty_maps()
    qualitative_full()


@torch.no_grad()
def onepage_strip(n="0005.png"):
    """Compact 4-panel infographic for the one-page template report."""
    import matplotlib.gridspec as gs
    fig = plt.figure(figsize=(7.6, 1.35))
    g = gs.GridSpec(1, 5, width_ratios=[1.25, 0.75, 1.1, 1.1, 1.1], wspace=0.45)
    ax = fig.add_subplot(g[0])
    for r in ["baseline_l50", "ours_l50"]:
        rows = [x for x in log(r) if "bank_psnr" in x]
        ax.plot([x["iter"] / 1000 for x in rows], [x["bank_psnr"] for x in rows], color=COL[r], lw=1.6,
                label="Semi-UIR" if r == "baseline_l50" else "UA-Bank")
    ax.set_title("(a) Bank PSNR, 50 labels", fontsize=6.5)
    ax.set_xlabel("iteration (k)", fontsize=6); ax.tick_params(labelsize=5.5)
    ax.legend(frameon=False, fontsize=5.5, loc="lower center")
    src = json.load(open("results/uncertainty_source.json"))
    ax = fig.add_subplot(g[1])
    ax.bar([0, 1], [src["flip"], src["color"]], color=[GREY, OURS], width=0.65)
    ax.set_xticks([0, 1], ["flip", "colour"], fontsize=6); ax.tick_params(labelsize=5.5)
    ax.set_title("(b) ρ(U, error)", fontsize=6.5); ax.grid(axis="x", visible=False)
    dev = get_device()
    ck = torch.load("checkpoints/ours/final.pth", map_location="cpu")
    t = UNet(ck["cfg"]["width"]).to(dev).eval(); t.load_state_dict(ck["teacher"])
    x = torch.from_numpy(np.array(Image.open(f"data/UIEB/test/input/{n}"))).permute(2, 0, 1)[None].float().div(255).to(dev)
    _, u = teacher_tta(t, x, 4, "flip")
    ims = [(np.array(Image.open(f"results/outputs/baseline/{n}")), "(c) Semi-UIR · 23.32 dB"),
           (np.array(Image.open(f"results/outputs/ours/{n}")), "(d) UA-Bank · 23.95 dB"),
           (u[0, 0].cpu().numpy(), "(e) U flags the artefact")]
    for i, (im, title) in enumerate(ims):
        ax = fig.add_subplot(g[2 + i])
        ax.imshow(im[70:200, 170:330], cmap=None if im.ndim == 3 else "magma")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(title, fontsize=6.5)
    fig.savefig(f"{FIG}/report_onepage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{FIG}/report_onepage.png")
