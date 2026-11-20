"""Score restoration checkpoints and optionally compare matched images.

    python eval.py --ckpt checkpoints/ours/final.pth --save_dir results/outputs
    python eval.py --ckpt checkpoints/baseline/final.pth checkpoints/ours/final.pth \
        --per_image_csv results/per_image.csv --compare ours baseline
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon
from torch.utils.data import Subset

from models.unet import build_model
from semiuir.data import Paired
from semiuir.metrics import psnr_ssim, uciqe, uiqm
from train import get_device

METRICS = ("PSNR", "SSIM", "UIQM", "UCIQE")
INPUT = "input (no restoration)"
COMPARISON_FIELDS = ("candidate", "reference", "metric", "n", "delta_mean",
                     "ci95_low", "ci95_high", "win_rate", "wilcoxon_p")


@torch.no_grad()
def score_images(model, ds, dev, save_dir=None):
    """Compute metrics on float predictions, keyed by source filename."""
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for a, b, name in ds:
        out = a if model is None else model(a[None].to(dev))[0].clamp(0, 1).cpu()
        p, s = psnr_ssim(out, b)
        rows[name] = (p, s, uiqm(out[None])[0].item(), uciqe(out))
        if save_dir is not None:
            arr = (out.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(save_dir / name)
    if not rows:
        raise ValueError("The evaluation split has no paired images")
    return rows


def evaluate(model, ds, dev, save_dir=None):
    """Return mean PSNR, SSIM, UIQM and UCIQE for an existing caller."""
    return np.asarray(list(score_images(model, ds, dev, save_dir).values())).mean(axis=0)


def paired_comparison(candidate_scores, reference_scores, samples=10000, seed=0):
    """Compare two runs by resampling whole, matched test images with replacement."""
    if len(candidate_scores) == 0 or candidate_scores.keys() != reference_scores.keys():
        raise ValueError("A paired comparison requires the same nonempty set of image names")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")

    names = sorted(candidate_scores)
    candidate = np.asarray([candidate_scores[name] for name in names], dtype=np.float64)
    reference = np.asarray([reference_scores[name] for name in names], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        # A perfect prediction has infinite PSNR. Two identical scores are a tie.
        deltas = np.where(candidate == reference, 0.0, candidate - reference)
    indices = np.random.default_rng(seed).integers(0, len(names), size=(samples, len(names)))

    rows = []
    for metric, delta in zip(METRICS, deltas.T):
        if np.isfinite(delta).all():
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        else:
            low, high = np.nan, np.nan
        with np.errstate(invalid="ignore"):
            mean = delta.mean()
        p_value = wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0
        rows.append((metric, len(delta), mean, low, high, (delta > 0).mean(), p_value))
    return rows


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    print(f"wrote {path}")


def paired_subset(dataset, names_path):
    names = [line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("The image list must contain unique, nonempty filenames")
    indices = {name: i for i, name in enumerate(dataset.names)}
    missing = set(names).difference(indices)
    if missing:
        raise ValueError(f"Image list contains files outside this split: {', '.join(sorted(missing))}")
    return Subset(dataset, [indices[name] for name in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--data_root", type=Path, default=Path("data/UIEB"))
    ap.add_argument("--split", default="test", help="paired directory under data_root, e.g. test or labeled")
    ap.add_argument("--names", type=Path, help="score only filenames in this one-per-line file")
    ap.add_argument("--csv", type=Path, help="write mean metrics by method (legacy default: results/metrics.csv)")
    ap.add_argument("--per_image_csv", type=Path, help="write unrounded float metrics by method and filename")
    ap.add_argument("--compare", nargs=2, action="append", metavar=("CANDIDATE", "REFERENCE"),
                    default=[], help="report paired deltas, bootstrapped 95%% CIs and Wilcoxon p-values")
    ap.add_argument("--comparison_csv", type=Path, help="write the requested paired comparisons")
    ap.add_argument("--save_dir", type=Path, help="save restored images (one sub-folder per run)")
    ap.add_argument("--with_input", action="store_true", help="also score the raw inputs")
    ap.add_argument("--weights", choices=("student", "teacher"),
                    help="override each checkpoint's eval_weights setting (default student)")
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), help="override the automatically chosen device")
    ap.add_argument("--threads", type=int, help="set the number of intra-op CPU threads")
    args = ap.parse_args()
    if args.comparison_csv and not args.compare:
        ap.error("--comparison_csv requires at least one --compare pair")
    if args.threads is not None:
        if args.threads < 1:
            ap.error("--threads must be positive")
        torch.set_num_threads(args.threads)

    summary_csv = args.csv
    if (summary_csv is None and args.split == "test" and args.names is None
            and args.per_image_csv is None and not args.compare):
        summary_csv = Path("results/metrics.csv")
    paths = [path.resolve() for path in (summary_csv, args.per_image_csv, args.comparison_csv) if path is not None]
    if len(paths) != len(set(paths)):
        ap.error("Choose distinct paths for the summary, per-image and comparison CSVs")

    dev = torch.device(args.device) if args.device else get_device()
    ds = Paired(args.data_root / args.split)
    if args.names is not None:
        ds = paired_subset(ds, args.names)

    results = []
    scores = {}
    if args.with_input:
        scores[INPUT] = score_images(None, ds, dev)
        results.append((INPUT, *np.asarray(list(scores[INPUT].values())).mean(axis=0), 0.0))
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        name = ck["cfg"]["name"]
        if name in scores:
            ap.error(f"Checkpoint method name {name!r} was supplied more than once")
        model = build_model(ck["cfg"]).to(dev).eval()
        weights = args.weights or ck["cfg"].get("eval_weights", "student")
        if weights not in ("student", "teacher"):
            ap.error(f"Invalid checkpoint eval_weights setting for {name!r}: {weights!r}")
        model.load_state_dict(ck[weights])
        save_dir = args.save_dir / name if args.save_dir else None
        scores[name] = score_images(model, ds, dev, save_dir)
        m = np.asarray(list(scores[name].values())).mean(axis=0)
        results.append((name, *m, ck["train_minutes"]))
        print(f"{name:>14s} ({weights})  PSNR {m[0]:.3f}  SSIM {m[1]:.4f}  "
              f"UIQM {m[2]:.3f}  UCIQE {m[3]:.4f}", flush=True)

    comparisons = []
    for candidate, reference in args.compare:
        if candidate not in scores or reference not in scores:
            ap.error(f"Comparison requires both methods to be evaluated: {candidate}, {reference}")
        for metric, n, delta, low, high, win, p in paired_comparison(scores[candidate], scores[reference]):
            comparisons.append((candidate, reference, metric, n, delta, low, high, win, p))
            print(f"{candidate} vs {reference}  {metric} {delta:+.4f}  "
                  f"95% CI [{low:+.4f}, {high:+.4f}]  win {win:.1%}  Wilcoxon p={p:.4g}  n={n}")

    if summary_csv is not None:
        write_csv(summary_csv, ("method", *METRICS, "train_minutes"),
                  ((name, *(f"{v:.4f}" for v in values)) for name, *values in results))
    if args.per_image_csv is not None:
        write_csv(args.per_image_csv, ("method", "image", *METRICS),
                  ((method, image, *metrics) for method, rows in scores.items() for image, metrics in rows.items()))
    if args.comparison_csv is not None:
        write_csv(args.comparison_csv, COMPARISON_FIELDS, comparisons)


if __name__ == "__main__":
    main()
