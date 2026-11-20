"""Fit and evaluate a color calibration on frozen UIEB restoration checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval import COMPARISON_FIELDS, METRICS, paired_comparison, write_csv
from models.unet import build_model
from semiuir.calibration import color_features, restore_views, select_calibration
from semiuir.data import Paired
from semiuir.metrics import psnr_ssim, uciqe, uiqm
from train import get_device


DEFAULT_CHECKPOINTS = [f"checkpoints/{name}/final.pth"
                       for name in ("baseline_l50", "ours_l50", "supervised_l50")]


def checkpoint_data_root(path):
    root = Path(path)
    if root == Path("data/UIEB_L50"):
        return Path("data/UIEB/L50")
    return root


def load_models(paths, device):
    models, configs = [], []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["cfg"]
        if configs and checkpoint_data_root(config["data_root"]) != checkpoint_data_root(configs[0]["data_root"]):
            raise ValueError(f"Checkpoint {path} belongs to a different labeled split.")
        model = build_model(config).to(device).eval()
        model.load_state_dict(checkpoint["student"])
        models.append(model)
        configs.append(config)
    return models, configs[0]


def fit(args):
    device = get_device()
    models, reference = load_models(args.ckpt, device)
    data_root = args.data_root or checkpoint_data_root(reference["data_root"]).as_posix()
    dataset = Paired(Path(data_root) / "labeled")
    examples = []
    print(f"Fitting on {len(dataset)} labeled scenes using {device}", flush=True)
    for index, (image, gt, _) in enumerate(dataset, 1):
        image = image.to(device)
        predictions, uncertainty = restore_views(models, image[None], reference["crop"])
        prediction = predictions["confidence_ensemble"]
        stride = 4
        design = color_features(image, prediction, uncertainty, stride)
        # Avoid caching every forward pass during the scene-grouped fit.
        candidates = {key: value[:, ::stride, ::stride].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
                      for key, value in predictions.items()}
        targets = gt[:, ::stride, ::stride].permute(1, 2, 0).reshape(-1, 3).numpy()
        examples.append((design.flatten(0, 1).cpu().numpy(), candidates["confidence_ensemble"],
                         targets, candidates))
        if index % 10 == 0:
            print(f"Prepared {index}/{len(dataset)} labeled scenes", flush=True)

    result = select_calibration(examples, args.folds, args.seed)
    model = {"data_root": data_root, "checkpoints": args.ckpt, "reference": reference["name"],
             "crop_size": reference["crop"], "fit_split": "labeled", "cv_seed": args.seed,
             "labeled_images": dataset.names, **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    for method, score in result["cv_psnr"].items():
        print(f"{method:>23s}  calibration CV PSNR {score:.4f}")
    print(f"Selected {result['selected']} using {len(dataset)} labeled scenes. Saved {args.output}.")


def scores(prediction, gt):
    prediction = prediction.cpu().clamp(0, 1)
    return (*psnr_ssim(prediction, gt), float(uiqm(prediction[None])[0]), uciqe(prediction))


def evaluate(args):
    spec = json.loads(args.calibration.read_text(encoding="utf-8"))
    device = get_device()
    models, reference = load_models(spec["checkpoints"], device)
    data_root = Path(args.data_root) if args.data_root else checkpoint_data_root(spec["data_root"])
    dataset = Paired(data_root / "test")
    coefficients = spec["coefficients"]
    coeffs = torch.tensor(coefficients, device=device) if coefficients is not None else None
    heldout_scores = {}
    best_key = reference["name"] if spec["selected"] == "baseline" else spec["selected"]
    print(f"Evaluating {len(dataset)} held-out scenes using {device}", flush=True)
    for index, (image, gt, name) in enumerate(dataset, 1):
        image = image.to(device)
        guesses, uncertainty = restore_views(models, image[None], spec["crop_size"])
        if coeffs is not None:
            estimate = guesses["confidence_ensemble"]
            delta = color_features(image, estimate, uncertainty) @ coeffs
            guesses[spec["selected"]] = estimate + delta.permute(2, 0, 1)
        guesses[reference["name"]] = guesses.pop("baseline")
        for method, pred in guesses.items():
            heldout_scores.setdefault(method, {})[name] = scores(pred, gt)
        if args.save_dir:
            path = args.save_dir / best_key / name
            path.parent.mkdir(parents=True, exist_ok=True)
            arr = guesses[best_key].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(arr).save(path)
        if index % 20 == 0:
            print(f"Scored {index}/{len(dataset)} held-out scenes", flush=True)

    overall, by_image = [], []
    for method, image_scores in heldout_scores.items():
        averaged = np.mean(list(image_scores.values()), axis=0)
        overall.append((method, *averaged))
        by_image.extend((method, name, *values) for name, values in image_scores.items())
        print(f"{method:>23s}  PSNR {averaged[0]:.4f}  SSIM {averaged[1]:.4f}")
    write_csv(args.csv, ("method", *METRICS), overall)
    write_csv(args.per_image_csv, ("method", "image", *METRICS), by_image)
    comparisons = [(method, reference["name"], *row)
                   for method, values in heldout_scores.items() if method != reference["name"]
                   for row in paired_comparison(values, heldout_scores[reference["name"]])]
    if coeffs is not None:
        comparisons.extend((best_key, "confidence_ensemble", *row) for row in paired_comparison(
            heldout_scores[best_key], heldout_scores["confidence_ensemble"]))
    write_csv(args.comparison_csv, COMPARISON_FIELDS, comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="intra-op threads on the CPU")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="fit on labeled scenes and select by grouped CV")
    fitting.add_argument("--ckpt", nargs="+", default=DEFAULT_CHECKPOINTS)
    fitting.add_argument("--data-root", help="override the checkpoint's dataset path with a copy of the same split")
    fitting.add_argument("--folds", type=int, default=5)
    fitting.add_argument("--seed", type=int, default=0)
    fitting.add_argument("--output", type=Path, default=Path("checkpoints/calibrated_l50/calibration.json"))
    inference = commands.add_parser("eval", help="score a saved calibration on the untouched test split")
    inference.add_argument("--calibration", type=Path, default=Path("checkpoints/calibrated_l50/calibration.json"))
    inference.add_argument("--data-root", help="override the calibration's dataset path with a copy of the same split")
    inference.add_argument("--csv", type=Path, default=Path("results/calibrated_l50.csv"))
    inference.add_argument("--per-image-csv", type=Path, default=Path("results/calibrated_l50_by_image.csv"))
    inference.add_argument("--comparison-csv", type=Path, default=Path("results/calibrated_l50_paired.csv"))
    inference.add_argument("--save-dir", type=Path)

    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "eval":
        outputs = {path.resolve() for path in (args.csv, args.per_image_csv, args.comparison_csv)}
        if len(outputs) != 3 or args.calibration.resolve() in outputs:
            parser.error("choose distinct paths for calibration and evaluation outputs")
    torch.set_num_threads(args.threads)
    (fit if args.command == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()
