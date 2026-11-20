"""Calibrate the color of frozen restoration models using labeled scenes."""

import numpy as np
import torch
import torch.nn.functional as F

from .uncertainty import VIEWS


UNCALIBRATED = ("baseline", "flip_scale_ensemble", "confidence_ensemble")
RIDGE_STRENGTHS = (0.0003, 0.003, 0.03, 0.3)


@torch.inference_mode()
def restore_views(models, image, crop_size):
    """Return each inference baseline, confidence consensus, and disagreement map.

    The small view operates near the training crop size. Transfer its restoration
    residual to the full-resolution input to preserve the original image detail.
    """
    height, width = image.shape[-2:]
    small_shape = tuple(round(size * crop_size / min(height, width)) for size in (height, width))
    # Use the same antialiased resize on CPU, CUDA, and MPS.
    small = F.interpolate(image.cpu(), size=small_shape, mode="bilinear", align_corners=False,
                          antialias=True).to(image.device)
    means, variances = [], []
    for model in models:
        views = [model(image)]
        for dims in VIEWS[1:]:
            views.append(model(image.flip(dims)).flip(dims))
        residual = F.interpolate(model(small) - small, size=(height, width), mode="bilinear",
                                 align_corners=False)
        views.append(image + residual)
        batch = torch.cat(views).clamp(0, 1)
        mean = batch.mean(dim=0)
        means.append(mean)
        variances.append(batch.var(dim=0, unbiased=False))
        if len(means) == 1:
            baseline = batch[0]

    means = torch.stack(means)
    variances = torch.stack(variances)
    raw_uncertainty = variances.mean(dim=1, keepdim=True).sqrt()
    uncertainty = F.avg_pool2d(raw_uncertainty, kernel_size=9, stride=1, padding=4,
                               count_include_pad=False)
    temperature = uncertainty.mean().clamp_min(0.01)
    confidence = torch.softmax(-uncertainty / temperature, dim=0)
    uniform = means.mean(dim=0)
    consensus = (confidence * means).sum(dim=0)
    disagreement = (variances + (means - consensus).square()).mean(dim=(0, 1), keepdim=False)
    predictions = dict(zip(UNCALIBRATED, (baseline, uniform, consensus)))
    return predictions, disagreement.clamp_min(0).sqrt()


def color_features(image, prediction, disagreement, stride=1):
    """Shared inference and fitting features for an RGB color residual.

    Scene statistics estimate the global cast. Centered RGB bases estimate tone
    and channel mixing, and a disagreement interaction can adjust how far the
    enhancement moves away from the original input in uncertain regions.
    """
    def mean(x):
        return x.mean(dim=(-2, -1), keepdim=True)

    input_mean, pred_mean = mean(image), mean(prediction)
    input_centered, pred_centered = image - input_mean, prediction - pred_mean
    input_std, pred_std = mean(input_centered.square()).sqrt(), mean(pred_centered.square()).sqrt()
    tone = pred_centered.square() - pred_std.square()
    u = disagreement[None]
    u_mean = mean(u)
    local = (input_centered, pred_centered, tone, (image - prediction) * (u / (u_mean + 0.01)).tanh())
    features = torch.cat([t[:, ::stride, ::stride] for t in local])
    scene = torch.cat([torch.ones_like(u_mean), input_mean - 0.5, pred_mean - input_mean,
                       input_std, pred_std - input_std, u_mean])
    return torch.cat([features, scene.expand(-1, *features.shape[-2:])]).permute(1, 2, 0)


def fit_ridge(gram, target, strength):
    """Solve for a per-pixel RGB residual with equal weight for each scene."""
    regularizer = np.eye(len(gram)) * strength
    regularizer[12, 12] = 0  # scene intercept follows the four local RGB features
    return np.linalg.solve(gram + regularizer, target)


def _psnr(pred, gt):
    return -10 * np.log10(max(np.mean((np.clip(pred, 0, 1) - gt) ** 2), 1e-12))


def select_calibration(examples, folds=5, seed=0):
    """Select the inference variant by scene-grouped labeled calibration CV."""
    if not 2 <= folds <= len(examples):
        raise ValueError("Calibration needs at least one labeled image per fold and two folds.")
    groups = np.array_split(np.random.default_rng(seed).permutation(len(examples)), folds)
    grams = np.stack([x.T @ x / len(x) for x, _, _, _ in examples])
    targets = np.stack([x.T @ (target - pred) / len(x) for x, pred, target, _ in examples])
    scores = {method: [] for method in UNCALIBRATED}
    strengths = {f"color_ridge_{value:g}": value for value in RIDGE_STRENGTHS}
    scores.update({method: [] for method in strengths})
    for i, group in enumerate(groups):
        fit_indices = np.concatenate([g for j, g in enumerate(groups) if i != j])
        gram, target = grams[fit_indices].mean(axis=0), targets[fit_indices].mean(axis=0)
        weights = {method: fit_ridge(gram, target, value) for method, value in strengths.items()}
        for j in group:
            features, prediction, gt, precalibrated = examples[j]
            for method, estimate in precalibrated.items():
                scores[method].append(_psnr(estimate, gt))
            for method, coeffs in weights.items():
                scores[method].append(_psnr(prediction + features @ coeffs, gt))

    means = {method: float(np.mean(values)) for method, values in scores.items()}
    selected = max(means, key=means.get)
    coeffs = (fit_ridge(grams.mean(axis=0), targets.mean(axis=0), strengths[selected]).tolist()
              if selected in strengths else None)
    return {"selected": selected, "coefficients": coeffs, "cv_psnr": means,
            "fold_indices": [group.tolist() for group in groups]}
