"""Losses of Semi-UIR, with an optional per-pixel weight map for the unsupervised terms.

A single truncated VGG-16 (relu1_2, relu2_2, relu3_3) is shared by the perceptual loss and the
contrastive loss to keep the CPU cost low (official code: VGG-16 perceptual + VGG-19 contrastive).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        f = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        self.slices = nn.ModuleList([f[:4], f[4:9], f[9:16]])  # relu1_2, relu2_2, relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = []
        for s in self.slices:
            x = s(x)
            out.append(x)
        return out


def gradient(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def supervised_loss(vgg, out, gt):
    """0.8 L1 + 0.2 L2 + 0.3 perceptual + 0.1 gradient (official weights)."""
    struct = 0.8 * F.l1_loss(out, gt) + 0.2 * F.mse_loss(out, gt)
    per = sum(F.mse_loss(a, b) for a, b in zip(vgg(out), vgg(gt))) / 3
    (ox, oy), (gx, gy) = gradient(out), gradient(gt)
    grad = F.l1_loss(ox, gx) + F.l1_loss(oy, gy)
    return struct + 0.3 * per + 0.1 * grad


def weighted_l1(a, b, w):
    """mean over pixels of w * |a - b|, normalised by sum(w). w: (B,1,h,w), resized if needed."""
    if w.shape[-2:] != a.shape[-2:]:
        w = F.interpolate(w, size=a.shape[-2:], mode="area")
    num = (w * (a - b).abs()).sum()
    return num / (w.sum() * a.shape[1] + 1e-8)


class ContrastLoss(nn.Module):
    """Semi-UIR contrastive regularisation:  sum_l  lambda_l * d(a,p) / d(a,n)
    anchor a = student output, positive p = reliable-bank pseudo-label, negative n = degraded input.
    [UA-Bank] the anchor-positive distance is confidence-weighted per pixel via `w`.
    """

    weights = [1 / 4, 1 / 2, 1.0]

    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg

    def forward(self, a, p, n, w):
        fa, fp, fn = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        for lam, xa, xp, xn in zip(self.weights, fa, fp, fn):
            d_ap = weighted_l1(xa, xp.detach(), w)
            d_an = weighted_l1(xa, xn.detach(), w)
            loss = loss + lam * d_ap / (d_an + 1e-7)
        return loss
