"""Calibrate pseudo-label fidelity on the labeled scenes and score both bank candidates alike."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bank import ReliableBank


class FidelityRisk(nn.Module):
    """Predict log patch RMSE from aligned input, restoration, and teacher disagreement."""

    def __init__(self, tile=8):
        super().__init__()
        self.tile = tile
        self.regress = nn.Sequential(nn.Conv2d(16, 24, 1), nn.SiLU(), nn.Conv2d(24, 1, 1))

    def forward(self, raw, candidate, uncertainty):
        diff = candidate - raw
        local = F.avg_pool2d(torch.cat((raw, candidate, diff.abs(), uncertainty), dim=1), self.tile)
        context = torch.cat((raw, candidate), dim=1).mean(dim=(-2, -1), keepdim=True)
        context = context.expand(-1, -1, *local.shape[-2:])
        return self.regress(torch.cat((local, context), dim=1))

    def loss(self, raw, prediction, uncertainty, target):
        squared_error = (prediction - target).square().mean(dim=1, keepdim=True)
        patch_rmse = F.avg_pool2d(squared_error, self.tile).sqrt()
        return F.smooth_l1_loss(self(raw, prediction, uncertainty), (patch_rmse + 0.01).log())


class FidelityBank(ReliableBank):
    """Store an entire aligned frame when the latest calibrator predicts a lower error."""

    def __init__(self, n, size, risk, device, margin=0.03):
        super().__init__(n, size, device=device)
        self.risk = risk
        self.margin = margin

    @torch.no_grad()
    def update(self, idx, yx, raw, prediction, uncertainty):
        prediction = (prediction.clamp(0, 1) * 255).round() / 255
        prev, old_uncertainty, filled = self._get(idx, yx, raw.shape[-1])
        new_error = self.risk(raw, prediction, uncertainty).exp()
        old_error = self.risk(raw, prev, old_uncertainty.masked_fill(~filled, 0)).exp()
        new_risk = new_error.square().mean(dim=(-2, -1), keepdim=True)
        old_risk = old_error.square().mean(dim=(-2, -1), keepdim=True)
        take = ~filled[..., :1, :1] | (new_risk < old_risk * (1 - self.margin))
        pseudo = torch.where(take, prediction, prev)
        uncertainty = torch.where(take, uncertainty, old_uncertainty)
        self._put(idx, yx, raw.shape[-1], pseudo, uncertainty, filled | take)

        estimated_error = torch.where(take, new_error, old_error)
        # Downweight uncertain patches and let low-confidence frames contribute less overall.
        weights = (0.08 / estimated_error).clamp(0.15, 1)
        weights = F.interpolate(weights, size=raw.shape[-2:], mode="bilinear", align_corners=False)
        return pseudo, weights, take.float().mean()
