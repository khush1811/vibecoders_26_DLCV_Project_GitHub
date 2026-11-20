"""Residual U-Net and the scene-conditioned restoration model built from it."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=32, in_channels=3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.inc = ConvBlock(in_channels, c[0])
        self.down = nn.ModuleList([ConvBlock(c[i], c[i + 1]) for i in range(3)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(3))])
        self.dec = nn.ModuleList([ConvBlock(c[i] * 2, c[i]) for i in reversed(range(3))])
        self.out = nn.Conv2d(c[0], 3, 3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        feats = [self.inc(xp)]
        for d in self.down:
            feats.append(d(F.max_pool2d(feats[-1], 2)))
        y = feats.pop()
        for up, dec in zip(self.up, self.dec):
            y = dec(torch.cat([up(y), feats.pop()], 1))
        # global residual: predict a correction to the degraded input
        return (xp[:, :3] + self.out(y))[..., :h, :w]


class SceneColorUNet(nn.Module):
    """Image-conditioned tone correction followed by lightweight local refinement."""

    def __init__(self, base=16):
        super().__init__()
        self.register_buffer("percentiles", torch.tensor([0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]))
        self.register_buffer("knots", torch.linspace(0, 1, 5).view(1, 1, 5, 1, 1))
        self.global_color = nn.Sequential(nn.Linear(27, 64), nn.SiLU(), nn.Linear(64, 24))
        self.refine = UNet(base, in_channels=9)
        nn.init.zeros_(self.global_color[-1].weight)
        nn.init.zeros_(self.global_color[-1].bias)
        nn.init.zeros_(self.refine.out.weight)
        nn.init.zeros_(self.refine.out.bias)

    def forward(self, x):
        # Statistics of the full frame are available at the same scale in training and inference.
        thumbnail_input = x
        if x.device.type == "mps" and any(size % 48 for size in x.shape[-2:]):
            # Older MPS builds require the input to be divisible by the output for area pooling.
            thumbnail_input = x.cpu()
        pixels = F.interpolate(thumbnail_input, (48, 48), mode="area").to(x.device).float().flatten(2)
        ordered = pixels.sort(dim=-1).values
        ranks = (self.percentiles * (ordered.shape[-1] - 1)).long()
        quantiles = ordered.index_select(-1, ranks)
        low, high = quantiles[..., :1, None], quantiles[..., -1:, None]
        stretch = ((x - low) / (high - low).clamp_min(0.15)).clamp(0, 1)
        stats = torch.cat((quantiles, pixels.mean(-1, keepdim=True), pixels.std(-1, keepdim=True)), dim=-1)
        adjustment = self.global_color(stats.flatten(1)).tanh()
        matrix = adjustment[:, :9].reshape(-1, 3, 3) * 0.3
        curve = adjustment[:, 9:].reshape(-1, 3, 5, 1, 1) * 0.4
        basis = (1 - (stretch.unsqueeze(2) - self.knots).abs() * 4).clamp_min(0)
        tone = (curve * basis).sum(2)
        color = torch.einsum("bij,bjhw->bihw", matrix, stretch - 0.5)
        corrected = 0.5 * (x + stretch) + tone + color
        return self.refine(torch.cat((corrected, x, stretch), dim=1))


def build_model(cfg):
    architecture = cfg.get("architecture", "unet")
    if architecture == "unet":
        return UNet(cfg["width"])
    if architecture == "scene_color_unet":
        return SceneColorUNet(cfg["width"])
    raise ValueError(f"Unknown model architecture {architecture!r}")


if __name__ == "__main__":
    m = UNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
