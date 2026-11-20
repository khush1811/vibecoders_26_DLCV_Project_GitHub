"""Image-quality metrics.

* PSNR / SSIM : full-reference (skimage), on [0,1] RGB.
* UIQM        : no-reference underwater metric (Panetta et al. 2016), batched torch implementation.
                Used (i) as an evaluation metric and (ii) as the cheap reliability score that
                replaces MUSIQ in the reliable bank (CPU-feasible).
* UCIQE       : no-reference underwater metric (Yang & Sowmya 2015), numpy implementation.
"""
import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.color import rgb2lab
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ModuleNotFoundError as exc:
    if exc.name != "skimage":
        raise
    from scipy.ndimage import uniform_filter

    def peak_signal_noise_ratio(gt, pred, *, data_range):
        mse = np.mean(np.square(np.subtract(gt, pred, dtype=np.float64)))
        return float(10 * np.log10(data_range ** 2 / mse)) if mse else float("inf")

    def structural_similarity(gt, pred, *, data_range, channel_axis):
        # Match the default 7x7 uniform window and sample covariance used by skimage.
        window = 7
        pad = window // 2
        if min(gt.shape[:2]) < window or channel_axis != 2:
            raise ValueError("SSIM expects RGB images at least 7 pixels wide and high")

        kernel = (window, window, 1)
        mean_gt = uniform_filter(gt, size=kernel)
        mean_pred = uniform_filter(pred, size=kernel)
        sample_correction = window ** 2 / (window ** 2 - 1)
        var_gt = sample_correction * (uniform_filter(gt ** 2, size=kernel) - mean_gt ** 2)
        var_pred = sample_correction * (uniform_filter(pred ** 2, size=kernel) - mean_pred ** 2)
        cov = sample_correction * (uniform_filter(gt * pred, size=kernel) - mean_gt * mean_pred)
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        ssim = ((2 * mean_gt * mean_pred + c1) * (2 * cov + c2)
                / ((mean_gt ** 2 + mean_pred ** 2 + c1) * (var_gt + var_pred + c2)))
        return float(ssim[pad:-pad, pad:-pad].mean(dtype=np.float64))

    def rgb2lab(rgb):
        matrix = np.array([[0.412453, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                           [0.019334, 0.119193, 0.950227]], dtype=rgb.dtype)
        linear_rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        xyz = (linear_rgb @ matrix.T) / np.array([0.95047, 1.0, 1.08883], dtype=rgb.dtype)
        xyz = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
        x, y, z = np.moveaxis(xyz, -1, 0)
        return np.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), axis=-1)


def psnr_ssim(pred, gt):
    """pred, gt: float tensors CHW in [0,1]."""
    p = pred.clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float64)
    g = gt.clamp(0, 1).permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return (peak_signal_noise_ratio(g, p, data_range=1.0),
            structural_similarity(g, p, data_range=1.0, channel_axis=2))


# ---------------------------------------------------------------- UIQM (batched, torch)
def _trimmed_mean(x, alpha=0.1):
    # x: (B, N); alpha-trimmed mean, alpha fraction removed on each side
    n = x.shape[1]
    k = int(alpha * n)
    xs, _ = x.sort(dim=1)
    return xs[:, k:n - k].mean(1)


def _uicm(img):
    r, g, b = img[:, 0].flatten(1), img[:, 1].flatten(1), img[:, 2].flatten(1)
    rg, yb = r - g, (r + g) / 2 - b
    mu_rg, mu_yb = _trimmed_mean(rg), _trimmed_mean(yb)
    var_rg = ((rg - mu_rg[:, None]) ** 2).mean(1)
    var_yb = ((yb - mu_yb[:, None]) ** 2).mean(1)
    return -0.0268 * torch.sqrt(mu_rg ** 2 + mu_yb ** 2) + 0.1586 * torch.sqrt(var_rg + var_yb)


def _blocks(x, bs):
    # x: (B, H, W) -> (B, nblocks, bs*bs), cropping to a multiple of bs
    b, h, w = x.shape[0], (x.shape[1] // bs) * bs, (x.shape[2] // bs) * bs
    x = x[:, :h, :w].reshape(b, h // bs, bs, w // bs, bs).permute(0, 1, 3, 2, 4)
    return x.reshape(b, -1, bs * bs)


def _eme(x, bs=10):
    blk = _blocks(x, bs)
    mx, mn = blk.amax(-1), blk.amin(-1)
    valid = mn > 0
    val = torch.where(valid, torch.log(mx.clamp_min(1e-6) / mn.clamp_min(1e-6)), torch.zeros_like(mx))
    return 2.0 * val.mean(1)


def _uism(img):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device)
    k = torch.stack([kx, kx.t()])[:, None]
    out = 0
    for c, wgt in zip(range(3), (0.299, 0.587, 0.114)):
        ch = img[:, c:c + 1]
        gr = F.conv2d(F.pad(ch, (1, 1, 1, 1), mode="replicate"), k)
        edge = torch.hypot(gr[:, 0], gr[:, 1]) * ch[:, 0] / 255.0  # edge map scaled by channel
        out = out + wgt * _eme(edge)
    return out


def _uiconm(img, bs=10):
    gray = img.mean(1)
    blk = _blocks(gray, bs)
    mx, mn = blk.amax(-1), blk.amin(-1)
    top, bot = mx - mn, mx + mn
    ratio = torch.where(bot > 0, top / bot.clamp_min(1e-6), torch.zeros_like(top))
    val = torch.where(ratio > 0, ratio * torch.log(ratio.clamp_min(1e-6)), torch.zeros_like(ratio))
    return val.mean(1).abs()


@torch.no_grad()
def uiqm(x):
    """x: (B,3,H,W) in [0,1]  ->  (B,) UIQM scores (higher is better)."""
    img = x.float().clamp(0, 1) * 255.0
    return 0.0282 * _uicm(img) + 0.2953 * _uism(img) + 3.5753 * _uiconm(img)


# ---------------------------------------------------------------- UCIQE (numpy)
def uciqe(x):
    """x: CHW tensor in [0,1] -> scalar UCIQE (higher is better)."""
    rgb = x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    lab = rgb2lab(rgb)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()
    ls = np.sort(l.ravel())
    n = max(1, int(0.01 * ls.size))
    con_l = ls[-n:].mean() - ls[:n].mean()
    sat = chroma / np.maximum(np.sqrt(chroma ** 2 + l ** 2), 1e-6)
    # L and chroma are normalised by 100 so that UCIQE lies in the usual ~[0, 1] range
    return 0.4680 * sigma_c / 100 + 0.2745 * con_l / 100 + 0.2576 * sat.mean()
