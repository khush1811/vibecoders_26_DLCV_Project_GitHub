"""In-memory datasets (images are small after data/prepare_uieb.py, so we preload them)."""
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def load(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).contiguous()  # uint8 CHW


def list_png(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".png"))


def strong_aug(x):
    """Strong augmentation of the official Semi-UIR (colour jitter, grayscale, blur). x: float CHW."""
    if random.random() < 0.8:
        x = T.ColorJitter(0.5, 0.5, 0.5, 0.25)(x)
    x = T.RandomGrayscale(0.2)(x)
    if random.random() < 0.5:
        k = random.choice([1, 3, 5])
        x = T.GaussianBlur(k, sigma=(0.1, 2.0))(x)
    return x


class Labeled(Dataset):
    """Paired (input, gt) random crops with flips."""

    def __init__(self, root, crop=128):
        names = list_png(os.path.join(root, "input"))
        self.inp = [load(os.path.join(root, "input", n)) for n in names]
        self.gt = [load(os.path.join(root, "gt", n)) for n in names]
        self.crop = crop

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, i):
        a, b = self.inp[i], self.gt[i]
        _, h, w = a.shape
        y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        a = a[:, y:y + self.crop, x:x + self.crop].float() / 255
        b = b[:, y:y + self.crop, x:x + self.crop].float() / 255
        if random.random() < 0.5:
            a, b = a.flip(-1), b.flip(-1)
        if random.random() < 0.5:
            a, b = a.flip(-2), b.flip(-2)
        return a, b


def resize_scene(image, size):
    shape = (size, size) if isinstance(size, int) else tuple(size)
    return F.interpolate(image[None].float(), size=shape, mode="bilinear",
                         antialias=True, align_corners=False)[0].round().clamp(0, 255).byte()


class ScenePairs(Dataset):
    """Resized whole scenes with paired geometric augmentation."""

    def __init__(self, root, names, size, tone_augmentation=0.0, cast_mix_probability=0.0):
        self.inputs = [resize_scene(load(os.path.join(root, "input", n)), size) for n in names]
        self.targets = [resize_scene(load(os.path.join(root, "gt", n)), size) for n in names]
        self.tone_augmentation = tone_augmentation
        self.cast_mix_probability = cast_mix_probability
        if cast_mix_probability:
            self.casts = [self.estimate_cast(a.float() / 255, b.float() / 255)
                          for a, b in zip(self.inputs, self.targets)]

    @staticmethod
    def estimate_cast(raw, reference):
        """Fit a per-channel slope and offset from a labeled pair."""
        raw_mean = raw.mean(dim=(-2, -1), keepdim=True)
        ref_mean = reference.mean(dim=(-2, -1), keepdim=True)
        ref_centered = reference - ref_mean
        covariance = ((raw - raw_mean) * ref_centered).mean(dim=(-2, -1), keepdim=True)
        variance = ref_centered.square().mean(dim=(-2, -1), keepdim=True)
        attenuation = (covariance / variance.clamp_min(1e-4)).clamp(0.1, 1.2)
        return attenuation, raw_mean - ref_mean * attenuation

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        a, b = self.inputs[i].float() / 255, self.targets[i].float() / 255
        for dim in (-1, -2):
            if random.random() < 0.5:
                a, b = a.flip(dim), b.flip(dim)
        if self.cast_mix_probability and random.random() < self.cast_mix_probability:
            attenuation, offset = random.choice(self.casts)
            simulated = b * attenuation + offset
            a = a.lerp(simulated, random.uniform(0.2, 0.65)).clamp(0, 1)
        if self.tone_augmentation and random.random() < 0.5:
            # Apply the same camera exposure and tone changes to both sides of the pair.
            strength = self.tone_augmentation
            gain = torch.empty(3, 1, 1).uniform_(-0.5 * strength, 0.5 * strength)
            gain = (gain + random.uniform(-strength, strength)).exp()
            gamma = math.exp(random.uniform(-strength, strength))
            a, b = (a.pow(gamma) * gain).clamp(0, 1), (b.pow(gamma) * gain).clamp(0, 1)
        return a, b


class Unlabeled(Dataset):
    """Unlabeled inputs resized to size x size (as in the official code) and randomly cropped.

    Returns the weak view (teacher input), the strong view (student input), the image index
    and the crop offset, so that the reliable bank can be addressed at pixel level.
    """

    def __init__(self, root, size=256, crop=128, sub="input", mild=False):
        names = list_png(os.path.join(root, sub))
        self.names = names
        self.inp = [resize_scene(load(os.path.join(root, sub, n)), size) for n in names]
        self.size, self.crop = size, crop
        self.mild = mild

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, i):
        offset = self.size - self.crop
        y, x = (random.randint(0, offset), random.randint(0, offset)) if offset else (0, 0)
        w = self.inp[i][:, y:y + self.crop, x:x + self.crop].float() / 255
        if self.mild:
            gains = 0.9 + 0.2 * torch.rand(3, 1, 1)
            strong = (w * gains).clamp(0, 1)
        else:
            strong = strong_aug(w)
        return w, strong, i, torch.tensor([y, x])


class Paired(Dataset):
    """Full-resolution test pairs (no augmentation)."""

    def __init__(self, root):
        self.names = list_png(os.path.join(root, "input"))
        self.root = root

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        a = load(os.path.join(self.root, "input", n)).float() / 255
        b = load(os.path.join(self.root, "gt", n)).float() / 255
        return a, b, n
