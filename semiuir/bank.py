"""Reliable bank: stores, for every unlabeled image, the best pseudo-label found so far.

Two update rules:
  * "image" (original Semi-UIR): a crop of the bank is replaced *as a whole* if the teacher
    prediction scores higher than the student prediction AND than the current bank entry,
    under a no-reference IQA score (MUSIQ in the paper, UIQM here for CPU feasibility).
  * "pixel" ([UA-Bank], ours): after the same image-level gate, the pixels on which the
    teacher itself is least confident are NOT overwritten (the previous bank value is kept),
    giving a composite pseudo-label; the per-pixel uncertainty is stored for loss weighting.
"""
import torch

from .metrics import uiqm


class ReliableBank:
    def __init__(self, n, size, mode="image", reject_q=0.2, device="cpu"):
        assert mode in ("image", "pixel")
        self.mode, self.reject_q, self.dev = mode, reject_q, device
        self.img = torch.zeros(n, 3, size, size, dtype=torch.uint8)
        self.unc = torch.full((n, 1, size, size), float("inf"), dtype=torch.float32)
        self.filled = torch.zeros(n, 1, size, size, dtype=torch.bool)

    def _get(self, idx, yx, c):
        p, u, f = [], [], []
        for i, (y, x) in zip(idx.tolist(), yx.tolist()):
            p.append(self.img[i, :, y:y + c, x:x + c])
            u.append(self.unc[i, :, y:y + c, x:x + c])
            f.append(self.filled[i, :, y:y + c, x:x + c])
        return (torch.stack(p).to(self.dev).float() / 255, torch.stack(u).to(self.dev),
                torch.stack(f).to(self.dev))

    def _put(self, idx, yx, c, p, u, f):
        p = (p.clamp(0, 1) * 255).round().byte().cpu()
        u, f = u.float().cpu(), f.cpu()
        for b, (i, (y, x)) in enumerate(zip(idx.tolist(), yx.tolist())):
            self.img[i, :, y:y + c, x:x + c] = p[b]
            self.unc[i, :, y:y + c, x:x + c] = u[b]
            self.filled[i, :, y:y + c, x:x + c] = f[b]

    @torch.no_grad()
    def quality(self, gt):
        """Diagnostic only: PSNR of the stored pseudo-labels vs. hidden GT on filled pixels."""
        f = self.filled.expand_as(self.img).float()
        se = ((self.img.float() - gt.float()) / 255) ** 2
        mse = (se * f).flatten(1).sum(1) / f.flatten(1).sum(1).clamp_min(1)
        ok = self.filled.flatten(1).any(1)
        return (10 * torch.log10(1 / mse[ok].clamp_min(1e-10))).mean().item(), self.filled.float().mean().item()

    @torch.no_grad()
    def update(self, idx, yx, teacher_pred, teacher_unc, student_pred):
        """Update the bank for a batch of crops and return (pseudo_label, uncertainty, filled)."""
        c = teacher_pred.shape[-1]
        p_bank, u_bank, f_bank = self._get(idx, yx, c)
        s_t, s_s = uiqm(teacher_pred), uiqm(student_pred)
        gate = (s_t > s_s).view(-1, 1, 1, 1)  # teacher must beat the student (both rules)

        # bank score of an empty entry is -inf; partially empty entries count as empty
        s_b = torch.where(f_bank.flatten(1).all(1), uiqm(p_bank), torch.full_like(s_t, -float("inf")))
        take = (gate & (s_t > s_b).view(-1, 1, 1, 1)).expand_as(f_bank)
        if self.mode == "pixel":
            # [UA-Bank] inside an accepted crop, the teacher's own least-confident pixels (top
            # `reject_q` fraction of ITS uncertainty) keep the previous bank value if one exists.
            # Uncertainties are only compared within one teacher snapshot: absolute U is not
            # comparable across training time (a near-identity early teacher is flip-equivariant,
            # hence has U ~ 0 although its output is poor).
            thr = torch.quantile(teacher_unc.flatten(1), 1 - self.reject_q, dim=1).view(-1, 1, 1, 1)
            take = take & ~((teacher_unc > thr) & f_bank)

        p_new = torch.where(take, teacher_pred, p_bank)
        u_new = torch.where(take, teacher_unc, u_bank)
        f_new = f_bank | take
        self._put(idx, yx, c, p_new, u_new, f_new)
        return p_new, u_new, f_new
