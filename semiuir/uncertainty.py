"""[UA-Bank] Per-pixel teacher uncertainty from test-time perturbations.

The teacher is run on K perturbed versions of the weak input and
    pseudo-label  P = mean_k f(T_k x)            (outputs mapped back to the input frame)
    uncertainty   U = mean_c std_k f(T_k x)      (B,1,H,W)
With pseudo="plain" the pseudo-label is the unperturbed prediction f(x) and the perturbations are
used only to estimate U (averaging colour-shifted outputs slightly blurs the colour of P).

Two perturbation families:
  * "flip"  : identity, h-flip, v-flip, hv-flip (exactly invertible geometric views).
  * "color" : identity + (K-1) random per-channel gains g ~ U[1-s, 1+s]^3, i.e. simulated changes
              of the water-type colour cast. The clean scene does not depend on the cast, so an
              ideal restorer is invariant to it; disagreement exposes colour errors, which dominate
              underwater restoration error and to which flips are blind.
"""
import torch

VIEWS = [(), (-1,), (-2,), (-1, -2)]


@torch.no_grad()
def teacher_tta(teacher, x, k=4, mode="flip", strength=0.15, pseudo="mean"):
    preds = []
    if mode == "flip":
        for dims in VIEWS[:k]:
            xi = x.flip(dims) if dims else x
            yi = teacher(xi)
            preds.append(yi.flip(dims) if dims else yi)
    else:
        preds.append(teacher(x))
        for _ in range(k - 1):
            g = 1 + strength * (2 * torch.rand(x.shape[0], 3, 1, 1, device=x.device) - 1)
            preds.append(teacher((x * g).clamp(0, 1)))
    preds = torch.stack(preds)  # K,B,3,H,W
    p = preds[0] if pseudo == "plain" else preds.mean(0)
    return p, preds.std(0, unbiased=False).mean(1, keepdim=True)
