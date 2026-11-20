# UA-Bank: Uncertainty-Aware Reliable Bank for Semi-Supervised Underwater Image Restoration

DLCV 2026 course project. Commands in this document run from the repository root.

Base paper: **Huang et al., "Contrastive Semi-Supervised Learning for Underwater Image Restoration via Reliable Bank", CVPR 2023** ([paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Contrastive_Semi-Supervised_Learning_for_Underwater_Image_Restoration_via_Reliable_Bank_CVPR_2023_paper.pdf), [official code](https://github.com/Huang-ShiRui/Semi-UIR)).

---

## Archived checkpoint and 200-label follow-up

The 22.6006 dB checkpoint is backed up locally at
`checkpoints/archive/scene_supervised_l50_22.6006dB/final.pth`. Its manifest records the SHA-256 and the benchmark
scores. The two files at the original and archive paths were checked for byte-for-byte equality.

| Checkpoint | Labeled pairs | PSNR | SSIM |
|---|---:|---:|---:|
| Archived whole-scene model | 50 | 22.6006 | 0.8923 |
| Whole-scene model trained from scratch | 200 | **22.6952** | 0.8999 |
| Whole-scene model with larger-frame fine-tuning | 200 | 22.6882 | **0.9007** |

The fine-tuned model improved SSIM on 66 of 90 UIEB test images. The paired mean gain was +0.0084 (95% bootstrap
CI +0.0049 to +0.0119). The PSNR gain was +0.09 dB (95% CI −0.26 to +0.43 dB). With the previously flagged
test image excluded, the SSIM gain was +0.0085. [Benchmark results](../results/scene_l200_benchmark.csv) ·
[Paired statistics](../results/scene_l200_paired.csv) · [89-image analysis](../results/scene_l200_no_overlap.csv)

The follow-up compared recipes on the same [15 held-out labeled scenes](../results/scene_dev15_names.txt).
The 50-label control achieved 22.96 dB validation PSNR. Larger-frame fine-tuning reached 22.82 dB and cast
augmentation reached 22.62 dB at the same budget. In the 200-label regime, training from scratch reached
22.88 dB and larger-frame fine-tuning lifted it to 23.06 dB. Continuing directly from the 50-label weights
reached 22.96 dB with 600 steps and 22.91 dB with 360 steps. These are final student scores on the common holdout.
The 200-label comparison uses the 150 additional labeled pairs already in `data/UIEB/labeled`.

```bash
python train.py --config configs/scene_supervised_l200.yaml
python train.py --config configs/scene_finetune_l50.yaml --data-root data/UIEB --init checkpoints/scene_supervised_l200/final.pth --iters 240 --name scene_finetune_l200
python eval.py --ckpt checkpoints/archive/scene_supervised_l50_22.6006dB/final.pth checkpoints/scene_supervised_l200/final.pth checkpoints/scene_finetune_l200/final.pth --csv results/scene_l200_benchmark.csv --per_image_csv results/scene_l200_by_image.csv --comparison_csv results/scene_l200_paired.csv --compare scene_finetune_l200 scene_supervised_l50 --compare scene_supervised_l200 scene_supervised_l50 --compare scene_finetune_l200 scene_supervised_l200 --save_dir results/outputs/scene_l200_benchmark
```

## Fidelity-calibrated follow-up (50 labels)

| Method | PSNR ↑ | SSIM ↑ | UIQM ↑ | UCIQE ↑ | ΔPSNR vs. Semi-UIR |
|---|---:|---:|---:|---:|---:|
| Semi-UIR reimplementation | 17.8282 | 0.8333 | 3.4446 | 0.3571 | — |
| Original UA-Bank | 17.7124 | 0.8187 | 3.4196 | 0.3597 | −0.12 dB |
| Whole-scene supervised control | **22.6006** | **0.8923** | 3.7495 | **0.3720** | **+4.77 dB** |
| Whole-scene model with fidelity bank | 22.5323 | 0.8920 | **3.7499** | 0.3696 | +4.70 dB |

On the 90-image UIEB split, the model with the fidelity bank improved PSNR on **81 of 90 images** over
Semi-UIR (paired 95% bootstrap CI for the mean gain **+3.94 to +5.50 dB**). The matched control shows that the
scene-level model and training recipe account for the improvement at this seed. Adding the bank changed PSNR
by −0.07 dB (95% CI −0.17 to +0.03 dB). Both models use one inference pass. The paired intervals capture
variation across test images for these checkpoints. [Test scores](../results/fidelity_l50.csv) ·
[Per-image scores](../results/fidelity_l50_by_image.csv) · [Paired comparisons](../results/fidelity_l50_paired.csv) ·
[Example restorations](../results/figures/fidelity_l50_examples.png)

The supplied split contains an exact RGB-pixel duplicate. Test pair `0067.png` matches unlabeled training
pair `0498.png`, whose reference was reserved for historical diagnostics. With this test image excluded,
the supervised control gains **+4.78 dB** and the fidelity-bank model gains **+4.71 dB** over the reproduced
baseline. [89-image sensitivity analysis](../results/fidelity_l50_no_overlap.csv)

We addressed two problems in the original experiments. The patch-trained U-Net had limited scene context for
correcting global color casts. The UIQM gate also accepted pseudo-labels that looked more enhanced while
losing fidelity to the reference.

- **Scene-level color correction.** Full-frame RGB percentiles condition a color transform with piecewise-linear
  tone curves and channel mixing. A small residual U-Net refines the result. The 0.288M-parameter model trains
  on 96×96 whole-frame views and restores each test image at its native benchmark resolution.
- **Fidelity-calibrated bank.** The EMA teacher predicts the original scene and a mildly color-shifted view.
  A small regressor estimates log patch RMSE from the raw image, candidate restoration, and view disagreement.
  It learns from the labeled pairs. After a supervised warm-up, the current regressor scores both the new
  teacher output and the stored candidate. The bank replaces a filled entry when estimated mean squared error
  drops by at least 3%. Estimated patch error also scales the consistency loss on the unlabeled branch.

Development used a deterministic holdout of 10 labeled scenes. The two final runs use all 50 labeled pairs.
Both share the same model, initialization, labeled data and 450-iteration schedule. The semi-supervised run
also uses 750 unlabeled inputs. The final checkpoints are scored on the 90 UIEB test pairs. The new training
recipe reserves reference access for the 50 labeled training pairs and final test scoring.

```bash
python data/make_lowlabel_split.py --src data/UIEB --dst data/UIEB/L50 --n_labeled 50
python train.py --config configs/fidelity_l50.yaml
python train.py --config configs/fidelity_l50.yaml --supervised --name scene_supervised_l50
python eval.py --ckpt checkpoints/fidelity_l50/final.pth checkpoints/baseline_l50/final.pth checkpoints/ours_l50/final.pth checkpoints/scene_supervised_l50/final.pth --csv results/fidelity_l50.csv --per_image_csv results/fidelity_l50_by_image.csv --comparison_csv results/fidelity_l50_paired.csv --compare fidelity_l50 baseline_l50 --compare fidelity_l50 ours_l50 --compare fidelity_l50 scene_supervised_l50 --save_dir results/outputs/fidelity_benchmark
```

The explicit split destination also works on Windows when an archive has flattened the original symbolic links.
To train a development run, add `--holdout 10 --name fidelity_dev` to the first training command.

### Inference with the original checkpoints

An optional ensemble reuses the three original 50-label checkpoints. It aligns flip views, adds a low-resolution
residual view, and uses disagreement to weight each model. It then applies a regularized RGB residual fitted on
the 50 labeled training scenes. Scene-grouped cross-validation selected the regularization strength on these
scenes, which the frozen models had seen during their original training. The locked ensemble achieved
**18.46 dB PSNR and 0.8547 SSIM** on the 90-image split (+0.63 dB versus the original single-pass Semi-UIR,
paired 95% CI +0.39 to +0.91 dB). It uses 15 forward passes per image. [Ensemble scores](../results/calibrated_l50.csv) ·
[Fit and fold metadata](../results/calibrated_l50_fit.json) · [Paired comparisons](../results/calibrated_l50_paired.csv)

```bash
python calibrate.py fit --data-root data/UIEB/L50 --output results/calibrated_l50_fit.json
python calibrate.py eval --calibration results/calibrated_l50_fit.json --save-dir results/outputs
```

## Original UA-Bank experiment

## 1. Project overview

**Semi-UIR** trains a restoration network with a mean teacher. The EMA teacher predicts pseudo-labels for
unlabeled underwater images, and a *reliable bank* keeps, for every unlabeled image, the best pseudo-label seen
so far. A pseudo-label enters the bank only if a no-reference IQA score says it beats both the student's
prediction and the current bank entry. The student is trained toward the bank entry with an L1 loss and a
contrastive loss whose negative is the degraded input.

**Our hypothesis.** The bank decision is *image-level and all-or-nothing*: an accepted pseudo-label is trusted
equally at every pixel, although teacher errors in underwater scenes are spatially local (back-scatter,
dark regions, colour casts). Wrong regions are therefore learned with full weight (confirmation bias).

**Our modification — UA-Bank** (all code tagged `[UA-Bank]`):

| | Semi-UIR (original) | **UA-Bank (ours)** |
|---|---|---|
| reliability signal | one global IQA score per image | **per-pixel teacher uncertainty** `U = std_k f(T_k x)` |
| uncertainty source | — | flip views, or **colour-cast consistency**: random per-channel gains `g ~ U[0.85, 1.15]³` (a restorer should be invariant to the water's colour cast) |
| bank update | replace the whole crop if the IQA score improves | same gate, but the teacher's **own 20 % most uncertain pixels keep their previous bank value** |
| unsupervised loss | uniform per-pixel weight | **confidence weight** `w = exp(−U/τ)`, τ = batch mean of U |
| IQA gate | MUSIQ | UIQM (closed-form, CPU-feasible); used for **both** methods |

Novel code: [`semiuir/uncertainty.py`](../semiuir/uncertainty.py), the `mode="pixel"` branch of
[`semiuir/bank.py`](../semiuir/bank.py), the weight map in [`semiuir/losses.py`](../semiuir/losses.py), and the
`tta*` / `weighted` flags in [`train.py`](../train.py).

### Summary of findings

1. **Semi-UIR reproduced, under the right conditions.** With 50 labeled + 750 unlabeled images, Semi-UIR beats
   supervised-only training by **+0.18 dB PSNR (Wilcoxon p = 0.015)**. With 200 labels, the unlabeled branch
   gives no measurable gain (−0.12 dB, p = 0.80).
2. **A failure mode of the reliable bank.** Measured against the hidden GT of the unlabeled images, the bank's
   pseudo-label PSNR **peaks at 19.46 dB and then falls to 18.88 dB**, although the bank is meant to only ever
   improve. The no-reference gate is fooled: as the teacher overfits the few labels, its outputs gain contrast
   and colourfulness (higher UIQM) while losing fidelity.
3. **UA-Bank does not improve on Semi-UIR.** Its results are +0.02/+0.05 dB at 200 labels (p ≈ 0.19) and
   −0.12 dB at 50 labels (p = 0.11). None of these differences is significant.
4. **Why.** Uncertainty localises errors (it lights up on the artefacts in `results/figures/qualitative.png`)
   but is only weakly correlated with pixel error. Flip-view uncertainty has Spearman ρ = 0.09: flips cannot
   see colour casts, which dominate underwater error. Colour-cast uncertainty reaches ρ = 0.20 (2.3× better),
   but it is no better than UIQM as a gate (63.3 % vs 64.0 % pairwise accuracy). It therefore cannot stop the
   bank degradation in finding 2, which is what limits Semi-UIR.

## 2. Environment and dependencies

The original UA-Bank runs used PyTorch MPS on an Apple M2 MacBook Air (8 GB RAM). The whole-scene follow-ups used CPU.
The code automatically uses MPS, then CUDA, then the CPU.

* Python 3.11, PyTorch 2.5.1, torchvision 0.20.1 (CUDA optional; any CUDA 11.8/12.x build works)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Dataset preparation

UIEB (Li et al., TIP 2019), standard 800 train / 90 test split, downloaded from the HuggingFace mirror
[`Hikari0608/UIEB`](https://huggingface.co/datasets/Hikari0608/UIEB) (original: [project page](https://li-chongyi.github.io/proj_benchmark)).

```bash
python data/prepare_uieb.py --root data/UIEB --n_labeled 200                        # ~1.5 GB download, ~290 MB on disk
python data/make_lowlabel_split.py --src data/UIEB --dst data/UIEB/L50 --n_labeled 50  # symlinks, no download
```

Preprocessing: every image is resized so that its short side is 256 px. Two semi-supervised splits are built:

```
data/UIEB/                       data/UIEB/L50/   (label-scarce)
├── labeled/{input,gt}/  200     ├── labeled/{input,gt}/  50
├── unlabeled/input/     600     ├── unlabeled/input/     750
├── unlabeled_gt_analysis/ 600   ├── unlabeled_gt_analysis/ 750
└── test/{input,gt}/      90     └── test -> ../test
```

`unlabeled_gt_analysis/` is read **only** by the bank-quality diagnostic (logged as `bank_psnr`) and by
`analysis.py gate`. It never enters a loss.

## 4. Training

The backbone is a 2.0M-parameter residual U-Net ([`models/unet.py`](../models/unet.py)), the same for every
method. All runs share seed 0, 8000 iterations, batch 4 labeled + 4 unlabeled, 128×128 crops, Adam (lr 2e-4,
cosine), EMA 0.996, consistency weight 0.2 with sigmoid ramp-up over the first 40 % of iterations, and the
official supervised loss (0.8 L1 + 0.2 L2 + 0.3 VGG perceptual + 0.1 gradient).

```bash
# 200 labels
python train.py --config configs/supervised.yaml       # labeled data only
python train.py --config configs/baseline.yaml         # Semi-UIR (image-level reliable bank)
python train.py --config configs/ours.yaml             # UA-Bank, flip uncertainty
python train.py --config configs/ours_color.yaml       # UA-Bank, colour-cast uncertainty
# 50 labels
python train.py --config configs/supervised_l50.yaml
python train.py --config configs/baseline_l50.yaml
python train.py --config configs/ours_l50.yaml         # UA-Bank, colour uncertainty, plain pseudo-label
# or everything (training, evaluation, statistics, figures):
bash scripts/run_all.sh
```

The configs differ **only** in `data_root` and the method flags (`semi`, `bank_mode`, `tta`, `tta_mode`,
`tta_pseudo`, `weighted`). Each run takes about 20–75 min on an M2. The `train_minutes` column in
`metrics.csv` is wall-clock time and is not comparable across runs: some runs shared the machine.
Per-iteration logs are in `results/logs/<name>.jsonl`. The PSNR on 20 test images (`val_psnr20`) is logged
for monitoring only and never used to select a model; the **final** checkpoint is always evaluated.

## 5. Inference and evaluation

```bash
python eval.py --with_input --save_dir results/outputs --csv results/metrics.csv \
  --ckpt checkpoints/{supervised,baseline,ours,ours_color,supervised_l50,baseline_l50,ours_l50}/final.pth
python analysis.py stats                                   # paired Wilcoxon tests -> results/significance.txt
python analysis.py curves --runs supervised_l50 baseline_l50 ours_l50 --tag l50
python analysis.py uncertainty --ckpt checkpoints/ours/final.pth          # U vs error (test set)
python analysis.py uncsource --ckpt checkpoints/ours/final.pth            # flip vs colour U (train set)
python analysis.py gate --runs supervised_l50 baseline_l50 supervised baseline ours ours_color
python analysis.py qualitative --ckpt checkpoints/ours_l50/final.pth --runs supervised_l50 baseline_l50 ours_l50
```

The evaluation runs on all 90 test images at full resolution (short side 256) and scores the student network,
as the official code does. It reports PSNR and SSIM (full-reference) and UIQM and UCIQE (no-reference; higher
is better). Restored images are written to `results/outputs/<run>/`.

## 6. Checkpoints and results

**UIEB test set (90 images)** — `results/metrics.csv`

| Labels | Method | PSNR ↑ | SSIM ↑ | UIQM ↑ | UCIQE ↑ |
|---|---|---|---|---|---|
| — | Raw input | 16.42 | 0.738 | 2.65 | 0.327 |
| 200 | Supervised only | **19.69** | **0.878** | 3.50 | **0.348** |
| 200 | Semi-UIR (re-impl.) | 19.57 | 0.871 | **3.53** | 0.345 |
| 200 | UA-Bank (flip U) | 19.59 | 0.872 | **3.53** | 0.343 |
| 200 | UA-Bank (colour U) | 19.62 | 0.869 | 3.51 | 0.344 |
| 50 | Supervised only | 17.65 | 0.833 | 3.42 | 0.359 |
| 50 | Semi-UIR (re-impl.) | **17.83** | **0.833** | **3.44** | 0.357 |
| 50 | UA-Bank (colour U) | 17.71 | 0.819 | 3.42 | **0.360** |

**Paired per-image PSNR differences** — `results/significance.txt`

| Comparison | ΔPSNR | images improved | Wilcoxon p |
|---|---|---|---|
| Semi-UIR vs supervised, 50 labels | **+0.181** | 58 % | **0.015** |
| Semi-UIR vs supervised, 200 labels | −0.117 | 52 % | 0.801 |
| UA-Bank (flip) vs Semi-UIR, 200 labels | +0.020 | 59 % | 0.193 |
| UA-Bank (colour) vs Semi-UIR, 200 labels | +0.051 | 58 % | 0.185 |
| UA-Bank vs Semi-UIR, 50 labels | −0.116 | 44 % | 0.106 |

**Diagnostics**

| Quantity | Value | File |
|---|---|---|
| Bank pseudo-label PSNR vs hidden GT (Semi-UIR, 50 labels): peak → final | 19.46 → 18.88 dB | `results/logs/baseline_l50.jsonl` |
| Same for UA-Bank (50 labels) | 19.33 → 18.91 dB | `results/logs/ours_l50.jsonl` |
| Spearman(U, pixel error), flip vs colour-cast U (train set) | 0.09 vs 0.20 | `results/uncertainty_source.json` |
| Gate accuracy (picks the higher-PSNR output): UIQM vs colour U | 64.0 % vs 63.3 % | `results/gate_accuracy.json` |
| Error of the top-10 % most uncertain pixels vs the rest (flip U, test) | +23 % | `results/uncertainty.json` |
| Naive pixel bank (absolute-U comparison), bank PSNR at iteration 1500 | 19.20 vs 19.49 dB (Semi-UIR) | `results/logs/abl_naivepixel_partial.jsonl` |

Figures: `results/figures/curves_l50.png` (student and bank PSNR over training),
`results/figures/qualitative.png`, and `results/figures/uncertainty_vs_error.png`.

The original checkpoint files are available in the local project bundle under `checkpoints/<name>/final.pth`. See the [checkpoint guide](../checkpoints/README.md) for placement and verification.

## 7. Code organisation

```
configs/                 one YAML per run (only data_root and the method flags differ)
data/prepare_uieb.py     download + 200-label split
data/make_lowlabel_split.py   50-label split (symlinks)
models/unet.py           shared lightweight backbone
semiuir/
  data.py                in-memory datasets, official strong augmentation
  losses.py              supervised loss, VGG features, contrastive loss (+ [UA-Bank] weight map)
  bank.py                reliable bank: "image" (original) and "pixel" ([UA-Bank]) update rules
  uncertainty.py         [UA-Bank] perturbation-based pseudo-label and per-pixel uncertainty
  metrics.py             PSNR, SSIM, UIQM (batched torch), UCIQE
train.py  eval.py  analysis.py
scripts/run_all.sh       full reproduction;  scripts/queue_lowprio.sh  sequential low-priority runner
```

## 8. Design history (kept for transparency)

1. **Naive pixel bank (discarded).** Our first rule replaced a bank pixel whenever the new uncertainty was
   lower than the *stored* one. It lowered pseudo-label quality, because uncertainty is not comparable across
   training time: an early, near-identity teacher is flip-equivariant and therefore has U≈0 although its
   output is poor. Final rule: only compare uncertainties *within one teacher snapshot*.
2. **Flip → colour-cast uncertainty.** Chosen on labeled **training** images (ρ 0.09 → 0.20), not on the test set.
3. **Plain pseudo-label for the colour variant.** Averaging outputs on colour-shifted inputs lowered bank PSNR
   (19.83 vs 20.08 dB, measured on the training split). The 50-label UA-Bank therefore uses colour perturbations
   only to estimate U.
4. **Label-scarce setting.** With 200 labels, no semi-supervised variant differed measurably from supervised
   training, so the comparison was repeated with 50 labels, where the unlabeled branch matters.

## 9. Differences from the official implementation (for a fair, CPU-feasible study)

* The backbone is a 2M-parameter U-Net instead of AIM-Net, and the illumination-map input is not used.
* MUSIQ is replaced by UIQM as the bank's IQA gate (MUSIQ is a transformer and too slow on CPU).
* The run is 8000 iterations at 128×128 crops with one truncated VGG-16 for the perceptual and contrastive
  losses (the official code uses VGG-16 + VGG-19).
* The bank starts empty, and empty entries are masked out of the loss (the official code initialises the bank
  with black images).

These changes apply **identically** to every method. Absolute numbers are therefore lower than in the paper,
which trains AIM-Net for 200 epochs on a GPU.
