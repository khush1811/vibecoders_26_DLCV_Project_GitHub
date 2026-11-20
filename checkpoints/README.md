# Model Checkpoints

This directory documents the model checkpoints from our **Semi-UIR / UA-Bank underwater image restoration project**.

## Available Checkpoints

| Model | Checkpoint Path |
|---|---|
| Final fine-tuned (200 labels) | `scene_finetune_l200/final.pth` |
| Supervised (200 labels) | `scene_supervised_l200/final.pth` |
| Archived supervised (50 labels) | `archive/scene_supervised_l50_22.6006dB/final.pth` |
| Original 50-label path | `scene_supervised_l50/final.pth` |
| Fidelity bank (50 labels) | `fidelity_l50/final.pth` |

All paths are relative to `checkpoints/`. The archived and original 50-label files have identical checksums.

## Model Performance

Evaluated on 90 UIEB test images.

| Model | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| Supervised (50 labels) | 22.6006 | 0.8923 |
| Supervised (200 labels) | **22.6952** | 0.8999 |
| Fine-tuned (200 labels) | 22.6882 | **0.9007** |

The final model uses approximately **288K parameters**, achieving high restoration quality with a lightweight architecture.

## Checkpoint Verification

Copy the `.pth` files from the local project bundle into their corresponding paths.

Verify file integrity using:

```bash
# Linux
sha256sum -c checkpoints/SHA256SUMS

# macOS
shasum -a 256 -c checkpoints/SHA256SUMS
```

Run from the repository root, provided the checksum manifest uses root-relative paths.

## Evaluation

```bash
python eval.py --data_root data/UIEB --split test --device cpu --weights student --ckpt checkpoints/scene_finetune_l200/final.pth --save_dir results/outputs/final
```

## References

- [Experiment Notes](https://github.com/khush1811/vibecoders_26_DLCV_Project_GitHub/blob/main/docs/experiments.md)
- [SHA-256 Checksums](https://github.com/khush1811/vibecoders_26_DLCV_Project_GitHub/blob/main/checkpoints/SHA256SUMS)
- [Main README](https://github.com/khush1811/vibecoders_26_DLCV_Project_GitHub)

**Recommended checkpoint:** `scene_finetune_l200/final.pth` — final selected model with the highest reported SSIM.
