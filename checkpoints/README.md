# Model checkpoints

Copy these `.pth` files from the local project bundle into the corresponding paths. The weights are local artifacts. [SHA-256 checksums](SHA256SUMS)

| Run | Path from the repository root |
|---|---|
| Final 200-label fine-tuned model | `checkpoints/scene_finetune_l200/final.pth` |
| Matched 200-label model | `checkpoints/scene_supervised_l200/final.pth` |
| Archived 50-label model | `checkpoints/archive/scene_supervised_l50_22.6006dB/final.pth` |
| Original path for the archived 50-label model | `checkpoints/scene_supervised_l50/final.pth` |
| 50-label fidelity bank | `checkpoints/fidelity_l50/final.pth` |

The archived checkpoint and its original-path copy have the same checksum. Historical runs use `checkpoints/<run>/final.pth`. Use the [experiment notes](../docs/experiments.md) to regenerate a checkpoint when the local copy is unavailable.
