# E4 — Gradient-Routing Ablation (planned)

*How much of RotationVQ's benefit comes from the rotation, and how
much from the rescaling?*

## Goal

Compare four configurations of the `RotationVQ` quantizer that differ
only in how the backward Jacobian is computed, while keeping the
forward pass bit-exactly identical across all four:

| Mode | Backward Jacobian | Code path |
|---|---|---|
| `ste` | Identity `I` | classic straight-through estimator |
| `no_rotation` | `s · I` | keeps only the scalar rescaling |
| `no_rescale` | `R` | keeps only the Householder reflection |
| `full` | `s · R` | rotation + rescaling (the proposed method) |

Because only the backward gradient changes across the four
configurations, any quality difference cleanly isolates each
component's contribution.

## Protocol

Run four training jobs with `model.quantizer.mode` set to each
value; everything else matches the E1 recipe.

```
for MODE in ste no_rotation no_rescale full:
  train.py --config configs/cifar10_rotation.yaml \
           --override model.quantizer.mode=$MODE \
                      train.max_steps=12000 \
                      train.batch_size=1024 \
                      train.lr=1.6e-3 \
                      logging.out_dir=results/cifar10_rotation_${MODE}_e4
```

Total: 4 runs × ~ 30 min on a single H100 = ~ 2 GPU-hours.

## Expected artefact

Table "Gradient-routing ablation" in the paper, four rows, each
reporting val PSNR, SSIM, usage, and perplexity. The four rows are
already wired into `paper/sections/results.tex` with placeholder
macros.

## Status

Not yet launched. Waiting for E1 to complete.
