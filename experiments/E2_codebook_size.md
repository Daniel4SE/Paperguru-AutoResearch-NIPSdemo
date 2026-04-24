# E2 — Codebook-Size Ablation (planned)

*Does the benefit of rotation-aware gradient routing grow or shrink
with codebook capacity?*

## Goal

Sweep the codebook size `K` over {1024, 4096, 8192, 16384} for both
the STE baseline and the full RotationVQ, keeping every other
hyperparameter fixed to the E1 recipe. The ablation probes two
hypotheses:

1. **Usage hypothesis.** Larger codebooks are more prone to collapse.
   If rotation-aware routing mitigates collapse, its advantage over
   STE in codebook usage and perplexity should grow with K.
2. **Reconstruction hypothesis.** Larger codebooks have more
   representational capacity. If this capacity is actually usable
   under rotation-aware routing, the PSNR gap between Rotation and
   STE should be at least stable, and ideally widen, as K grows.

## Protocol

Identical to E1 except for `model.quantizer.num_codes`:

```
for K in 1024 4096 8192 16384:
  for method in vanilla rotation:
    train.py --config configs/cifar10_${method}.yaml \
             --override model.quantizer.num_codes=$K \
                        train.max_steps=12000 \
                        train.batch_size=1024 \
                        train.lr=1.6e-3
```

Total: 8 runs × ~ 30 min on a single H100 = ~ 4 GPU-hours.

## Expected artefact

Figure "Codebook-size ablation" with two panels:

- **Left.** Val PSNR vs. log K, two lines (Rotation and STE), error
  bars from three seeds.
- **Right.** Codebook perplexity (log scale) vs. log K, same
  configuration.

This figure will replace the placeholder Table 2 in the paper.

## Status

Not yet launched. Waiting for E1 to complete so the shared GPU is
available.
