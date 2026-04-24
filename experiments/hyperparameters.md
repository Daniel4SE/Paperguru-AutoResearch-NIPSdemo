# Hyperparameters

All experiments in this repository share a common backbone and
training recipe; only the quantizer module differs. This document
lists *every* value that could conceivably affect a result, grouped
by concern, with citations to the source YAML where applicable.

## 1. Architecture

| Component | Value | Source |
|---|---|---|
| Image resolution | 32 × 32 (CIFAR-10) | all `configs/cifar10_*.yaml` |
| Encoder base channels | 64 | `cifar10_*.yaml :: model.encoder.base_ch` |
| Channel multipliers | (1, 2, 2) → downsample 32 → 16 → 8 | `model.encoder.ch_mults` |
| Residual blocks per level | 2 | `model.encoder.num_res_blocks` |
| Attention resolutions | [] (none) | `model.encoder.attn_resolutions` |
| Encoder bottleneck channels | 128 | `model.encoder.z_channels` |
| Quantizer working dim *d* | 8 (VQ, Rotation, Gumbel); 6 (FSQ) | `model.embed_dim` |
| Pre-/post-quant conv | 1 × 1, bias-free | `models/vqae.py` |
| Decoder | symmetric, +1 res block per level | `models/vqae.py` |
| Normalisation | GroupNorm, up to 32 groups | `models/vqae.py` |
| Activation | SiLU | `models/vqae.py` |
| Mid-block | ResBlock → AttnBlock → ResBlock | `models/vqae.py` |
| Total parameters | 5 403 464 (5.40 M) | `scripts/test_vqae.py` |

## 2. Quantizers

| Quantizer | Size | Learned? | Commitment β | EMA decay | Special |
|---|---|---|---|---|---|
| VQ-VAE (STE) | K = 1024, d = 8 | EMA codebook | 0.25 | 0.99 | dead-code reinit, threshold 1.0 |
| RotationVQ (full) | K = 1024, d = 8 | EMA codebook | 0.25 | 0.99 | Householder `R` + scalar `s` via stop-grad |
| RotationVQ (no_rotation) | same | same | same | same | `R = I`, `s` active |
| RotationVQ (no_rescale) | same | same | same | same | `R` active, `s = 1` |
| FSQ | levels (8,8,8,5,5,5) = 64 000 implicit | none | — | — | tanh-bounded, per-channel round |
| Gumbel-Softmax VQ | K = 1024, d = 8 | backprop codebook | 0.25 | — | τ anneal 1.0 → 0.5 @ rate 3 × 10⁻⁵ |

## 3. Training schedule

| Hyperparameter | Value | Source |
|---|---|---|
| Optimiser | AdamW | `train.py :: train()` |
| β₁, β₂ | (0.9, 0.99) | `train.betas` |
| Weight decay | 0.0 | `train.weight_decay` |
| Peak learning rate | 1.6 × 10⁻³ | `train.lr` (linearly scaled from 2e-4 @ bs 128) |
| Minimum learning rate | 1.0 × 10⁻⁶ | `train.lr_min` |
| LR schedule | Cosine to lr_min | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| Batch size | 1024 | `train.batch_size` |
| Gradient clip (L2 norm) | 1.0 | `train.grad_clip` |
| Precision | FP32 + TF32 tensor cores (AMP off) | `train.amp = false` |
| Step count | 12 000 | `train.max_steps` |
| Warm-up | none (cosine starts at peak) | `train.py` |
| Seed | 0 for torch, numpy, Python random | `train.py` |

Reconstruction loss: `l1_loss(x_hat.clamp(0, 1), x)`.
Commitment loss: `β · ‖z_e − sg[q]‖²`.
No LPIPS, no adversarial, no KL — we intentionally avoid loss
engineering to isolate the gradient-estimator comparison.

## 4. CIFAR-10 data pipeline

| Stage | Detail |
|---|---|
| Source | `torchvision.datasets.CIFAR10`, downloaded to `data/` |
| Normalisation | `ToTensor()` only, i.e. pixel values in [0, 1] |
| Augmentation | none (deliberate; additional augmentation would confound the quantizer comparison) |
| Dataloader | in-memory `TensorDataset` of all 50 000 training images + 10 000 validation images, kept as float32 on CPU |
| Shuffle | `shuffle=True` on the train loader |
| Num workers | 0 (data is already in RAM) |
| Drop last | `True` on train, `False` on val |

## 5. Evaluation

| Metric | Implementation |
|---|---|
| PSNR | `-10 * log10(MSE)` in torch, with inputs clamped to [0, 1] |
| SSIM | scikit-image `structural_similarity`, per-image, channel_axis=0, data_range=1.0 |
| LPIPS (optional) | `lpips` package with AlexNet backbone, frozen |
| Codebook usage | fraction of codes with ≥ 1 hit on the full validation set |
| Codebook perplexity | `exp(−Σ pₖ log pₖ)` where pₖ is the empirical code frequency |
| Throughput | mean over the second half of training; reported as samples/s |

`eval.max_val_batches` is set to 20 for mid-training validation
snapshots (time-bounded) and lifted to `len(val_loader)` for the
final ckpt's validation (exhaustive).

## 6. Hardware

| | Value |
|---|---|
| GPU | NVIDIA H100 80 GB HBM3, PCIe |
| Driver | 580.126.20 |
| CUDA runtime (torch wheel) | 12.4 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| Python | 3.10.12 |
| OS | Ubuntu 22.04 LTS, kernel 5.15.0-171-generic |
| CPU | 24 vCPU (x86_64) |
| RAM | 196 GiB |
| Storage | 97 GB / root, used ≈ 16 GB |

The torch wheel ships its own CUDA runtime; no system `cuda-*` package
is required and we explicitly `unset LD_LIBRARY_PATH` to avoid shadow
linking against `cuda-13.0/lib64`.
