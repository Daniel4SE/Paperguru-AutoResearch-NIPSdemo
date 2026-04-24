# Experiments

This directory documents every experiment run in this repository
with enough detail that an independent reader can reproduce it
byte-for-byte on their own hardware.

## Index

| File | Covers |
|---|---|
| [`hyperparameters.md`](hyperparameters.md) | Every hyperparameter of every configuration: model architecture, training schedule, precision. |
| [`E1_main_comparison.md`](E1_main_comparison.md) | The main CIFAR-10 benchmark: four quantizers under a shared recipe. |
| [`E2_codebook_size.md`](E2_codebook_size.md) | Codebook-size ablation (planned): K ∈ {1024, 4096, 8192, 16384}. |
| [`E4_gradient_routes.md`](E4_gradient_routes.md) | Gradient-routing ablation (planned): {STE, rotation-only, rescale-only, full}. |
| [`logs/`](logs/) | Copies of training logs, TensorBoard events, and captured stdout for each run. |

## Reproducibility checklist

- [x] **Code version** — pinned at the root of this repository; no
  hidden dependencies.
- [x] **Seed** — `torch.manual_seed(0)`, `numpy.random.seed(0)`,
  fixed at the start of every training run.
- [x] **GPU** — single NVIDIA H100 80 GB HBM3, driver 580.126.20,
  CUDA 12.4 runtime.
- [x] **Framework** — PyTorch 2.6.0+cu124, torchvision 0.21.0.
- [x] **Data** — CIFAR-10 via `torchvision`; the identical
  deterministic download routine produces the same bytes on every
  machine.
- [x] **Schedule** — cosine learning-rate decay, fixed step count.
- [x] **Precision** — FP32 + TF32 tensor cores (AMP is off; see
  `story/03-training.md` for why).
- [x] **Logging** — every run writes a TensorBoard event stream to
  `results/<run>/tb/`; these events are the source of every number
  in the paper.
- [x] **Numbers pipeline** — `scripts/collect_results.py` extracts
  metrics from the TensorBoard events into `paper_numbers.tex`;
  no manual re-typing of numbers anywhere.
