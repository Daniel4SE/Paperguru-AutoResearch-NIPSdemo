# E1 — Main CIFAR-10 Comparison

*Four quantizers, one training recipe, one GPU.*

## Goal

Compare four quantization strategies under an **identical** encoder,
decoder, loss, optimiser, and step budget so that any quality
difference is attributable to the quantizer alone:

1. **VQ-VAE (STE)** — the baseline of van den Oord et al. (2017).
2. **RotationVQ (ours)** — the closed-form Householder reflection
   plus scalar rescaling, wrapped in stop-gradient operators so that
   the forward value is bit-exactly `q`.
3. **FSQ** — Mentzer et al. (2024), per-channel tanh-bounded
   rounding with an implicit 64 000-entry codebook.
4. **Gumbel-Softmax VQ** — Jang et al. (2017), continuous relaxation
   of the nearest-neighbour lookup.

## Protocol

Each run uses the YAML config `code/configs/cifar10_{method}.yaml`
plus the following CLI overrides:

```
train.max_steps=12000
train.log_every=200
train.val_every=1000
train.save_every=6000
train.max_val_batches=20
train.batch_size=1024
train.lr=1.6e-3
train.lr_min=1.0e-6
train.amp=false
train.grad_clip=1.0
data.num_workers=0
```

Every run seeds `torch.manual_seed(0)`, `numpy.random.seed(0)`,
enables TF32 tensor cores
(`torch.set_float32_matmul_precision('high')` and
`torch.backends.cudnn.benchmark = True`), and writes its full
TensorBoard log to `results/cifar10_{method}_e1/tb/`.

## Orchestration

The four runs are chained by a `watcher.sh` that polls for each
preceding run's `ckpt_final.pt` and launches the next. This keeps
the GPU occupied with exactly one run at a time and makes the chain
robust to SSH session drops:

```bash
wait_for_done results/cifar10_vanilla_e1/ckpt_final.pt
launch cifar10_rotation
wait_for_done results/cifar10_rotation_e1/ckpt_final.pt
launch cifar10_fsq
...
```

## Throughput observations

| Batch size | Wall / 200 steps | GPU utilisation | Samples/s |
|-----------:|-----------------:|----------------:|----------:|
|        128 |            145 s |            19 % |       883 |
|        256 |            175 s |              —  |     1 461 |
|        512 |            180 s |              —  |     2 845 |
|    **1024** |        **183 s** |        **89 %** | **5 607–6 565** |

Doubling the batch barely changes wall time in this compute-bound-
by-launch-overhead regime, so per-sample throughput rises
almost linearly with batch size up to ~ 1024, where we become
genuinely compute-bound.

## Current results (Vanilla baseline, complete)

Measured on the 10 000-image CIFAR-10 validation set at step 12 000:

| Metric | Value |
|---|---|
| val PSNR | **25.84 dB** |
| val SSIM | **0.87** |
| val codebook usage | **1.000** (1024/1024 codes active) |
| val codebook perplexity | **948.67** (92.6 % of codebook max) |
| train PSNR (mean, last 200 steps) | 26.98 dB |
| throughput | **6 565 samples/s** |
| wall time | 1 846 s ≈ 30 min 46 s |

The codebook usage of 100 % at perplexity 948/1024 indicates that
the dead-code reinitialisation loop is functioning as intended: no
code has collapsed, and the token distribution is near-uniform on
this well-mixed training distribution.

## Current results (RotationVQ, FSQ, Gumbel — in flight)

At the time of writing, RotationVQ has just started training under
the same watcher. FSQ and Gumbel-Softmax VQ follow automatically.
Their numbers will be appended to this file and propagated to the
paper via `scripts/collect_results.py` → `paper/results_numbers.tex`
the moment each `ckpt_final.pt` appears.

## Artefacts

For each completed run we keep:

- `results/cifar10_{method}_e1/config.yaml` — the exact merged
  configuration used.
- `results/cifar10_{method}_e1/tb/events.out.tfevents.*` — full
  TensorBoard log stream (loss, reconstruction PSNR, usage,
  perplexity, learning rate, at every log step plus every val step).
- `results/cifar10_{method}_e1/ckpt_step6000.pt` — mid-training
  checkpoint, for comparing partially-trained behaviours.
- `results/cifar10_{method}_e1/ckpt_final.pt` — final checkpoint,
  used for all reported numbers and for reconstruction case studies.

Copies of these (minus the large `.pt` files, which are ~ 65 MB
each) are in [`logs/`](logs/).
