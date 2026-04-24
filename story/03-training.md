# 03 · Training

*What happens after the GPU finally works.*

## The surprise: AMP makes things slower

Naïve assumption: turn on mixed precision, get a free 2–3× speedup.

Empirical reality on this small (5.4 M parameter) autoencoder:

```
full iter fp32 (bs=128):   91 ms/step → 11.0 steps/s
full iter AMP  (bs=128):  137 ms/step →  7.3 steps/s  ← slower
```

The GradScaler scale-unscale pair, the autocast context switches,
and the dtype casts inside the quantizer (`embed_avg` is kept in
FP32 for numerical stability; encoder features arrive as FP16 under
autocast) together cost more than the tensor-core acceleration
saves at this batch size. The entire AMP path was removed.

## The real bottleneck: GPU under-utilisation

First serious training run: `GPU utilisation = 19 %`, throughput ~ 6
steps/second. The H100 was doing almost nothing.

Batch-size sweep on the new VM:

| Batch size | Wall / 200 steps | GPU util | Samples/s |
|-----------:|-----------------:|---------:|----------:|
|        128 |            145 s |      19% |       883 |
|        256 |            175 s |      —   |     1 461 |
|        512 |            180 s |      —   |     2 845 |
|       1024 |            183 s |      89% |  **5 607** |

Doubling the batch barely increases wall time because the compute
grid is dominated by launch overhead at the smaller sizes. At batch
1024 the H100 is finally busy: 77–89 % utilisation, 450–520 W
draw, ~49 °C under steady load.

We scaled the learning rate linearly from `2e-4 @ bs=128` to
`1.6e-3 @ bs=1024` (the "linear scaling rule") and kept the step
count at 12 000 — giving ~ 12.3 M samples of exposure, roughly
equivalent to 96 k steps at the original batch size.

## Orchestration

Running four quantizers in sequence on a single GPU, without
supervision, means the shell script that launches the next run
must survive the death of any individual run. Our `watcher.sh`
pattern:

```bash
wait_for_done ~/vq-rotation/results/${PREV}/ckpt_final.pt
launch ${NEXT}
```

The watcher polls for `ckpt_final.pt` every 30 seconds with a
50-minute guard, so a lost SSH session or a `pkill` of the parent
does not poison the chain. The full runner is in
[`../code/scripts/run_e1.sh`](../code/scripts/run_e1.sh)
and its state is recorded in `/tmp/watcher.log` on the server.

## The Vanilla run

The first baseline (classic STE-VQ-VAE) took 31 minutes to complete
12 000 steps and finished at:

| Metric | Value |
|---|---|
| val PSNR | **25.84 dB** |
| val SSIM | **0.87** |
| codebook usage | **100 %** (all 1024 codes active on val) |
| perplexity | **948.67** (92.6 % of the codebook maximum) |
| throughput | **6 565 samples/s** |

The codebook usage of 100 % at perplexity 948 / 1024 means the
dead-code reinit loop is doing its job: no code is permanently
silent, and the token distribution is close to uniform.

## What comes next

The watcher is currently running RotationVQ, then FSQ, then
Gumbel-Softmax on the same H100 under the identical recipe. When
all four `ckpt_final.pt` files exist, `scripts/collect_results.py`
scans every TensorBoard event stream, produces
`paper_numbers.tex` with `\renewcommand` wrappers for every metric,
and `make all` rebuilds the PDF with fresh numbers in Table 1.
