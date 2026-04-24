# Code

All PyTorch code from the server-side training environment
(`ubuntu@217.18.55.93:~/vq-rotation/`) as of 2026-04-24, minus
dataset and checkpoint binaries.

## Layout

```
code/
├── models/
│   ├── quantizers.py   ← 4 quantizers + EMA codebook + Jacobian tests
│   └── vqae.py         ← VQ-GAN-style encoder/decoder + wrapper
├── configs/            ← one YAML per (dataset × quantizer) pair
├── scripts/
│   ├── test_quantizers.py   ← CPU unit tests (Jacobian to 1.2e-7)
│   ├── test_vqae.py          ← end-to-end smoke test
│   ├── test_metrics.py       ← PSNR / SSIM / perplexity checks
│   ├── visualize.py          ← curves and reconstructions → PNG
│   ├── viz_paper.py          ← paper-quality figures with error maps
│   ├── collect_results.py    ← TensorBoard → LaTeX macros
│   └── config_probe.py       ← load every YAML, do one forward
├── eval/
│   └── metrics.py       ← PSNR, SSIM, LPIPS, codebook stats, FID
├── train.py             ← single training entry point
└── requirements.txt     ← pinned dependencies
```

## Quickstart

```bash
# 1. Create venv (Python 3.10+)
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. CPU unit tests (~10 seconds, no GPU required)
python scripts/test_quantizers.py
python scripts/test_vqae.py
python scripts/test_metrics.py

# 3. GPU training (H100 recommended; the script runs on any CUDA GPU)
python train.py --config configs/cifar10_rotation.yaml --device cuda \
  --override train.max_steps=12000 \
             train.batch_size=1024 \
             train.lr=1.6e-3 \
             train.amp=false

# 4. Produce training-dynamics figures once training has started
python scripts/visualize.py        # basic curves
python scripts/viz_paper.py        # paper-quality figures

# 5. Dump final numbers into a LaTeX macro file
python scripts/collect_results.py
```

## The quantizer interface

Every quantizer subclasses `BaseQuantizer`:

```python
class BaseQuantizer(nn.Module):
    num_codes: int  # K for learned codebooks; implicit grid size for FSQ
    def forward(self, z_e: Tensor) -> QuantizerOutput: ...
```

and returns a `QuantizerOutput(z_q, indices, stats)`. This lets
`train.py` stay quantizer-agnostic.

## Unit tests

The unit tests are the single most important verification in this
repository. In particular:

```
$ python scripts/test_quantizers.py
...
[jacobian check] max |J - sR| = 1.19e-07
[jacobian check] passed OK
ALL TESTS PASSED
```

This asserts that the autograd Jacobian of `RotationVQ` at a known
input matches the analytic `s·R` to within FP32 machine precision.
This is the test that went from failing to passing after the
numerical-stability fix in `quantizers.py`; any regression in the
stop-gradient construction shows up here immediately.

## Configs

Every experiment is described by a single YAML file in
`configs/`. The same file runs on CPU or GPU; switch via
`--device cpu|cuda`.

CLI overrides use a dot path:

```bash
python train.py --config configs/cifar10_rotation.yaml \
  --override train.lr=5e-5 model.quantizer.num_codes=4096
```
