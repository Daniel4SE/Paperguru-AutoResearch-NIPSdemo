import sys, os
sys.path.insert(0, os.path.expanduser("~/vq-rotation"))
import torch
from eval.metrics import compute_psnr, compute_ssim_batch, CodebookStats

torch.manual_seed(0)
x = torch.rand(4, 3, 32, 32)
y = x + 0.05 * torch.randn_like(x)

psnr = compute_psnr(x, y.clamp(0, 1))
ssim = compute_ssim_batch(x, y.clamp(0, 1))
print(f"PSNR={psnr.item():.2f}dB  SSIM={ssim.item():.4f}")

cs = CodebookStats(num_codes=16)
idx = torch.randint(0, 16, (128,))
cs.update(idx)
cs.update(torch.randint(0, 8, (128,)))  # bias toward first half
s = cs.summary()
print("codebook:", {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in s.items()})

print("metrics OK")
