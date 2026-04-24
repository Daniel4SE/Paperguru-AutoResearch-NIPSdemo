"""Verify each config loads and a single forward/backward completes."""
import sys, os, time, glob
sys.path.insert(0, os.path.expanduser("~/vq-rotation"))
import torch, yaml
from models.vqae import build_vqae

for path in sorted(glob.glob(os.path.expanduser("~/vq-rotation/configs/cifar10_*.yaml"))):
    cfg = yaml.safe_load(open(path))
    name = os.path.basename(path)
    t0 = time.time()
    try:
        model = build_vqae(cfg["model"])
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        loss = (out.x_hat - x).pow(2).mean() + out.stats["commit_loss"]
        loss.backward()
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[OK ] {name}: {n:.2f}M params, "
              f"z={out.z_q.shape}, loss={loss.item():.4f}, "
              f"usage={out.stats['usage'].item():.3f}, "
              f"{time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {e}")
