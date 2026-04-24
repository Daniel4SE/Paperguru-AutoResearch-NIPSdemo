"""Smoke test for models.vqae on CIFAR-sized inputs."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/vq-rotation"))

import torch
from models.vqae import VQAutoEncoder, build_vqae

torch.manual_seed(0)


def run(qtype: str, **kw):
    cfg = {
        "quantizer": {"type": qtype, **kw},
        "encoder": {
            "in_channels": 3, "base_ch": 32, "ch_mults": (1, 2, 2),
            "num_res_blocks": 1, "z_channels": 64, "input_resolution": 32,
            "attn_resolutions": (),
        },
        "decoder": {
            "out_channels": 3, "base_ch": 32, "ch_mults": (1, 2, 2),
            "num_res_blocks": 1, "z_channels": 64, "output_resolution": 32,
            "attn_resolutions": (),
        },
        "embed_dim": 8 if qtype != "fsq" else 6,
    }
    m = build_vqae(cfg)
    m.train()
    x = torch.randn(2, 3, 32, 32, requires_grad=False)
    out = m(x)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    assert out.x_hat.shape == x.shape, f"x_hat shape {out.x_hat.shape} vs {x.shape}"
    recon = torch.nn.functional.mse_loss(out.x_hat, x)
    loss = recon + out.stats["commit_loss"]
    loss.backward()
    # Check encoder receives gradient.
    enc_grad = next(m.encoder.parameters()).grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    print(
        f"[{qtype}] params={n_params:.2f}M "
        f"z_e={out.z_e.shape} z_q={out.z_q.shape} "
        f"usage={out.stats['usage'].item():.3f} "
        f"perp={out.stats['perplexity'].item():.1f} "
        f"recon_loss={recon.item():.4f}"
    )


if __name__ == "__main__":
    run("vanilla", num_codes=128)
    run("rotation", num_codes=128, mode="full")
    run("rotation", num_codes=128, mode="no_rotation")
    run("rotation", num_codes=128, mode="no_rescale")
    run("fsq", levels=(8, 8, 8, 5, 5, 5))
    run("gumbel", num_codes=128)
    print("\nVQAE smoke tests PASSED")
