"""Smoke tests for models/quantizers.py. CPU-only, small batches.

Verifies, for each quantizer:
 1. forward output shape and dtype match z_e;
 2. for VanillaVQ and RotationVQ, forward output equals the codebook
    vector q at every spatial location (to float precision);
 3. backward pass produces non-NaN gradients;
 4. for RotationVQ("full"), the autograd Jacobian of z_q w.r.t. z_e at
    one position equals (scale * R), where R is the Householder
    reflection that sends z_hat to q_hat.
"""

import sys, math, os
sys.path.insert(0, os.path.expanduser("~/vq-rotation"))

import torch

from models.quantizers import (
    VanillaVQ, RotationVQ, FSQ, GumbelVQ, build_quantizer,
)

torch.manual_seed(0)


def check_shape(name, out, expected_shape):
    assert out.z_q.shape == expected_shape, (
        f"[{name}] shape mismatch {out.z_q.shape} vs {expected_shape}"
    )


def forward_equals_q(name, q, z_e, out):
    # Reconstruct q by re-looking-up in the codebook
    if out.indices is None:
        return
    # For RotationVQ / VanillaVQ, z_q must equal the gathered codebook
    # vector at every position.
    from torch.nn.functional import embedding as F_embedding
    b, d, h, w = z_e.shape
    idx = out.indices.view(-1)
    emb = q
    expected = F_embedding(idx, emb).view(b, h, w, d).permute(0, 3, 1, 2)
    err = (out.z_q - expected).abs().max().item()
    assert err < 1e-4, f"[{name}] forward mismatch {err}"
    print(f"[{name}] forward equals q (max err {err:.2e}) OK")


def test_vanilla():
    q = VanillaVQ(num_codes=64, dim=8)
    q.train()
    z = torch.randn(2, 8, 4, 4, requires_grad=True)
    out = q(z)
    check_shape("VanillaVQ", out, z.shape)
    forward_equals_q("VanillaVQ", q.codebook.embedding, z, out)
    loss = out.z_q.pow(2).mean() + out.stats["commit_loss"]
    loss.backward()
    assert torch.isfinite(z.grad).all()
    print(f"[VanillaVQ] grad norm {z.grad.norm().item():.3f}; "
          f"usage {out.stats['usage'].item():.3f}")


def test_rotation_modes():
    for mode in ("full", "no_rotation", "no_rescale", "ste"):
        q = RotationVQ(num_codes=64, dim=8, mode=mode)
        q.train()
        z = torch.randn(2, 8, 4, 4, requires_grad=True)
        out = q(z)
        check_shape(f"RotationVQ[{mode}]", out, z.shape)
        forward_equals_q(f"RotationVQ[{mode}]", q.codebook.embedding, z, out)
        loss = out.z_q.pow(2).mean() + out.stats["commit_loss"]
        loss.backward()
        assert torch.isfinite(z.grad).all()
        print(f"[RotationVQ[{mode}]] grad norm {z.grad.norm().item():.3f}")


def test_fsq():
    q = FSQ(levels=(8, 8, 8, 5, 5, 5))
    q.train()
    z = torch.randn(2, 6, 4, 4, requires_grad=True)
    out = q(z)
    check_shape("FSQ", out, z.shape)
    loss = out.z_q.pow(2).mean()
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert out.indices.max().item() < q.num_codes
    print(f"[FSQ] num_codes={q.num_codes} usage={out.stats['usage'].item():.3f}")


def test_gumbel():
    q = GumbelVQ(num_codes=64, dim=8)
    q.train()
    z = torch.randn(2, 8, 4, 4, requires_grad=True)
    out = q(z)
    check_shape("GumbelVQ", out, z.shape)
    loss = out.z_q.pow(2).mean() + out.stats["commit_loss"]
    loss.backward()
    assert torch.isfinite(z.grad).all()
    print(f"[GumbelVQ] grad norm {z.grad.norm().item():.3f}")


def test_rotation_jacobian():
    """For a single (B=1,D=4,H=W=1) position, verify the Jacobian of
    z_q w.r.t. z_e matches s*R where s=||q||/||z||, R is the Householder
    reflection sending z_hat to q_hat.

    Runs in eval() mode so EMA updates do not perturb the codebook we
    manually set.
    """
    torch.manual_seed(42)
    D = 4
    q = RotationVQ(num_codes=8, dim=D, mode="full")
    # Inject a known codebook. Also sync embed_avg so that a stray EMA
    # call (should not happen in eval mode, but defensive) would not
    # overwrite our values.
    with torch.no_grad():
        q.codebook.embedding.zero_()
        q.codebook.embedding[0] = torch.tensor([1.0, 0.5, -0.2, 0.1])
        q.codebook.embed_avg.copy_(q.codebook.embedding)
        q.codebook.cluster_size.fill_(1.0)
        q.codebook.initialized.fill_(True)
    q.eval()

    z_e = torch.tensor(
        [0.9, 0.45, -0.15, 0.08]
    ).view(1, D, 1, 1).requires_grad_(True)

    out = q(z_e)
    # Sanity: forward output must equal codebook[0].
    fwd = out.z_q.view(-1).detach()
    expected_q = q.codebook.embedding[0]
    assert torch.allclose(fwd, expected_q, atol=1e-4), (
        f"forward != q: got {fwd}, want {expected_q}"
    )

    # Analytic sR.
    z = z_e.view(-1).detach()
    qv = expected_q
    z_hat = z / z.norm()
    q_hat = qv / qv.norm()
    u = z_hat - q_hat
    v = u / u.norm()
    R = torch.eye(D) - 2 * torch.outer(v, v)
    s = qv.norm() / z.norm()
    expected_J = s * R

    # autograd jacobian.
    z_q_v = out.z_q.view(D)
    J = torch.zeros(D, D)
    for i in range(D):
        grad = torch.autograd.grad(
            z_q_v[i], z_e, retain_graph=(i < D - 1)
        )[0].view(D)
        J[i] = grad

    err = (J - expected_J).abs().max().item()
    print(f"[jacobian check] max |J - sR| = {err:.2e}")
    assert err < 1e-4, (
        f"Jacobian mismatch: {err}\nJ=\n{J}\nexpected_J=\n{expected_J}"
    )
    print(f"[jacobian check] passed OK")


def test_factory():
    for cfg in [
        {"type": "vanilla", "num_codes": 16, "dim": 4},
        {"type": "rotation", "num_codes": 16, "dim": 4, "mode": "full"},
        {"type": "fsq", "levels": (4, 4, 4, 4)},
        {"type": "gumbel", "num_codes": 16, "dim": 4},
    ]:
        q = build_quantizer(cfg)
        ch = cfg.get("dim", len(cfg.get("levels", (4, 4, 4, 4))))
        z = torch.randn(1, ch, 2, 2)
        out = q(z)
        assert out.z_q.shape == z.shape
    print("[factory] all quantizer types built OK")


if __name__ == "__main__":
    test_vanilla()
    test_rotation_modes()
    test_fsq()
    test_gumbel()
    test_rotation_jacobian()
    test_factory()
    print("\nALL TESTS PASSED")
