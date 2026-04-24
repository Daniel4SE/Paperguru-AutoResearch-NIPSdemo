"""Reconstruction and codebook metrics.

Provides:
  * compute_psnr, compute_ssim_batch: per-sample PSNR and SSIM averaged
    over a batch.
  * compute_lpips: perceptual distance with a frozen AlexNet (LPIPS).
  * CodebookStats: running meter for codebook usage, perplexity, active
    codes per batch.
  * FIDTracker: accumulates Inception-v3 features and computes FID from
    two distributions, using the pytorch_fid backbone.
"""

from __future__ import annotations

from typing import Optional, Dict
import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as _sk_ssim


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------

def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR averaged over batch. Inputs in [0, max_val]."""
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    mse = mse.clamp_min(1e-10)
    return (20.0 * torch.log10(torch.tensor(max_val, device=x.device))
            - 10.0 * torch.log10(mse)).mean()


def compute_ssim_batch(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Batch-averaged SSIM via scikit-image (CPU, channel_axis=0)."""
    x_np = x.detach().cpu().clamp(0, max_val).numpy()
    y_np = y.detach().cpu().clamp(0, max_val).numpy()
    vals = []
    for i in range(x_np.shape[0]):
        vals.append(_sk_ssim(
            x_np[i], y_np[i],
            data_range=max_val, channel_axis=0,
        ))
    return torch.tensor(float(np.mean(vals)))


# ---------------------------------------------------------------------------
# LPIPS (lazy-loaded; weights pulled on first use)
# ---------------------------------------------------------------------------

_LPIPS_NET = None

def compute_lpips(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """LPIPS(x, y) with inputs in [0, 1]; internally rescaled to [-1, 1]."""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips
        _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False).eval()
        for p in _LPIPS_NET.parameters():
            p.requires_grad_(False)
    net = _LPIPS_NET.to(x.device)
    x01 = x.clamp(0, 1) * 2 - 1
    y01 = y.clamp(0, 1) * 2 - 1
    with torch.no_grad():
        return net(x01, y01).mean()


# ---------------------------------------------------------------------------
# Codebook statistics
# ---------------------------------------------------------------------------

class CodebookStats:
    """Running statistics for codebook usage and perplexity across an epoch."""

    def __init__(self, num_codes: int, device: Optional[torch.device] = None):
        self.num_codes = num_codes
        self.reset(device)

    def reset(self, device: Optional[torch.device] = None) -> None:
        dev = device if device is not None else torch.device("cpu")
        self.counts = torch.zeros(self.num_codes, dtype=torch.long, device=dev)
        self.total_tokens = 0

    @torch.no_grad()
    def update(self, indices: torch.Tensor) -> None:
        flat = indices.view(-1).to(self.counts.device)
        binned = torch.bincount(flat, minlength=self.num_codes)
        self.counts += binned
        self.total_tokens += int(flat.numel())

    @torch.no_grad()
    def summary(self) -> Dict[str, float]:
        if self.total_tokens == 0:
            return {"usage": 0.0, "perplexity": 0.0, "dead_fraction": 1.0}
        p = self.counts.float() / self.total_tokens
        p = p.clamp_min(1e-12)
        perplexity = float(torch.exp(-(p * p.log()).sum()))
        used = int((self.counts > 0).sum())
        return {
            "usage": used / self.num_codes,
            "perplexity": perplexity,
            "dead_fraction": 1.0 - used / self.num_codes,
            "active_codes": used,
        }


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

class FIDTracker:
    """Accumulates Inception-v3 pool3 features and computes FID.

    Uses the pytorch_fid implementation of InceptionV3 to match standard
    FID scores. Inputs are expected in [0, 1] range, shape (B, 3, H, W);
    any resolution is accepted (Inception resizes internally).
    """

    def __init__(self, device: torch.device, dims: int = 2048):
        from pytorch_fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.dims = dims
        self.reset()

    def reset(self) -> None:
        self._real_feats: list[np.ndarray] = []
        self._fake_feats: list[np.ndarray] = []

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.device).clamp(0, 1)
        feat = self.model(x)[0]
        if feat.shape[-1] != 1:
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
        return feat.squeeze(-1).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def update_real(self, x: torch.Tensor) -> None:
        self._real_feats.append(self._extract(x))

    @torch.no_grad()
    def update_fake(self, x: torch.Tensor) -> None:
        self._fake_feats.append(self._extract(x))

    def compute(self) -> float:
        from scipy.linalg import sqrtm
        feats_r = np.concatenate(self._real_feats, axis=0)
        feats_f = np.concatenate(self._fake_feats, axis=0)
        mu_r, sig_r = feats_r.mean(0), np.cov(feats_r, rowvar=False)
        mu_f, sig_f = feats_f.mean(0), np.cov(feats_f, rowvar=False)
        diff = mu_r - mu_f
        covmean, _ = sqrtm(sig_r.dot(sig_f), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sig_r + sig_f - 2.0 * covmean))
