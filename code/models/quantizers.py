"""
Vector quantization layers for the rotation-aware gradient estimation study.

This file provides a unified interface `BaseQuantizer` with four concrete
implementations that can be swapped via configuration:

    * VanillaVQ  -- classic VQ-VAE with Straight-Through Estimator (STE).
    * RotationVQ -- proposed method: Householder-reflection-based rotation
                    plus scalar rescaling that reshapes the gradient
                    without changing the forward output.
    * FSQ        -- Finite Scalar Quantization (Mentzer et al., ICLR 2024),
                    codebook-free baseline.
    * GumbelVQ   -- Gumbel-Softmax / Concrete-relaxation VQ baseline.

All quantizers share the signature:

    z_q, indices, stats = quantizer(z_e)

where `z_e` is a tensor of shape (B, D, H, W) produced by an image encoder,
`z_q` is the (differentiable surrogate of the) quantized output with the
same shape, `indices` are the integer codebook indices (shape (B, H, W))
or `None` for FSQ-style codebook-less quantizers, and `stats` is a dict
of scalar losses and diagnostics (commitment loss, codebook loss,
perplexity, codebook usage).

Design principles:
  * All forward outputs are numerically identical to hard quantization,
    so that a frozen tokenizer used by a downstream autoregressive model
    sees discrete tokens.
  * Gradient behaviour differs per quantizer, which is exactly the
    subject of the study.
  * EMA codebook updates and dead-code reinitialisation are implemented
    in a single reusable `EMACodebook` module consumed by VanillaVQ and
    RotationVQ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_spatial(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """(B, D, H, W) -> (B*H*W, D). Returns flat tensor + original shape."""
    b, d, h, w = z.shape
    z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, d)
    return z_flat, (b, d, h, w)


def _unflatten_spatial(z_flat: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
    b, d, h, w = shape
    return z_flat.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()


def _compute_perplexity(indices: torch.Tensor, num_codes: int) -> torch.Tensor:
    """Codebook perplexity = exp(-sum p_k log p_k). Higher = more uniform usage."""
    with torch.no_grad():
        one_hot = F.one_hot(indices.view(-1), num_codes).float()
        avg = one_hot.mean(dim=0)
        return torch.exp(-(avg * (avg + 1e-10).log()).sum())


def _codebook_usage(indices: torch.Tensor, num_codes: int) -> torch.Tensor:
    """Fraction of codes that were used at least once in this batch."""
    with torch.no_grad():
        used = torch.unique(indices.view(-1)).numel()
        return torch.tensor(used / num_codes, device=indices.device)


# ---------------------------------------------------------------------------
# EMA codebook (used by both VanillaVQ and RotationVQ)
# ---------------------------------------------------------------------------

class EMACodebook(nn.Module):
    """Codebook with exponential-moving-average updates and dead-code reinit.

    Closely follows van den Oord et al. (2017), with optional dead-code
    reinitialisation from the current batch as in Razavi et al. (2019).
    """

    def __init__(
        self,
        num_codes: int,
        dim: int,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.ema_decay = ema_decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(num_codes, dim) * 0.01
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _data_dependent_init(self, z_flat: torch.Tensor) -> None:
        """Initialise codebook from the first batch for stable starts.

        Codebook buffers are float32 regardless of upstream AMP dtype.
        """
        if self.initialized.item():
            return
        z_flat = z_flat.to(self.embedding.dtype)
        n = z_flat.shape[0]
        if n >= self.num_codes:
            idx = torch.randperm(n, device=z_flat.device)[: self.num_codes]
            self.embedding.copy_(z_flat[idx])
        else:
            reps = (self.num_codes + n - 1) // n
            tiled = z_flat.repeat(reps, 1)[: self.num_codes]
            self.embedding.copy_(tiled + 1e-3 * torch.randn_like(tiled))
        self.embed_avg.copy_(self.embedding)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def lookup(self, z_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest-neighbour lookup. Returns (quantized, indices)."""
        # squared L2: ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2
        a2 = (z_flat ** 2).sum(dim=1, keepdim=True)              # (N, 1)
        b2 = (self.embedding ** 2).sum(dim=1)                    # (K,)
        ab = z_flat @ self.embedding.t()                         # (N, K)
        dist = a2 - 2 * ab + b2.unsqueeze(0)
        indices = dist.argmin(dim=1)                             # (N,)
        quantized = F.embedding(indices, self.embedding)         # (N, D)
        return quantized, indices

    @torch.no_grad()
    def ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA update of cluster sizes and codebook vectors.

        Codebook buffers live in float32 for numerical stability even
        when the surrounding training loop uses fp16/bf16 AMP; we cast
        ``z_flat`` to the codebook dtype on entry.
        """
        ref_dtype = self.embedding.dtype
        z_flat = z_flat.to(ref_dtype)

        one_hot = F.one_hot(indices, self.num_codes).to(ref_dtype)   # (N, K)
        cluster_counts = one_hot.sum(dim=0)                          # (K,)
        embed_sum = one_hot.t() @ z_flat                             # (K, D)

        self.cluster_size.mul_(self.ema_decay).add_(
            cluster_counts, alpha=1.0 - self.ema_decay
        )
        self.embed_avg.mul_(self.ema_decay).add_(
            embed_sum, alpha=1.0 - self.ema_decay
        )

        # Laplace smoothing to avoid divide-by-zero.
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.embedding.copy_(self.embed_avg / smoothed.unsqueeze(1))

        # Dead-code reinit: codes with very low cluster_size are resampled
        # from the current batch.
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead > 0 and z_flat.shape[0] >= n_dead:
            idx = torch.randperm(z_flat.shape[0], device=z_flat.device)[:n_dead]
            self.embedding[dead] = z_flat[idx]
            self.embed_avg[dead] = z_flat[idx]
            self.cluster_size[dead] = 1.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class QuantizerOutput:
    """Return value of every quantizer's forward method."""
    z_q: torch.Tensor
    indices: Optional[torch.Tensor]
    stats: Dict[str, torch.Tensor]


class BaseQuantizer(nn.Module):
    """Common interface.

    Subclasses must implement `forward(z_e) -> QuantizerOutput`.
    """

    num_codes: int  # for logging; FSQ sets this to its implicit grid size

    def forward(self, z_e: torch.Tensor) -> QuantizerOutput:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# VanillaVQ -- classic STE baseline (van den Oord et al., 2017)
# ---------------------------------------------------------------------------

class VanillaVQ(BaseQuantizer):
    """VQ with standard Straight-Through Estimator.

    Forward:  z_q = q  (nearest codebook vector)
    Backward: dL/dz_e = dL/dz_q  (identity copy; see Bengio et al., 2013)
    """

    def __init__(
        self,
        num_codes: int = 8192,
        dim: int = 8,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = commitment_beta
        self.codebook = EMACodebook(num_codes, dim, ema_decay=ema_decay)

    def forward(self, z_e: torch.Tensor) -> QuantizerOutput:
        z_flat, shape = _flatten_spatial(z_e)

        if self.training:
            self.codebook._data_dependent_init(z_flat.detach())

        q_flat, idx_flat = self.codebook.lookup(z_flat.detach())

        # Straight-through: z_q forward = q, grad flows identity to z_e.
        z_q_flat = z_flat + (q_flat - z_flat).detach()

        commit = F.mse_loss(z_flat, q_flat.detach())

        if self.training and self.codebook.initialized.item():
            self.codebook.ema_update(z_flat.detach(), idx_flat)

        z_q = _unflatten_spatial(z_q_flat, shape)
        indices = idx_flat.view(shape[0], shape[2], shape[3])

        stats = {
            "commit_loss": self.beta * commit,
            "codebook_loss": torch.zeros((), device=z_e.device),  # EMA handles codebook
            "perplexity": _compute_perplexity(indices, self.num_codes),
            "usage": _codebook_usage(indices, self.num_codes),
        }
        return QuantizerOutput(z_q=z_q, indices=indices, stats=stats)


# ---------------------------------------------------------------------------
# RotationVQ -- our proposed method
# ---------------------------------------------------------------------------

class RotationVQ(BaseQuantizer):
    """Rotation-aware gradient estimation for VQ.

    Core idea: replace the identity STE with a transformation that, in the
    forward pass, is numerically identical to returning the codebook
    vector q, but whose *Jacobian* with respect to z_e encodes the
    angular relationship between z_e and q. Concretely,

        q_tilde = sg[s * R] * z_e

    where
        R = I - 2 v v^T   with   v = (z_hat - q_hat) / ||z_hat - q_hat||
        s = ||q|| / ||z_e||
        z_hat, q_hat = z_e / ||z_e||,  q / ||q||
        sg[.] is the stop-gradient operator.

    Because s*R maps z_e to q exactly (R reflects z_hat onto q_hat, and s
    rescales the norm), forward output equals q up to numerical precision.
    Because s and R are stop-gradients, the backward Jacobian is simply
    s*R, which is a rotation+scaling of the upstream gradient and thus
    preserves angular information between z_e and q.

    Ablation modes (set via `mode`):
        "full"         : rotation + rescaling           (proposed)
        "no_rotation"  : rescaling only, R = I          (ablation)
        "no_rescale"   : rotation only, s = 1           (ablation)
        "ste"          : equivalent to VanillaVQ        (sanity check)
    """

    def __init__(
        self,
        num_codes: int = 8192,
        dim: int = 8,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        mode: str = "full",
        eps: float = 1e-6,
        entropy_weight: float = 0.0,
        entropy_tau: float = 1.0,
    ):
        super().__init__()
        assert mode in ("full", "no_rotation", "no_rescale", "ste")
        self.num_codes = num_codes
        self.dim = dim
        self.beta = commitment_beta
        self.mode = mode
        self.eps = eps
        # Entropy regulariser: add -lambda * H(<p>) to the loss so that
        # the batch-averaged soft assignment distribution stays close to
        # uniform. Counteracts the R-induced positive-feedback collapse.
        self.entropy_weight = entropy_weight
        self.entropy_tau = entropy_tau
        self.codebook = EMACodebook(num_codes, dim, ema_decay=ema_decay)

    def _householder_apply(
        self,
        z_flat: torch.Tensor,
        q_flat: torch.Tensor,
        detach_R: bool = True,
    ) -> torch.Tensor:
        """Return R @ z_flat where R reflects z_hat onto q_hat, per row.

        Implemented implicitly as (I - 2 v v^T) x = x - 2 (v . x) v, so no
        D x D matrix is materialised. When ``detach_R=True`` the normal
        vector ``v`` is computed from a detached copy of z_flat, so the
        backward Jacobian is *exactly* the (constant) reflection matrix
        R applied to the upstream gradient; no extra terms from
        differentiating v through z_flat are introduced. This is the
        behaviour required by the paper's gradient-flow analysis.

        Shapes:
            z_flat, q_flat : (N, D)
            returns        : (N, D)
        """
        if detach_R:
            z_for_v = z_flat.detach()
        else:
            z_for_v = z_flat

        z_norm = z_for_v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        q_norm = q_flat.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        z_hat = z_for_v / z_norm
        q_hat = q_flat / q_norm

        # v = (z_hat - q_hat); reflecting z_hat about this hyperplane
        # sends z_hat -> q_hat exactly.
        v = z_hat - q_hat
        v_norm = v.norm(dim=-1, keepdim=True)

        # If v ~ 0, z_hat already equals q_hat; R is effectively the
        # identity. Use a safe divisor and a zero mask.
        safe = v_norm > self.eps
        v = torch.where(safe, v / v_norm.clamp_min(self.eps), torch.zeros_like(v))

        dot = (v * z_flat).sum(dim=-1, keepdim=True)
        return z_flat - 2.0 * dot * v

    def forward(self, z_e: torch.Tensor) -> QuantizerOutput:
        z_flat, shape = _flatten_spatial(z_e)

        if self.training:
            self.codebook._data_dependent_init(z_flat.detach())

        q_flat, idx_flat = self.codebook.lookup(z_flat.detach())

        if self.mode == "ste":
            # Classic STE: forward = q, backward = identity.
            z_q_flat = z_flat + (q_flat - z_flat).detach()
        else:
            # Scaling factor s = ||q|| / ||z_e||; treated as a constant
            # during backprop via .detach().
            if self.mode in ("full", "no_rotation"):
                scale = (
                    q_flat.norm(dim=-1, keepdim=True)
                    / z_flat.norm(dim=-1, keepdim=True).clamp_min(self.eps)
                ).detach()
            else:  # no_rescale
                scale = torch.ones_like(z_flat[:, :1])

            # Gradient carrier: the term that actually produces the
            # backward Jacobian we want. The Householder reflection is
            # computed with detach_R=True so that the backward Jacobian
            # of `carrier` w.r.t. z_flat is exactly (scale * R), where R
            # is the stop-gradient Householder matrix.
            if self.mode == "full":
                carrier = scale * self._householder_apply(
                    z_flat, q_flat.detach(), detach_R=True
                )
            elif self.mode == "no_rotation":
                carrier = scale * z_flat
            else:  # no_rescale
                carrier = self._householder_apply(
                    z_flat, q_flat.detach(), detach_R=True
                )

            # Straight-through correction: force forward output to be
            # *numerically exact* q_flat, regardless of finite-precision
            # errors in the reflection (which can blow up when z and q
            # are nearly parallel and u_norm is tiny). Backward gradient
            # is unaffected because the correction is detached.
            z_q_flat = carrier + (q_flat - carrier).detach()

        commit = F.mse_loss(z_flat, q_flat.detach())

        # Soft-assignment entropy regulariser (Baykal et al. 2023; also
        # standard in mixture-of-experts load-balancing losses).
        # p_ik = softmax(-||z_i - e_k||^2 / tau) over codes; the batch-
        # mean distribution <p>_k should stay close to uniform. We add
        # -lambda * H(<p>) to the total loss so that high entropy
        # (uniform use) is encouraged. The embedding table is frozen
        # during this computation (EMA-updated separately); only z flows
        # gradient, and therefore pushes the encoder away from
        # already-crowded codes.
        if self.entropy_weight > 0:
            codebook = self.codebook.embedding.detach().clone()
            # squared L2: (N, K)
            z2 = (z_flat.detach() ** 2).sum(-1, keepdim=True)
            e2 = (codebook ** 2).sum(-1)
            ze = z_flat @ codebook.t()
            dist2 = z2 - 2 * ze + e2.unsqueeze(0)
            logits = -dist2 / max(self.entropy_tau, self.eps)
            # softmax over codes → (N, K)
            soft = F.softmax(logits, dim=-1)
            p_bar = soft.mean(dim=0).clamp_min(self.eps)  # (K,)
            neg_entropy = (p_bar * p_bar.log()).sum()     # ≤ 0
            entropy_loss = self.entropy_weight * neg_entropy
        else:
            entropy_loss = torch.zeros((), device=z_e.device)

        if self.training and self.codebook.initialized.item():
            self.codebook.ema_update(z_flat.detach(), idx_flat)

        z_q = _unflatten_spatial(z_q_flat, shape)
        indices = idx_flat.view(shape[0], shape[2], shape[3])

        stats = {
            "commit_loss": self.beta * commit,
            "codebook_loss": torch.zeros((), device=z_e.device),
            "entropy_loss": entropy_loss,
            "perplexity": _compute_perplexity(indices, self.num_codes),
            "usage": _codebook_usage(indices, self.num_codes),
        }
        return QuantizerOutput(z_q=z_q, indices=indices, stats=stats)


# ---------------------------------------------------------------------------
# FSQ -- Finite Scalar Quantization (Mentzer et al., ICLR 2024)
# ---------------------------------------------------------------------------

class FSQ(BaseQuantizer):
    """Finite Scalar Quantization.

    Projects each channel of z_e independently onto a small set of
    scalar levels via a bounded non-linearity (tanh) followed by round().
    No codebook is learned; the implicit codebook is the Cartesian
    product of per-channel level sets.
    """

    def __init__(self, levels: Tuple[int, ...] = (8, 8, 8, 5, 5, 5)):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        self.num_codes = int(torch.prod(torch.tensor(levels)).item())
        self.register_buffer(
            "_levels_t", torch.tensor(levels, dtype=torch.float32)
        )

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Squash each channel into [-(L-1)/2, (L-1)/2]."""
        half_l = (self._levels_t - 1) / 2.0
        return torch.tanh(z) * half_l.view(1, -1, 1, 1)

    def forward(self, z_e: torch.Tensor) -> QuantizerOutput:
        assert z_e.shape[1] == self.dim, (
            f"FSQ expects {self.dim} channels, got {z_e.shape[1]}."
        )
        z_bounded = self._bound(z_e)
        z_q = z_bounded + (z_bounded.round() - z_bounded).detach()  # STE round

        # Encode discrete index for downstream AR models / perplexity.
        # Map z_q in [-(L-1)/2, (L-1)/2] back to integer levels [0, L-1].
        # Clamp per-channel to guard against FP drift at the tanh boundaries.
        z_shifted = (z_q + (self._levels_t - 1).view(1, -1, 1, 1) / 2.0).round().long()
        max_per_ch = (self._levels_t.long() - 1).view(1, -1, 1, 1)
        z_shifted = z_shifted.clamp(min=0).minimum(max_per_ch)
        strides = torch.ones(self.dim, dtype=torch.long, device=z_q.device)
        for i in range(self.dim - 2, -1, -1):
            strides[i] = strides[i + 1] * int(self.levels[i + 1])
        indices = (z_shifted * strides.view(1, -1, 1, 1)).sum(dim=1)
        indices = indices.clamp(0, self.num_codes - 1)

        stats = {
            "commit_loss": torch.zeros((), device=z_e.device),
            "codebook_loss": torch.zeros((), device=z_e.device),
            "perplexity": _compute_perplexity(indices, self.num_codes),
            "usage": _codebook_usage(indices, self.num_codes),
        }
        return QuantizerOutput(z_q=z_q, indices=indices, stats=stats)


# ---------------------------------------------------------------------------
# GumbelVQ -- Gumbel-Softmax relaxation (Jang et al., 2017)
# ---------------------------------------------------------------------------

class GumbelVQ(BaseQuantizer):
    """VQ with Gumbel-Softmax categorical reparameterisation.

    During training: z_q = sum_k p_k * e_k with soft p from Gumbel-Softmax.
    During eval:    z_q = e_argmax(p).
    The codebook is learned directly via backprop (no EMA).
    """

    def __init__(
        self,
        num_codes: int = 8192,
        dim: int = 8,
        commitment_beta: float = 0.25,
        tau: float = 1.0,
        tau_min: float = 0.5,
        anneal_rate: float = 0.00003,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = commitment_beta
        self.tau = tau
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.embedding = nn.Parameter(torch.randn(num_codes, dim) * 0.01)
        self.proj = nn.Conv2d(dim, num_codes, kernel_size=1, bias=False)

    def set_tau(self, step: int) -> None:
        self.tau = max(self.tau_min, math.exp(-self.anneal_rate * step))

    def forward(self, z_e: torch.Tensor) -> QuantizerOutput:
        logits = self.proj(z_e)                 # (B, K, H, W)
        b, k, h, w = logits.shape

        if self.training:
            soft = F.gumbel_softmax(
                logits.permute(0, 2, 3, 1).reshape(-1, k),
                tau=self.tau,
                hard=True,
            )                                    # (B*H*W, K)
            z_q_flat = soft @ self.embedding     # (B*H*W, D)
            indices = soft.argmax(dim=-1)
        else:
            indices = logits.argmax(dim=1).reshape(-1)
            z_q_flat = F.embedding(indices, self.embedding)

        z_q = z_q_flat.view(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
        indices = indices.view(b, h, w)

        # A light L2 pull-in so that codebook vectors stay matched to
        # encoder statistics.
        z_flat, _ = _flatten_spatial(z_e)
        q_flat = F.embedding(indices.view(-1), self.embedding)
        commit = F.mse_loss(z_flat, q_flat.detach())

        stats = {
            "commit_loss": self.beta * commit,
            "codebook_loss": torch.zeros((), device=z_e.device),
            "perplexity": _compute_perplexity(indices, self.num_codes),
            "usage": _codebook_usage(indices, self.num_codes),
            "tau": torch.tensor(self.tau, device=z_e.device),
        }
        return QuantizerOutput(z_q=z_q, indices=indices, stats=stats)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_quantizer(cfg: Dict) -> BaseQuantizer:
    """Construct a quantizer from a config dict.

    Example cfg:
        {"type": "rotation", "num_codes": 8192, "dim": 8, "mode": "full"}
    """
    t = cfg["type"].lower()
    if t in ("vanilla", "vq", "ste"):
        return VanillaVQ(
            num_codes=cfg.get("num_codes", 8192),
            dim=cfg.get("dim", 8),
            commitment_beta=cfg.get("commitment_beta", 0.25),
            ema_decay=cfg.get("ema_decay", 0.99),
        )
    if t in ("rotation", "rot"):
        return RotationVQ(
            num_codes=cfg.get("num_codes", 8192),
            dim=cfg.get("dim", 8),
            commitment_beta=cfg.get("commitment_beta", 0.25),
            ema_decay=cfg.get("ema_decay", 0.99),
            mode=cfg.get("mode", "full"),
            entropy_weight=cfg.get("entropy_weight", 0.0),
            entropy_tau=cfg.get("entropy_tau", 1.0),
        )
    if t == "fsq":
        return FSQ(levels=tuple(cfg.get("levels", (8, 8, 8, 5, 5, 5))))
    if t in ("gumbel", "gumbel_vq"):
        return GumbelVQ(
            num_codes=cfg.get("num_codes", 8192),
            dim=cfg.get("dim", 8),
            commitment_beta=cfg.get("commitment_beta", 0.25),
            tau=cfg.get("tau", 1.0),
            tau_min=cfg.get("tau_min", 0.5),
            anneal_rate=cfg.get("anneal_rate", 3e-5),
        )
    raise ValueError(f"Unknown quantizer type: {cfg['type']}")
