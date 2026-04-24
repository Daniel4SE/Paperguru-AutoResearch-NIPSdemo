"""
VQ-AutoEncoder model: Encoder -> quant_conv -> Quantizer -> post_quant_conv -> Decoder.

Architecture follows the VQ-GAN recipe (Esser et al., CVPR 2021), simplified:
  * GroupNorm + SiLU + Conv2d residual blocks.
  * Downsample/Upsample via strided / transposed convs.
  * Optional attention at the bottleneck.
  * 1x1 pre/post-quant convs project encoder features to the quantizer
    dimension (usually much smaller than encoder width, e.g. 8).

The model is resolution-agnostic: it uses a `ch_mults` tuple to control
spatial downsampling factors, so the same code works for CIFAR-10 (32x32)
and ImageNet-256.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import BaseQuantizer, QuantizerOutput, build_quantizer


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _group_norm(ch: int, groups: int = 32) -> nn.GroupNorm:
    # Use at most `groups` groups but always divide ch evenly.
    g = min(groups, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    """Pre-activation residual block: GN -> SiLU -> Conv -> GN -> SiLU -> Conv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Non-local self-attention at the bottleneck, as in VQ-GAN."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = _group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, c, h * w).permute(0, 2, 1)        # (B, HW, C)
        k = k.view(b, c, h * w)                          # (B, C, HW)
        v = v.view(b, c, h * w).permute(0, 2, 1)        # (B, HW, C)
        attn = torch.softmax(q @ k * self.scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Conv encoder. Output shape: (B, z_channels, H / 2^(len(ch_mults)-1), W / ...)."""

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        ch_mults: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
        attn_resolutions: Sequence[int] = (),  # absolute resolution at which to apply attn
        input_resolution: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        blocks: list[nn.Module] = []
        ch = base_ch
        cur_res = input_resolution
        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, dropout=dropout))
                ch = out_ch
                if cur_res in attn_resolutions:
                    blocks.append(AttnBlock(ch))
            if i != len(ch_mults) - 1:
                blocks.append(Downsample(ch))
                cur_res //= 2
        self.blocks = nn.Sequential(*blocks)

        # Middle block with attention.
        self.mid = nn.Sequential(
            ResBlock(ch, ch, dropout=dropout),
            AttnBlock(ch),
            ResBlock(ch, ch, dropout=dropout),
        )

        self.norm_out = _group_norm(ch)
        self.conv_out = nn.Conv2d(ch, z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.blocks(h)
        h = self.mid(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        base_ch: int = 64,
        ch_mults: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
        attn_resolutions: Sequence[int] = (),
        output_resolution: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = base_ch * ch_mults[-1]
        self.conv_in = nn.Conv2d(z_channels, ch, kernel_size=3, padding=1)
        self.mid = nn.Sequential(
            ResBlock(ch, ch, dropout=dropout),
            AttnBlock(ch),
            ResBlock(ch, ch, dropout=dropout),
        )

        blocks: list[nn.Module] = []
        cur_res = output_resolution // (2 ** (len(ch_mults) - 1))
        for i, mult in enumerate(reversed(ch_mults)):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks + 1):  # +1 per VQ-GAN convention
                blocks.append(ResBlock(ch, out_ch, dropout=dropout))
                ch = out_ch
                if cur_res in attn_resolutions:
                    blocks.append(AttnBlock(ch))
            if i != len(ch_mults) - 1:
                blocks.append(Upsample(ch))
                cur_res *= 2
        self.blocks = nn.Sequential(*blocks)

        self.norm_out = _group_norm(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.blocks(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# Full VQ-Autoencoder
# ---------------------------------------------------------------------------

@dataclass
class VQAEOutput:
    x_hat: torch.Tensor
    z_e: torch.Tensor            # encoder output, before quantizer_conv
    z_q: torch.Tensor            # post-quantizer (low-dim) tensor
    indices: Optional[torch.Tensor]
    stats: Dict[str, torch.Tensor]


class VQAutoEncoder(nn.Module):
    """Encoder + 1x1 quant projections + Quantizer + Decoder."""

    def __init__(
        self,
        quantizer_cfg: Dict,
        encoder_cfg: Optional[Dict] = None,
        decoder_cfg: Optional[Dict] = None,
        embed_dim: int = 8,        # dim going INTO the quantizer
    ):
        super().__init__()
        encoder_cfg = dict(encoder_cfg or {})
        decoder_cfg = dict(decoder_cfg or {})

        # Infer input/output resolution from whichever side is provided.
        if "input_resolution" not in encoder_cfg:
            encoder_cfg["input_resolution"] = decoder_cfg.get("output_resolution", 32)
        if "output_resolution" not in decoder_cfg:
            decoder_cfg["output_resolution"] = encoder_cfg["input_resolution"]

        self.encoder = Encoder(**encoder_cfg)
        self.decoder = Decoder(**decoder_cfg)

        z_channels = encoder_cfg.get("z_channels", 256)
        self.quant_conv = nn.Conv2d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)

        # Sync quantizer dim with embed_dim for codebook-based quantizers.
        if quantizer_cfg["type"].lower() not in ("fsq",):
            quantizer_cfg = dict(quantizer_cfg)
            quantizer_cfg["dim"] = embed_dim
        self.quantizer: BaseQuantizer = build_quantizer(quantizer_cfg)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, QuantizerOutput]:
        h = self.encoder(x)
        z_e = self.quant_conv(h)
        qout = self.quantizer(z_e)
        return z_e, qout

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        h = self.post_quant_conv(z_q)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> VQAEOutput:
        z_e, qout = self.encode(x)
        x_hat = self.decode(qout.z_q)
        return VQAEOutput(
            x_hat=x_hat, z_e=z_e, z_q=qout.z_q,
            indices=qout.indices, stats=qout.stats,
        )

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x).x_hat


def build_vqae(cfg: Dict) -> VQAutoEncoder:
    """Build VQAE from the 'model' subsection of a config dict."""
    return VQAutoEncoder(
        quantizer_cfg=cfg["quantizer"],
        encoder_cfg=cfg.get("encoder", {}),
        decoder_cfg=cfg.get("decoder", {}),
        embed_dim=cfg.get("embed_dim", 8),
    )
