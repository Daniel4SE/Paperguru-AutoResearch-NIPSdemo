#!/usr/bin/env python
"""Training entry point for VQ-AutoEncoder experiments.

Usage:
    python train.py --config configs/cifar10_rotation.yaml
    python train.py --config configs/cifar10_vanilla.yaml --override train.max_steps=5000
    python train.py --config configs/cifar10_rotation.yaml --device cpu --smoke

All experiments (baselines and our method) share this script; quantizer
selection and hyperparameters live entirely in the YAML config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.vqae import build_vqae
from models.quantizers import GumbelVQ
from eval.metrics import (
    compute_psnr, compute_ssim_batch, compute_lpips, CodebookStats,
)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

def load_config(path: str, overrides: list[str]) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got: {ov}")
        key, val = ov.split("=", 1)
        # Cast to int/float/bool if possible.
        for caster in (int, float):
            try:
                val = caster(val)
                break
            except ValueError:
                pass
        if isinstance(val, str) and val.lower() in ("true", "false"):
            val = val.lower() == "true"
        _set_nested(cfg, key.split("."), val)
    return cfg


def _set_nested(d: dict, keys: list[str], val) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    import torchvision
    from torchvision import transforms

    name = cfg["data"]["name"].lower()
    root = os.path.expanduser(cfg["data"].get("root", "~/vq-rotation/data"))
    img_size = cfg["data"]["image_size"]
    batch = cfg["train"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    if name == "cifar10":
        # Load entire CIFAR-10 into GPU-pinned fp32 tensors; 50k*32*32*3*4 = 600MB
        # and val is 120MB. This removes the CPU PIL+ToTensor bottleneck that
        # otherwise caps H100 utilisation at <15%.
        tr_raw = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
        val_raw = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

        def _to_tensor(raw):
            # raw.data is uint8 (N, 32, 32, 3); targets is a list[int]
            import numpy as np
            arr = torch.from_numpy(np.ascontiguousarray(raw.data)).permute(0, 3, 1, 2)
            arr = arr.float().div_(255.0)
            if img_size != 32:
                arr = torch.nn.functional.interpolate(
                    arr, size=img_size, mode="bilinear", align_corners=False,
                )
            targets = torch.tensor(raw.targets, dtype=torch.long)
            return torch.utils.data.TensorDataset(arr, targets)

        tr_set = _to_tensor(tr_raw)
        val_set = _to_tensor(val_raw)
    elif name == "imagefolder":
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        tr_set = torchvision.datasets.ImageFolder(
            os.path.join(root, "train"), transform=tfm,
        )
        val_set = torchvision.datasets.ImageFolder(
            os.path.join(root, "val"), transform=tfm,
        )
    else:
        raise ValueError(f"unknown dataset {name}")

    tr = DataLoader(
        tr_set, batch_size=batch, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val = DataLoader(
        val_set, batch_size=batch, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return tr, val


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    x: torch.Tensor, x_hat: torch.Tensor, kind: str = "l1"
) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(x_hat, x)
    if kind == "l2":
        return F.mse_loss(x_hat, x)
    if kind == "charbonnier":
        eps = 1e-3
        return torch.sqrt((x_hat - x) ** 2 + eps ** 2).mean()
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any], device: torch.device, smoke: bool = False) -> None:
    out_dir = Path(cfg["logging"]["out_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    torch.manual_seed(cfg.get("seed", 0))
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # Data
    tr_loader, val_loader = build_dataloaders(cfg)
    print(f"[data] train={len(tr_loader.dataset)} val={len(val_loader.dataset)}")

    # Model
    model = build_vqae(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {n_params:.2f}M params; quantizer={cfg['model']['quantizer']['type']}")

    # Optim
    opt_cfg = cfg["train"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.99))),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=opt_cfg["max_steps"], eta_min=opt_cfg.get("lr_min", 0.0),
    )
    use_amp = opt_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Loss mix
    loss_kind = opt_cfg.get("recon_loss", "l1")
    lpips_w = opt_cfg.get("lpips_weight", 0.0)

    # Run
    step = 0
    max_steps = opt_cfg["max_steps"] if not smoke else 20
    log_every = opt_cfg.get("log_every", 50)
    val_every = opt_cfg.get("val_every", 1000)
    save_every = opt_cfg.get("save_every", 5000)

    t0 = time.time()
    model.train()
    done = False
    while not done:
        for x, _ in tr_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                recon = reconstruction_loss(x, out.x_hat.clamp(0, 1), kind=loss_kind)
                commit = out.stats.get("commit_loss", torch.tensor(0.0, device=device))
                entropy = out.stats.get("entropy_loss", torch.tensor(0.0, device=device))
                loss = recon + commit + entropy
                if lpips_w > 0:
                    lp = compute_lpips(x, out.x_hat.clamp(0, 1))
                    loss = loss + lpips_w * lp

            optim.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt_cfg.get("grad_clip", 1.0)
                )
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt_cfg.get("grad_clip", 1.0)
                )
                optim.step()
            sched.step()

            # Gumbel tau annealing
            if isinstance(model.quantizer, GumbelVQ):
                model.quantizer.set_tau(step)

            if step % log_every == 0:
                with torch.no_grad():
                    psnr = compute_psnr(x, out.x_hat.clamp(0, 1)).item()
                elapsed = time.time() - t0
                print(
                    f"[{step:6d}/{max_steps}] "
                    f"loss={loss.item():.4f} recon={recon.item():.4f} "
                    f"commit={commit.item():.4f} "
                    f"psnr={psnr:.2f}dB "
                    f"usage={out.stats['usage'].item():.3f} "
                    f"perp={out.stats['perplexity'].item():.1f} "
                    f"lr={sched.get_last_lr()[0]:.2e} "
                    f"{elapsed:.1f}s"
                )
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/recon", recon.item(), step)
                writer.add_scalar("train/commit", commit.item(), step)
                writer.add_scalar("train/psnr", psnr, step)
                writer.add_scalar("train/usage", out.stats["usage"].item(), step)
                writer.add_scalar("train/perplexity", out.stats["perplexity"].item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)

            if step > 0 and step % val_every == 0:
                validate(model, val_loader, device, writer, step, cfg)
                model.train()

            if step > 0 and step % save_every == 0:
                ckpt = {
                    "step": step, "model": model.state_dict(),
                    "optim": optim.state_dict(), "cfg": cfg,
                }
                torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")

            step += 1
            if step >= max_steps:
                done = True
                break

    # Final validation + save
    validate(model, val_loader, device, writer, step, cfg)
    torch.save(
        {"step": step, "model": model.state_dict(), "cfg": cfg},
        out_dir / "ckpt_final.pt",
    )
    writer.close()
    print(f"[done] total steps={step} elapsed={time.time()-t0:.1f}s")


@torch.no_grad()
def validate(model, loader, device, writer, step, cfg) -> None:
    model.eval()
    num_codes = getattr(model.quantizer, "num_codes", 1)
    cs = CodebookStats(num_codes)
    psnrs, ssims, lpipses, recons = [], [], [], []
    t0 = time.time()
    for i, (x, _) in enumerate(loader):
        if i >= cfg["train"].get("max_val_batches", 20):
            break
        x = x.to(device, non_blocking=True)
        out = model(x)
        x_hat = out.x_hat.clamp(0, 1)
        psnrs.append(compute_psnr(x, x_hat).item())
        ssims.append(compute_ssim_batch(x, x_hat).item())
        if cfg["train"].get("val_lpips", False):
            lpipses.append(compute_lpips(x, x_hat).item())
        recons.append(F.l1_loss(x_hat, x).item())
        if out.indices is not None:
            cs.update(out.indices)
    cb = cs.summary()
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    print(
        f"[val {step}] "
        f"recon_l1={mean(recons):.4f} psnr={mean(psnrs):.2f}dB "
        f"ssim={mean(ssims):.4f} "
        + (f"lpips={mean(lpipses):.4f} " if lpipses else "")
        + f"usage={cb['usage']:.3f} perp={cb['perplexity']:.1f} "
        f"({time.time()-t0:.1f}s)"
    )
    writer.add_scalar("val/psnr", mean(psnrs), step)
    writer.add_scalar("val/ssim", mean(ssims), step)
    writer.add_scalar("val/recon_l1", mean(recons), step)
    writer.add_scalar("val/usage", cb["usage"], step)
    writer.add_scalar("val/perplexity", cb["perplexity"], step)
    if lpipses:
        writer.add_scalar("val/lpips", mean(lpipses), step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[],
                   help="dot-path=value overrides, e.g. train.lr=5e-5")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--smoke", action="store_true",
                   help="run only a few steps to verify the loop")
    args = p.parse_args()

    cfg = load_config(args.config, args.override)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    train(cfg, device, smoke=args.smoke)


if __name__ == "__main__":
    main()
