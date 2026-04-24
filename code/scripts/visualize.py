"""Generate paper-quality PNG figures from TensorBoard events and checkpoints.

Outputs under ~/vq-rotation/results/figures/:
  * curves_<metric>.png    : training/validation curves across runs
  * recon_case_<run>.png   : 2x8 grid of original + reconstruction for 8 val images
  * codebook_usage.png     : usage trajectory across runs
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter  # for version alignment
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path(os.path.expanduser("~/vq-rotation/results"))
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# TB events -> dict[run][metric] -> (steps, values)
# ---------------------------------------------------------------------------

def load_all_runs(run_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    runs: Dict[str, Dict[str, np.ndarray]] = {}
    for name in run_names:
        tb_dir = RESULTS / name / "tb"
        if not tb_dir.exists():
            print(f"[warn] {tb_dir} missing, skipping")
            continue
        # Aggregate across event files in the same tb dir.
        steps = defaultdict(list)
        values = defaultdict(list)
        for ev_path in sorted(glob.glob(str(tb_dir / "events.out.tfevents.*"))):
            acc = EventAccumulator(ev_path)
            try:
                acc.Reload()
            except Exception as e:
                print(f"[warn] failed to load {ev_path}: {e}")
                continue
            for tag in acc.Tags().get("scalars", []):
                events = acc.Scalars(tag)
                for e in events:
                    steps[tag].append(e.step)
                    values[tag].append(e.value)
        merged: Dict[str, np.ndarray] = {}
        for tag in steps:
            order = np.argsort(steps[tag])
            merged[tag] = (
                np.array(steps[tag])[order],
                np.array(values[tag])[order],
            )
        runs[name] = merged
    return runs


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_curve(runs, metric: str, ylabel: str, fname: str, ylog: bool = False):
    plt.figure(figsize=(6, 3.5))
    plotted = 0
    for name, data in runs.items():
        if metric not in data:
            continue
        steps, vals = data[metric]
        label = name.replace("cifar10_", "").replace("_e1", "")
        plt.plot(steps, vals, label=label, linewidth=1.5)
        plotted += 1
    if plotted == 0:
        print(f"[skip] no data for {metric}")
        plt.close()
        return
    plt.xlabel("training step")
    plt.ylabel(ylabel)
    if ylog:
        plt.yscale("log")
    plt.legend(frameon=False, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Reconstruction case study: load ckpt, reconstruct 8 val images
# ---------------------------------------------------------------------------

def recon_case(ckpt_path: Path, title: str, fname: str, n_images: int = 8):
    from models.vqae import build_vqae
    import yaml

    if not ckpt_path.exists():
        print(f"[skip recon] {ckpt_path} not found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_vqae(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load CIFAR-10 val
    val = torchvision.datasets.CIFAR10(
        root=os.path.expanduser(cfg["data"]["root"]), train=False, download=False,
    )
    arr = torch.from_numpy(val.data[:n_images]).permute(0, 3, 1, 2).float() / 255.0
    arr = arr.to(device)

    with torch.no_grad():
        out = model(arr)
        x_hat = out.x_hat.clamp(0, 1).cpu()
        orig = arr.cpu()

    # Build 2 x n figure
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3.2))
    for i in range(n_images):
        axes[0, i].imshow(orig[i].permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=11)
    # re-enable y labels (imshow off removes them)
    for r, lbl in enumerate(("Original", "Recon.")):
        axes[r, 0].axis("on")
        axes[r, 0].set_yticks([])
        axes[r, 0].set_xticks([])
        axes[r, 0].set_ylabel(lbl, fontsize=11)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Also report PSNR for this batch
    mse = ((orig - x_hat) ** 2).mean(dim=(1, 2, 3))
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse.clamp_min(1e-10))
    print(f"[recon] {out}  per-image PSNR: {psnr.numpy().round(2)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+",
        default=[
            "smoke_rotation",
            "bench_bs128", "bench_bs256", "bench_bs512",
            "cifar10_rotation_e1", "cifar10_vanilla_e1",
        ],
        help="subdirs under results/",
    )
    args = ap.parse_args()

    runs = load_all_runs(args.runs)
    print(f"loaded {len(runs)} runs: {list(runs.keys())}")

    # Training curves
    plot_curve(runs, "train/loss",        "train loss",        "curve_train_loss.png", ylog=True)
    plot_curve(runs, "train/recon",       "recon loss (L1)",   "curve_train_recon.png", ylog=True)
    plot_curve(runs, "train/psnr",        "train PSNR (dB)",   "curve_train_psnr.png")
    plot_curve(runs, "train/usage",       "codebook usage",    "curve_train_usage.png")
    plot_curve(runs, "train/perplexity",  "codebook perplexity", "curve_train_perplexity.png")
    plot_curve(runs, "val/psnr",          "val PSNR (dB)",     "curve_val_psnr.png")
    plot_curve(runs, "val/ssim",          "val SSIM",          "curve_val_ssim.png")
    plot_curve(runs, "val/usage",         "val codebook usage","curve_val_usage.png")

    # Recon case study using smoke_rotation checkpoint (the most-trained one)
    recon_case(
        RESULTS / "smoke_rotation" / "ckpt_final.pt",
        "RotationVQ (500 steps CIFAR-10)", "recon_smoke_rotation.png",
    )
    for bs in (128, 256, 512):
        recon_case(
            RESULTS / f"bench_bs{bs}" / "ckpt_final.pt",
            f"RotationVQ bench bs={bs} (200 steps)", f"recon_bench_bs{bs}.png",
        )

    print("\n=== figures ===")
    for p in sorted(FIG_DIR.iterdir()):
        print(p.name, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
