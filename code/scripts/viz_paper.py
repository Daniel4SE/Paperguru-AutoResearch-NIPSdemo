"""Paper-quality figures with consistent style.

Generates:
  * fig_recon_compare_<run>.png  : 3-row grid Original / Recon / |error|
  * fig_codebook_usage.png       : usage curves across runs (paper style)
  * fig_val_psnr.png             : val PSNR curves
  * fig_train_dynamics.png       : 2x2 panel [loss, PSNR, usage, perplexity]
"""
from __future__ import annotations

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
from matplotlib.gridspec import GridSpec
import torch
import torchvision
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, "/home/ubuntu/vq-rotation")
from models.vqae import build_vqae

# Paper style (serif, small)
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS = Path("/home/ubuntu/vq-rotation/results")
OUT = RESULTS / "figures_paper"
OUT.mkdir(parents=True, exist_ok=True)

# Distinct colours + labels for each run type
RUN_STYLE = {
    # Main E1 runs (12000 steps)
    "cifar10_vanilla_e1":  {"label": "VQ-VAE (STE)",        "color": "#1f77b4", "ls": "-"},
    "cifar10_rotation_e1": {"label": "RotationVQ (full)",   "color": "#d62728", "ls": "-"},
    "cifar10_fsq_e1":      {"label": "FSQ",                  "color": "#2ca02c", "ls": "-"},
    "cifar10_gumbel_e1":   {"label": "Gumbel-Softmax VQ",    "color": "#ff7f0e", "ls": "-"},
    # E4 ablation modes (2000 steps, dashed)
    "cifar10_rotation_ste_e4":         {"label": "E4 STE",            "color": "#1f77b4", "ls": "--"},
    "cifar10_rotation_no_rotation_e4": {"label": "E4 Rescale only",   "color": "#9467bd", "ls": "--"},
    "cifar10_rotation_no_rescale_e4":  {"label": "E4 Rotation only",  "color": "#d62728", "ls": "--"},
    "cifar10_rotation_full_e4":        {"label": "E4 Full",           "color": "#8c564b", "ls": "--"},
    # Legacy smoke / benchmark runs (preserved for history; dotted)
    "smoke_rotation":      {"label": "smoke",               "color": "#bbbbbb", "ls": ":"},
}


def load_runs(names: List[str]) -> Dict[str, Dict[str, tuple]]:
    out = {}
    for name in names:
        tb = RESULTS / name / "tb"
        if not tb.exists(): continue
        steps = defaultdict(list); values = defaultdict(list)
        for ev in sorted(glob.glob(str(tb / "events.out.tfevents.*"))):
            try:
                acc = EventAccumulator(ev); acc.Reload()
                for tag in acc.Tags()["scalars"]:
                    for e in acc.Scalars(tag):
                        steps[tag].append(e.step); values[tag].append(e.value)
            except Exception as e:
                print(f"[warn] {ev}: {e}")
        merged = {}
        for tag in steps:
            idx = np.argsort(steps[tag])
            merged[tag] = (np.array(steps[tag])[idx], np.array(values[tag])[idx])
        out[name] = merged
    return out


def plot_panel_curves(runs, outfile="fig_train_dynamics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    spec = [
        ("train/loss",       "train loss",          True,  axes[0, 0]),
        ("train/psnr",       "train PSNR (dB)",     False, axes[0, 1]),
        ("train/usage",      "codebook usage",      False, axes[1, 0]),
        ("train/perplexity", "codebook perplexity", False, axes[1, 1]),
    ]
    for tag, ylabel, logy, ax in spec:
        plotted = 0
        for name, data in runs.items():
            if tag not in data: continue
            s, v = data[tag]
            # skip tiny runs (<3 points) in summary plot to keep it clean
            if len(s) < 3: continue
            style = RUN_STYLE.get(name, {"label": name, "color": "gray", "ls": "-"})
            ax.plot(s, v, label=style["label"], color=style["color"],
                    linestyle=style["ls"], linewidth=1.4, alpha=0.9)
            plotted += 1
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        if logy: ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if plotted > 0:
            ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / outfile, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[plot] {OUT / outfile}")


def plot_single_curve(runs, tag, ylabel, outfile, ylog=False):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for name, data in runs.items():
        if tag not in data: continue
        s, v = data[tag]
        if len(s) < 3: continue
        style = RUN_STYLE.get(name, {"label": name, "color": "gray", "ls": "-"})
        ax.plot(s, v, label=style["label"], color=style["color"],
                linestyle=style["ls"], linewidth=1.6, alpha=0.92)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    if ylog: ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] {OUT / outfile}")


def recon_compare(ckpt_path: Path, label: str, outfile: str, n: int = 8):
    """3-row: Original / Recon / amplified |error|"""
    if not ckpt_path.exists():
        print(f"[skip] {ckpt_path}"); return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_vqae(cfg["model"])
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    val = torchvision.datasets.CIFAR10(
        root=os.path.expanduser(cfg["data"]["root"]), train=False, download=False,
    )
    np.random.seed(0)
    idx = np.random.choice(len(val), size=n, replace=False)
    arr = torch.from_numpy(val.data[idx]).permute(0, 3, 1, 2).float() / 255.0
    arr = arr.to(device)

    with torch.no_grad():
        out = model(arr)
        x_hat = out.x_hat.clamp(0, 1).cpu()
        orig = arr.cpu()

    err = (orig - x_hat).abs()
    err_amp = (err * 5.0).clamp(0, 1)  # 5x amplification for visibility
    psnr = -10.0 * torch.log10(((orig - x_hat) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10))

    fig, axes = plt.subplots(3, n, figsize=(n * 1.3, 3.9))
    for i in range(n):
        axes[0, i].imshow(orig[i].permute(1, 2, 0).numpy()); axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].permute(1, 2, 0).numpy()); axes[1, i].axis("off")
        axes[2, i].imshow(err_amp[i].permute(1, 2, 0).numpy()); axes[2, i].axis("off")
        axes[1, i].set_title(f"{psnr[i].item():.1f} dB", fontsize=8, pad=2)
    for r, lbl in enumerate(("Original", "Reconstructed", "|error| (x5)")):
        axes[r, 0].axis("on")
        axes[r, 0].set_yticks([]); axes[r, 0].set_xticks([])
        for s in ("top", "right", "bottom", "left"): axes[r, 0].spines[s].set_visible(False)
        axes[r, 0].set_ylabel(lbl, fontsize=9)
    fig.suptitle(label, fontsize=10, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUT / outfile, dpi=200, bbox_inches="tight")
    plt.close()
    mean_psnr = psnr.mean().item()
    print(f"[recon] {OUT / outfile}  mean PSNR {mean_psnr:.2f} dB")


def main():
    names = [
        "cifar10_vanilla_e1",
        "cifar10_rotation_e1",
        "cifar10_fsq_e1",
        "cifar10_gumbel_e1",
        "cifar10_rotation_ste_e4",
        "cifar10_rotation_no_rotation_e4",
        "cifar10_rotation_no_rescale_e4",
        "cifar10_rotation_full_e4",
    ]
    runs = load_runs(names)
    print(f"loaded {len(runs)} runs")

    plot_panel_curves(runs)
    plot_single_curve(runs, "train/usage",       "codebook usage",    "fig_codebook_usage.png")
    plot_single_curve(runs, "val/psnr",          "val PSNR (dB)",     "fig_val_psnr.png")
    plot_single_curve(runs, "train/psnr",        "train PSNR (dB)",   "fig_train_psnr.png")
    plot_single_curve(runs, "train/perplexity",  "codebook perplexity (log)", "fig_perplexity.png", ylog=True)

    # Reconstruction case studies from available checkpoints
    recon_compare(RESULTS / "cifar10_vanilla_e1" / "ckpt_final.pt",
                  "VQ-VAE (STE), 12k steps CIFAR-10 (val PSNR 25.84)",
                  "fig_recon_vanilla_e1.png")
    recon_compare(RESULTS / "cifar10_rotation_e1" / "ckpt_final.pt",
                  "RotationVQ (full), 12k steps CIFAR-10 (val PSNR 21.39, collapsed)",
                  "fig_recon_rotation_e1.png")
    recon_compare(RESULTS / "cifar10_fsq_e1" / "ckpt_final.pt",
                  "FSQ, 12k steps CIFAR-10 (val PSNR 27.91)",
                  "fig_recon_fsq_e1.png")
    recon_compare(RESULTS / "smoke_rotation" / "ckpt_final.pt",
                  "RotationVQ, 500 steps CIFAR-10", "fig_recon_smoke_rotation.png")
    for bs in (128, 512):
        p = RESULTS / f"bench_bs{bs}" / "ckpt_final.pt"
        recon_compare(p, f"RotationVQ, 200 steps, batch {bs}", f"fig_recon_bench_bs{bs}.png")

    print("\n=== paper figures ===")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  {os.path.getsize(p)/1024:.1f} KB")


if __name__ == "__main__":
    main()
