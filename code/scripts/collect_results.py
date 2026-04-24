"""Extract final metrics from TensorBoard events for all E1 runs,
compute wall-clock throughput, and emit a LaTeX file of \\newcommand
definitions that results.tex can reference.

Writes ~/vq-rotation/results/paper_numbers.tex on the server.
"""

import glob, os, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS = Path(os.path.expanduser("~/vq-rotation/results"))

RUNS = {
    "vanilla":   "cifar10_vanilla_e1",
    "rotation":  "cifar10_rotation_e1",
    "fsq":       "cifar10_fsq_e1",
    "gumbel":    "cifar10_gumbel_e1",
    # E4 ablation modes (2000-step runs)
    "rotste":    "cifar10_rotation_ste_e4",
    "rotnorot":  "cifar10_rotation_no_rotation_e4",
    "rotnoresc": "cifar10_rotation_no_rescale_e4",
    "rotfull":   "cifar10_rotation_full_e4",
    # Entropy-regularised fix (lambda sweep + full 12k run)
    "rotent":    "cifar10_rotation_entropy_e1",
    "rotlamA":   "cifar10_rotation_fix_lam0.3_e4",
    "rotlamB":   "cifar10_rotation_fix_lam1.0_e4",
    "rotlamC":   "cifar10_rotation_fix_lam3.0_e4",
}


def load_run(name: str) -> dict:
    path = RESULTS / name / "tb"
    if not path.exists():
        return {}
    data = defaultdict(list)
    wall_times = defaultdict(list)
    for ev in sorted(glob.glob(str(path / "events.out.tfevents.*"))):
        try:
            acc = EventAccumulator(ev)
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            for e in acc.Scalars(tag):
                data[tag].append((e.step, e.value))
                wall_times[tag].append(e.wall_time)
    merged = {}
    for tag, pairs in data.items():
        pairs.sort()
        steps, vals = zip(*pairs)
        merged[tag] = (np.array(steps), np.array(vals))
    if "train/loss" in wall_times and len(wall_times["train/loss"]) > 1:
        wt = sorted(wall_times["train/loss"])
        merged["_wall"] = wt[-1] - wt[0]
        merged["_n_steps"] = (
            merged["train/loss"][0][-1] - merged["train/loss"][0][0] + 1
        )
    return merged


def summary(run_data: dict) -> dict:
    out = {}
    for tag, label in [
        ("val/psnr", "psnr"),
        ("val/ssim", "ssim"),
        ("val/lpips", "lpips"),
        ("val/usage", "usage"),
        ("val/perplexity", "perplexity"),
        ("train/psnr", "trainpsnr"),
        ("train/usage", "trainusage"),
        ("train/perplexity", "trainperplexity"),
    ]:
        if tag in run_data:
            steps, vals = run_data[tag]
            out[label] = float(vals[-1])
    if "_wall" in run_data and "_n_steps" in run_data:
        wall = run_data["_wall"]
        nsteps = run_data["_n_steps"]
        out["wallsec"] = wall
        out["stepspersec"] = nsteps / wall if wall > 0 else 0.0
    return out


def fmt(v, d=2):
    if v is None:
        return "??"
    try:
        return f"{v:.{d}f}"
    except Exception:
        return str(v)


# Every macro that results.tex / main.tex might reference.
# Always emit a provide+renew pair so the command is guaranteed to
# exist regardless of whether main.tex already pre-defined a fallback.
MACROS = {
    "psnr": ("Psnr", 2),
    "ssim": ("Ssim", 2),
    "lpips": ("Lpips", 3),
    "usage": ("Usage", 3),
    "perplexity": ("Perplexity", 2),
    "trainpsnr": ("Trainpsnr", 2),
    "trainusage": ("Trainusage", 3),
    "trainperplexity": ("Trainperplexity", 2),
    "stepspersec": ("Stepspersec", 2),
    "wallsec": ("Wallsec", 1),
}


def main():
    all_summaries = {}
    for label, rundir in RUNS.items():
        rd = load_run(rundir)
        s = summary(rd) if rd else {}
        all_summaries[label] = s
        print(f"[{label}] {s}")

    (RESULTS / "paper_numbers.json").write_text(json.dumps(all_summaries, indent=2))

    lines = [
        "% Auto-generated from TensorBoard events by scripts/collect_results.py",
        "% Safe to \\input repeatedly; each macro is provided-then-renewed.",
        "",
    ]
    for method, s in all_summaries.items():
        lines.append(f"% ---- {method} ----")
        if not s:
            lines.append(f"% {method}: no data yet")
            lines.append("")
            continue
        for key, (suffix, digits) in MACROS.items():
            cmd = f"\\{method}{suffix}"
            if key in s:
                val = fmt(s[key], digits)
                lines.append(
                    f"\\providecommand{{{cmd}}}{{??}}\\renewcommand{{{cmd}}}{{{val}}}"
                )
        # Derived: samples/s = steps/s * batch (1024)
        if "stepspersec" in s:
            samples = int(s["stepspersec"] * 1024)
            cmd = f"\\{method}Samplespersec"
            lines.append(
                f"\\providecommand{{{cmd}}}{{??}}\\renewcommand{{{cmd}}}{{{samples}}}"
            )
        lines.append("")

    tex_out = RESULTS / "paper_numbers.tex"
    tex_out.write_text("\n".join(lines))
    print(f"\nWrote {tex_out}")
    print(tex_out.read_text())


if __name__ == "__main__":
    main()
