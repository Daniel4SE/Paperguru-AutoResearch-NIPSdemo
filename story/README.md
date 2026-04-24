# The Research Journey

This directory is a post-hoc reconstruction of the entire AI-driven
research session that produced the contents of this repository. It is
written in **English prose**, chronologically, so that someone
reviewing the repository can follow every major decision without
reading the raw conversation transcript.

## Contents

| File | Covers |
|---|---|
| [`00-research-brief.md`](00-research-brief.md) | The original research question — rotation-aware gradient estimation for VQ-VAEs — copied verbatim as it was supplied to the agent. |
| [`01-planning.md`](01-planning.md) | Decisions made before any code was written: venue targeting, baseline selection, directory structure, LaTeX skeleton. |
| [`02-gpu-saga.md`](02-gpu-saga.md) | The dead-H100 drama: `cuInit=802`, `GPU requires reset`, `CUDA_ERROR_SYSTEM_NOT_READY`, fabricmanager mismatches, GSP firmware timeouts, and the VM migration that ultimately unblocked training. |
| [`03-training.md`](03-training.md) | How the baseline was trained: the 19 %-GPU-util problem, the switch to batch 1024 with TF32, the watcher script that auto-chains configurations. |
| [`04-paper-writing.md`](04-paper-writing.md) | LaTeX skeleton, section drafting, Figure 1 generation, and the auto-refreshing results pipeline that links TensorBoard events to paper macros. |
| [`05-rotation-collapse.md`](05-rotation-collapse.md) | **The result we did not want but must report:** RotationVQ's full-Jacobian implementation collapses after ~1 000 steps. Step-by-step diagnostic and the ablation strategy to localise the failure mode. |
| [`06-rescue.md`](06-rescue.md) | **Can we fix it?** Yes, partially: an entropy regulariser on top of the $s\cdot\mathbf{R}$ Jacobian restores 100 % codebook utilisation and recovers +2.99 dB val PSNR. A $\lambda$ sweep finds the sweet spot at $\lambda=3.0$. |
| [`conversation-full.md`](conversation-full.md) | A cleaned-up English translation of the complete interactive session, including every design choice, error, and fix. |

The story is told from the agent's perspective but every fact is
verifiable against the artefacts in [`code/`](../code/),
[`paper/`](../paper/), and [`experiments/`](../experiments/).
