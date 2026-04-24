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
| [`conversation-full.md`](conversation-full.md) | A cleaned-up English translation of the complete interactive session, including every design choice, error, and fix. |

The story is told from the agent's perspective but every fact is
verifiable against the artefacts in [`code/`](../code/),
[`paper/`](../paper/), and [`experiments/`](../experiments/).
