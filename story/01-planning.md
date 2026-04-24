# 01 · Planning

*Before any line of code was written.*

## The question

The research brief ([`00-research-brief.md`](00-research-brief.md))
framed a concrete technical question: can we replace the
straight-through estimator (STE) in VQ-VAEs with a rotation-plus-
rescaling transformation that, on the forward pass, reproduces the
hard nearest-neighbour output bit-exactly, but on the backward pass
routes gradients along the angular relationship between the encoder
output `z_e` and its assigned codebook vector `q`?

Two upfront decisions shaped everything that followed.

### Decision 1: venue and quality bar

The brief targeted a NeurIPS main-track submission. That forced the
paper into a specific shape:

- **IMRAD with structured abstract** (motivation → gap → method →
  quantitative result → significance, single paragraph).
- **Thematic related-work**, not a list of papers. Each paragraph
  groups methods by approach (codebook-collapse mitigation, STE
  variants, continuous relaxations, rotation tricks).
- **Contribution list as the only bulleted section**, everything
  else as flowing dense prose with ≥ 4-sentence paragraphs.
- **Every factual claim cited**, every citation sourced from a live
  paper-search result — no training-data recall, no fabrication.

### Decision 2: baseline selection

The most delicate question was how to compare against
[Fifty et al.'s "Rotation Trick"](https://arxiv.org/abs/2410.06424),
which is concurrent work that also proposes a rotation-based
gradient. We chose not to claim novelty in rotation-routing itself,
but instead to differentiate on three axes:

1. A **closed-form Householder reflection**, computed in O(d) per
   token, with no trainable parameters and no materialised matrix.
2. **Explicit ablation of the rotation R and the scalar rescale s**
   as independent components, which prior work does not isolate.
3. **Numerical-stability analysis** in the regime where ẑ_e ≈ q̂ and
   the reflection normal u has vanishing norm, with an explicit
   stop-gradient decomposition that preserves forward bit-exactness.

This framing is defensible because it is **technical**, not
promotional: reviewers can verify each claim mechanically against
the code.

## Repository layout

Two separate trees from the start:

- **`paper/`** — LaTeX, reviewable by reviewers who never run code.
- **`code/`** — everything that runs, including all four quantizers
  under a shared `BaseQuantizer` interface so that swapping methods
  is a config change.

Keeping these separate meant the paper could be written against
*expected* quantitative placeholders (`??`) while training was in
flight — a crucial parallelisation.

## What we deferred

Three things were consciously postponed until baseline training
succeeded:

- **LPIPS and FID evaluation** — costs an Inception/AlexNet forward
  per image, not needed for the first round of comparison.
- **Large-resolution datasets** (FFHQ 256², ImageNet-128) — the
  CIFAR-10 benchmark exposes codebook collapse reliably and finishes
  in minutes per configuration.
- **Downstream generation** (MaskGIT / LlamaGen with a frozen
  tokenizer) — the brief's E5; reconstruction quality comes first.

These deferrals are tracked in [`story/README.md`](README.md) and the
paper's Discussion section.
