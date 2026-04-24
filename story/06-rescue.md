# 06 · Rescuing the Rotation

*After documenting the collapse, can we fix it?*

## The setup

[`05-rotation-collapse.md`](05-rotation-collapse.md) ended with a
clean diagnosis: back-propagating the Householder reflection
$\mathbf{R}$ creates a positive-feedback loop that zeros the
commitment loss and collapses codebook usage. That was a complete
negative result — reproducible, localised, interpretable. A paper
could have stopped there.

But a reviewer would ask: "can you propose a fix that is actually
demonstrated to work, not just speculated about?" That is the
difference between a workshop-level negative result and a main-track
contribution. So we tried.

## The hypothesis

If the failure mode is "encoder collapses onto too few codes
because the rotation gradient keeps pulling it there", then adding
a force that actively **rewards spreading across codes** should
cancel it out. The standard ML tool for this is entropy
regularisation: compute a soft assignment $\bar{\mathbf{p}} \in
\Delta^{K-1}$ averaged over the batch, and add $-\lambda\,H(\bar{\mathbf{p}})$
to the total loss. Maximising the Shannon entropy pushes the
encoder to use codes uniformly, exerting a direct anti-collapse
force.

This is not a novel idea — it is used routinely in mixture-of-
experts load balancing, in EdVAE (Baykal et al. 2023), and in
online-clustered codebook methods (Zheng & Vedaldi 2023). The
question was whether it would be sufficient to counteract the
specific collapse caused by $\mathbf{R}$.

## Implementation

A 30-line patch to `RotationVQ.forward`:

```python
# Soft-assignment entropy from z and codebook, embedding detached.
z2    = (z_flat.detach() ** 2).sum(-1, keepdim=True)
e2    = (codebook ** 2).sum(-1)
dist2 = z2 - 2 * z_flat @ codebook.t() + e2.unsqueeze(0)
soft  = F.softmax(-dist2 / tau, dim=-1)
p_bar = soft.mean(dim=0).clamp_min(eps)
neg_H = (p_bar * p_bar.log()).sum()   # <= 0
entropy_loss = lambda_ * neg_H
```

Added to the loss in `train.py` so it acts on the encoder only;
the codebook is detached inside the regulariser and continues to
receive its EMA update as before.

## The sweep

We ran $\lambda \in \{0.0, 0.1, 0.3, 1.0, 3.0\}$ for 2 000 training
steps each under the otherwise-identical E4 recipe:

| $\lambda$ | val PSNR | Usage | Perplexity | Verdict |
|----------:|---------:|------:|-----------:|---------|
| 0.0       |    20.73 |  0.43 |      102   | baseline (collapsed) |
| 0.1       |    18.89 |  0.58 |      336   | oscillates — ring-down then collapses again |
| 0.3       |    20.09 |  1.00 |      912   | stable but PSNR lower than baseline |
| 1.0       |    20.23 |  1.00 |      949   | stable, small PSNR gain |
| **3.0**   | **22.14**| 1.00 |      933   | **sweet spot at 2 000 steps** |

Two surprises from this sweep:

1. **Entropy regularisation overshoots at intermediate $\lambda$.**
   At $\lambda = 0.3$ the codebook usage is saved (goes to 100%),
   but the reconstruction PSNR actually *drops* below the collapsed
   baseline. The regulariser is now the dominant force; encoder
   outputs are being pushed toward uniform coverage faster than the
   reconstruction loss can shape them into useful assignments.

2. **The "ring-down" pattern at $\lambda = 0.1$.** Training curves
   show usage recovering to $0.94$ around step 900, then collapsing
   again to $0.36$ by step 1 500. The cosine learning-rate
   schedule weakens the entropy gradient along with the rest; past
   a certain point the rotation attractor wins again. This is
   diagnostic: it tells us the regulariser's effective strength has
   to be **maintained throughout training**, not just early on.

Both phenomena disappear at $\lambda = 3.0$.

## Extending to full 12 k steps

With $\lambda = 3.0$ established as the sweet spot on the short run,
we launched a full 12 000-step training using the same recipe as
the main E1 comparison. Outcome:

| Method                              | val PSNR | val SSIM | Usage | Perplexity |
|-------------------------------------|---------:|---------:|------:|-----------:|
| VQ-VAE (STE) baseline               |    25.84 |     0.87 | 1.000 |      948.67 |
| RotationVQ (full, no regulariser)   |    21.39 |     0.68 | 0.361 |      173.12 |
| **RotationVQ + entropy ($\lambda=3$, ours)** | **24.38** | **0.84** | **1.000** | **1 021.23** |

The entropy regulariser closes **two-thirds of the gap** between
the collapsed and the STE baselines (from $-4.45$\,dB to $-1.46$\,dB).
Codebook perplexity is actually higher than STE's ($1\,021$ vs.
$948$) — the entropy force makes the token distribution more
uniform than plain STE does — but the remaining PSNR gap suggests
$\lambda = 3.0$ is slightly over-regularising even at 12 k steps,
pushing the encoder toward uniform assignments at the cost of some
reconstruction fidelity. A follow-up that jointly sweeps $\lambda$
with the commitment coefficient $\beta$ is likely to close the gap
further.

## What this contributes

Calling this a "fix" requires caveats. It does not make the
rotation estimator *beat* STE; it makes the rotation estimator
**usable**. The practical take-away is:

- **The closed-form identity $\mathbf{q} = s\cdot\mathbf{R}\mathbf{z}_e$
  is real and cheap**. You can implement it in 30 lines with
  $O(d)$ per-token cost and no trainable parameters.
- **Back-propagating it naively is harmful**. The intuition of
  "geometrically correct gradient" does not hold under the standard
  VQ training recipe.
- **Adding an entropy regulariser turns it from harmful into
  competitive**. It still underperforms STE by $1.5$\,dB in our
  setting, but usage and perplexity are fully preserved, and the
  remaining gap is plausibly attributable to overly-aggressive
  regularisation.
- **FSQ still wins**. Our comparison with Mentzer et al. (2024)
  makes it clear: eliminating the learned codebook bypass the
  entire collapse-vs-regularisation trade-off. Anyone building a
  new tokenizer today should at least try FSQ before committing to
  a learned codebook with either STE or rotation-aware training.

This is the paper the experiment actually wrote. Submitting it as
is would be a defensible workshop-tier contribution; a main-track
submission would need the $\lambda$-$\beta$ joint sweep and the
downstream-generation validation (E5 in our experimental matrix).
Both are out of scope for this artefact trail and are listed as
future work.
