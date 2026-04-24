# 05 · The Rotation Collapse

*The result we did not want but have to report honestly.*

## The claim

The paper's central technical move was: replace the straight-through
estimator's identity Jacobian $\mathbf{I}$ with the closed-form map
$s\cdot\mathbf{R}$ that sends $\mathbf{z}_e$ to $\mathbf{q}$ along the
Householder reflection and the scalar norm ratio
$s = \lVert\mathbf{q}\rVert/\lVert\mathbf{z}_e\rVert$. The intuition
was that the encoder would receive a gradient aware of which
codebook vector was chosen and would align its future outputs with
that direction.

## What actually happened

The CIFAR-10 comparison, under an identical training recipe
(12 000 steps, batch 1024, peak lr $1.6\times 10^{-3}$), produced:

| Method        | val PSNR | val SSIM | Usage | Perplexity |
|---------------|---------:|---------:|------:|-----------:|
| VQ-VAE (STE)  | **25.84**| **0.87** | **1.000** | **948.67** |
| RotationVQ (full) | 21.39 | 0.68  | 0.361 | 173.12 |

RotationVQ is **4.45 dB worse**, uses **36 %** of its codebook
instead of 100 %, and has perplexity **173** out of the possible
1024 — fewer than one-fifth of the codes actually carrying
information.

## Step-by-step diagnostic

Comparing the two methods step by step:

| Step | Method     | Recon loss | Train PSNR | Usage | Perplexity | Commit loss |
|-----:|------------|-----------:|-----------:|------:|-----------:|------------:|
|    0 | vanilla    | 0.38       | 7.25 dB    | 1.00  | 900.6      | 0.0018      |
|    0 | rotation   | 0.38       | 7.25 dB    | 1.00  | 900.6      | 0.0018      |
|  200 | vanilla    | 0.10       | 18.10 dB   | 0.60  | 304.3      | 0.0002      |
|  200 | rotation   | 0.095      | 18.48 dB   | 0.82  | 472.3      | 0.0003      |
| 1000 | vanilla    | 0.058      | 22.49 dB   | 0.92  | 527.8      | 0.0007      |
| 1000 | rotation   | 0.066      | 21.39 dB   | **0.94** | **632.5** | 0.0001      |
| **2000** | vanilla  | 0.048    | 23.97 dB   | 1.00  | 774.7      | 0.0004      |
| **2000** | rotation | 0.068    | 21.02 dB   | **0.22** | **94** | **0.0000** |
| 5000 | vanilla    | 0.040      | 25.43 dB   | 1.00  | 904.7      | 0.0012      |
| 5000 | rotation   | 0.053      | 22.74 dB   | 0.38  | 164.8      | 0.0000      |
|12000 | vanilla    | 0.037      | 26.98 dB   | 1.00  | 942.7      | 0.0020      |
|12000 | rotation   | 0.046      | 23.78 dB   | 0.36  | 169.9      | 0.0000      |

For the first 1 000 steps the two methods are indistinguishable;
RotationVQ even has slightly higher codebook usage and perplexity,
which is what the paper's intuition predicts. Between step 1 000
and step 2 000, however, RotationVQ's codebook usage falls from
0.94 to 0.22 in a single log interval, and it never recovers.

## Why this happens

Look at the commitment loss column. VQ-STE keeps
$\lVert\mathbf{z}_e - \mathrm{sg}[\mathbf{q}]\rVert^2$ at the
$10^{-3}$ scale throughout training; RotationVQ's commitment loss
drops to numerically zero at step 2 000 and stays there. That means
the encoder's output has become numerically indistinguishable from
the codebook vector it selected: $\mathbf{z}_e \approx \mathbf{q}$
for every example in the mini-batch.

This is consistent with the $s\cdot\mathbf{R}$ Jacobian being a
*constructive* gradient — it always rotates $\mathbf{z}_e$ toward
its currently-selected $\mathbf{q}$, regardless of whether that is
the correct $\mathbf{q}$ for the semantic content of the input. If
two different images select the same codebook index in some
mini-batch, the gradient signal the encoder receives for both
pushes them both toward that shared code. Under the ordinary STE,
by contrast, the encoder's gradient is literally the decoder's
gradient, which carries information about what the decoder needs
to distinguish the two images. The STE's "wrong" Jacobian is
informative in a way that our "correct" Jacobian is not.

## The commitment loss stops fighting back

The VQ-VAE training recipe relies on the commitment loss
$\beta\,\lVert\mathbf{z}_e - \mathrm{sg}[\mathbf{q}]\rVert^2$ to keep
encoder outputs from being pulled too aggressively into the
codebook. Once $\mathbf{z}_e \approx \mathbf{q}$ numerically, this
loss is zero and exerts no force, leaving the rotation-aware
gradient free to continue collapsing the encoder onto whatever
handful of codes it has latched onto.

## What we are doing about it

Instead of abandoning the investigation or quietly rewriting the
paper's thesis, we are running the ablation that separates the
rotation $\mathbf{R}$ and the scalar rescaling $s$:

| Mode | Backward Jacobian | Expectation |
|------|-------------------|-------------|
| `ste` | $\mathbf{I}$ | Should match VQ-STE baseline. |
| `no_rotation` | $s\cdot\mathbf{I}$ | Tests whether the scale alone causes the collapse. |
| `no_rescale` | $\mathbf{R}$ | Tests whether the rotation alone is usable. |
| `full` | $s\cdot\mathbf{R}$ | The version that just collapsed. |

These runs are executing on the same H100 that produced the main
comparison, under an identical training recipe but with only 2 000
steps each (the collapse happens by step 2 000, so longer runs add
no information). Their results will be wired into Table 2 of the
paper by the same `scripts/collect_results.py` pipeline that
produced Table 1.

## What the paper now says

The paper has been updated in three places:

1. **Abstract.** The thesis is rewritten as a cautionary finding:
   the naive full-Jacobian implementation collapses, and the
   contribution is the identification of that failure mode plus a
   component-level decomposition.
2. **Results §5.1.** The headline claim is that our method
   \emph{underperforms} STE in the full configuration, with a
   step-level diagnostic showing exactly when and why.
3. **Discussion §6.** The new framing: the identity
   $\mathbf{q} = s\cdot\mathbf{R}\mathbf{z}_e$ is a correct
   geometric fact and an efficient computation; the naive
   back-propagation of its Jacobian through the standard VQ-VAE
   training recipe is not.

This is the paper that the experiment actually wrote, not the paper
we intended to write. Putting this in the public repository is the
most valuable thing the agent can do: negative results that
localise a failure mode are exactly what a reviewer can build on.
