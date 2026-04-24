# 04 · Paper Writing

*Writing a 15-page NeurIPS-style paper while training runs on the
server.*

## Skeleton first

Before drafting any prose we created the full section skeleton so
that every cross-reference existed from day one:

```
paper/
├── main.tex                  ← class, preamble, \input{…} skeleton
├── refs.bib                  ← 35 BibTeX entries
├── results_numbers.tex       ← auto-generated (empty at first)
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── method.tex
│   ├── experimental_setup.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── conclusion.tex
│   └── appendix.tex
└── Makefile                  ← `make all` refreshes numbers + rebuilds
```

Every section is its own file, short enough that a reviewer can read
and edit it in a single sitting without scrolling past the rest of
the document. `main.tex` contains nothing but the preamble,
`\input{}` lines, and the `\begin{document}…\end{document}` pair.

## Method section: derivation before implementation

The Method section was written **before** the quantizer was
implemented in earnest. Concretely, we first derived, on paper, the
identity

```
 s·R·z_e = q
```

for `s = ‖q‖ / ‖z_e‖` and `R = I − 2 v vᵀ` the Householder
reflection with `v = (ẑ_e − q̂) / ‖ẑ_e − q̂‖`. We then verified by
direct substitution that `R·ẑ_e = q̂`, and only then wrote the
PyTorch operator and the stop-gradient wrapper that makes the
forward value bit-exactly `q` while leaving the backward Jacobian
equal to `s·R`.

This order matters. If we had started from code, the stop-gradient
placement would have been trial-and-error; starting from the maths
turned it into a mechanical translation.

## Figure 1, generated

The architecture figure was produced via the agent's
`image_generate` tool. The prompt contained the exact mathematical
content of the middle panel (codebook, nearest-neighbour arrow,
unit-sphere inset with Householder reflection, boxed equation
`q̃ = sg[sR]·z_e + sg[q − sg[sR]·z_e]`) so that the generated image
is a faithful schematic of the paper's content rather than
decoration. The final PNG lives at
[`../paper/figures/fig1_architecture.png`](../paper/figures/fig1_architecture.png).

## The auto-refresh pipeline

The most important pattern in this workflow is the link between
training logs and paper numbers. It works like this:

1. **The training script writes TensorBoard events** at each
   log/validation step.
2. **`scripts/collect_results.py`** scans
   `results/cifar10_*_e1/tb/` for each quantizer, extracts the
   final-step value of every tracked metric, and emits a
   `paper_numbers.tex` file of the form
   ```tex
   \providecommand{\vanillaPsnr}{??}\renewcommand{\vanillaPsnr}{25.84}
   \providecommand{\rotationPsnr}{??}\renewcommand{\rotationPsnr}{26.12}
   ...
   ```
   Provide-then-renew means the file is safe to `\input` even when
   the paper has not defined a fallback, and it overrides any
   placeholder when numbers are available.
3. **`paper/main.tex`** calls
   `\IfFileExists{results_numbers.tex}{\input{...}}{}`
   so the build works whether numbers are present or not.
4. **`paper/Makefile`** wires the two together:
   ```make
   make all   # ssh → collect_results.py → rsync → pdflatex × 2
   ```

The net effect is that we can run `make all` at any time during
training and get a PDF that reflects the latest numbers, without
editing a single LaTeX character by hand. Table 1 was filled in
this way for the Vanilla baseline the moment its training completed.

## Lessons

- **Dense paragraphs, no inline numbered fragments.** The only
  itemised list in the body is the contribution list in the
  introduction. Everything else is 4+-sentence prose.
- **Every citation came from a live `paper_search` result.** Before
  writing any paragraph that cites X, we ran
  `paper_search(X-related query)` and copied the BibTeX metadata
  verbatim. This is the only reliable defence against hallucinated
  citations, which is a silent failure mode of all LLM-assisted
  writing.
- **Two passes of `pdflatex` + one of `bibtex` + two more
  `pdflatex`** is the only reliable way to resolve cross-references
  and citations. The `Makefile` encodes this.
- **LSP "undefined reference" warnings inside `\input`-ed files are
  not errors.** The LSP analyses each `.tex` file independently and
  cannot see the parent; only the final `pdflatex` output is
  diagnostic.
