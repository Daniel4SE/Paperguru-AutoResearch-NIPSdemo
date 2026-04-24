# Paper

LaTeX sources and compiled PDF for *Rotation-Aware Gradient
Estimation for Vector-Quantized Autoencoders*.

## Layout

```
paper/
├── main.tex                 ← class, preamble, \input{sections/*}
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex     ← includes Figure 1
│   ├── related_work.tex
│   ├── method.tex           ← Householder derivation + Algorithm 1
│   ├── experimental_setup.tex
│   ├── results.tex          ← tables wired to auto-numbers
│   ├── discussion.tex
│   ├── conclusion.tex
│   └── appendix.tex
├── figures/                 ← curves and reconstructions
├── refs.bib                 ← 35 entries, all from paper_search
├── results_numbers.tex      ← auto-generated; do not edit by hand
├── Makefile                 ← `make all` → refresh numbers + rebuild
└── main.pdf                 ← 15-page compiled output
```

## Compilation

```bash
# Local build (assumes pdflatex + bibtex on PATH)
make pdf

# Refresh numbers from the training server and rebuild
make all
```

Requires TeX Live 2020 or newer. On macOS via Homebrew:

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended collection-latexextra \
                   booktabs microtype cleveref algorithmicx \
                   algorithms natbib subcaption
```

## Auto-refresh pipeline

```
┌──────────────────────┐   1. train
│ training on H100     │ ──────────────────────────────┐
└──────────────────────┘                                │
             │                                          ▼
             │ 2. writes                   results/<run>/tb/events.*
             ▼
┌──────────────────────┐                                │
│ scripts/             │ 3. extracts metrics            │
│ collect_results.py   │ ◀──────────────────────────────┘
└──────────────────────┘
             │
             │ 4. emits
             ▼
┌──────────────────────┐
│ paper_numbers.tex    │  ← \providecommand + \renewcommand pairs
└──────────────────────┘
             │
             │ 5. \input'd by
             ▼
┌──────────────────────┐
│ paper/main.tex       │  6. pdflatex × 2 + bibtex
└──────────────────────┘
             │
             ▼
         main.pdf
```

The paper builds with or without `results_numbers.tex`; when it is
missing, every metric macro falls back to `??`. This lets the PDF
compile during early drafting and transitions smoothly to live
numbers as training progresses.

## Figure index

| Figure | File | Used in |
|---|---|---|
| Fig 1 — Architecture | `figures/fig1_architecture.png` | Introduction |
| Fig 2 — Training dynamics | `figures/fig_train_dynamics.png` | Results (RQ2) |
| Fig 3 — Reconstruction case study | `figures/fig_recon_smoke_rotation.png` | Results |
| Supplementary | `figures/fig_*`, `figures/curve_*` | Appendix |

## Citation style

NeurIPS 2026 uses numbered author-year citations via `natbib`:
`\cite{}` → `[n]`, `\citep{}` → `(Author et al. 2024)`,
`\citet{}` → `Author et al. [n]`. The `plainnat` bibliography style
is used until the official NeurIPS 2026 style file ships.
