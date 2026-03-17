# Paper Assets

This directory contains the publishable manuscript package for the Phase 1
screening study.

Contents:

- `data/phase1_screening_results.tsv`: frozen copy of the final screening data
- `figures/`: generated figure outputs for the manuscript
- `scripts/make_figures.py`: reproduces the figures from the frozen TSV
- `tables/phase1_results.tex`: LaTeX table used by the manuscript
- `references.bib`: bibliography for the draft
- `main.tex`: manuscript draft
- `Makefile`: figure generation and LaTeX build helpers

Recommended workflow:

```bash
python3 paper/scripts/make_figures.py
make -C paper figures
make -C paper pdf
```

If `latexmk` is unavailable, the figure step still works and `main.tex` remains
ready for Overleaf or a local TeX toolchain.
