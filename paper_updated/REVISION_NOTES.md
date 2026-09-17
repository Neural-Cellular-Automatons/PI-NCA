# paper_updated — the ICLR manuscript

**PI-NCA: Neural Cellular Automata that Conserve by Construction.** Two proposed
architectures, PI-NCA and MC-PI-NCA; the bound projection is a parameter-free
configuration of PI-NCA ("bounded PI-NCA"), not a third architecture.

## Files

| File | What it is |
|---|---|
| `main.tex` | Manuscript (ICLR 2027 style) |
| `appendix.tex` | Proofs, equations, architecture details, additional tables |
| `references.tex` | Inline `thebibliography` (no BibTeX step) |
| `iclr2027_conference.sty` | Conference style |
| `pinca-figure.tex`, `intervention-figure.tex`, `architecture-diagrams.tex` | TikZ / pgfplots figures |
| `table-*.tex` | Tables, each `\input` exactly once |
| `pinca_tests.py` | Recomputes every ranking and paired test the tables quote, from `results/` |
| `check_manuscript.py` | Structural checks standing in for a LaTeX build |

```bash
PYTHONPATH=src python paper_updated/pinca_tests.py   # the numbers behind the tables
python paper_updated/check_manuscript.py            # refs, citations, table widths, duplicates
cd paper_updated && pdflatex main && pdflatex main  # build (two passes)
```

## Scope decisions to keep in mind

- Ranks are among the **eleven reported models**. The same GPU run also trained
  exploratory conservative variants (dilated-perception and spectral hybrids) that are
  outside this paper; the setup section says so. With them included, bounded PI-NCA would
  be second rather than first on Cahn--Hilliard, and the released results files contain
  them, so that sentence must stay.
- Every number comes from the 4090 run (grid 48², five seeds; ten on heat, Cahn--Hilliard
  and Navier--Stokes). The paper never uses reduced-scale CPU results.
- The wording avoids "state-of-the-art" (the solvers are our own); `check_manuscript.py`
  enforces this.

## Open before submission

- Build the PDF once with a real LaTeX toolchain (not available where this was written).
- Rerun the three evaluation axes that failed at GPU scale (out-of-distribution,
  guard-disabled stability, resolution transfer); the limitations section currently
  states they are not measured.
- Author review of the scientific interpretation, the related-work positioning against
  FINN, and the bibliography.
