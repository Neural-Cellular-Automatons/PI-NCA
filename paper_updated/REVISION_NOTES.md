# ICLR manuscript revision — PI-NCA as the contribution

## This revision (framing)

The paper is now a method paper. PI-NCA (a neural cellular automaton whose shared rule
predicts fluxes, so every field is conserved exactly for any weights) is the main
contribution; MC-PI-NCA is its per-field extension to coupled systems; the
headroom projection is the bound-preserving component. The benchmark is the evidence
for them rather than the subject.

No measurement changed. What changed is which comparisons are foregrounded, and two
pairs of tests the run did not tabulate were added, recomputed from the saved
per-initial-condition errors by `pinca_tests.py` with the run's own paired test and Holm
correction:

- `table-pinca.tex` — each PI-NCA model against the strongest baseline outside its
  family (10 tests, all Holm-significant).
- `table-vsnca.tex` — PI-NCA against the standard NCA on all ten phenomena (better on 7,
  tie on 1, worse on the 2 non-conservative reaction systems).

Other changes:

- Three propositions (exact conservation, per-field conservation, bounded mass-exact
  projection) stated in Section 3 with proofs in a new appendix section.
- New main-text figure `pinca-figure.tex`; model display names (PI-NCA, MC-PI-NCA,
  PI-NCA-B, PI-NCA-MS, PI-NCA-MS-B, PI-NCA-Spec) used consistently in every table, with
  the mapping to code names in the appendix.
- `table-regime2d` gains a "best PI-NCA (rank)" column.
- Fixed: the previous draft `\input` several tables from both `main.tex` and
  `appendix.tex`, printing them twice with clashing labels. Each table is now input once,
  and `check_manuscript.py` fails on duplicate inputs or labels.
- Clarified a point a reviewer would raise: the unbounded model's mass drift on
  Cahn--Hilliard comes from the evaluation's divergence guard clamping the state, not from
  the flux update.
- Internal history of earlier drafts (withdrawn reduced-scale claims) removed from the
  manuscript; it remains below and in the repository.
- The limitations still state the three unmeasured axes, the teacher-error ceiling on heat
  and advection--diffusion, the regimes where PI-NCA is the wrong prior, and that the 3-D
  suite does not favour it.
- Wording avoids "state-of-the-art": the solvers are our own, so the defensible claim is
  "best of fourteen architectures in this benchmark", which the paper states with tests.

---

## Previous revision — rewritten against the GPU run

## What changed and why

The previous revision was written against reduced-scale CPU results, with a headline
protocol at a single seed on a $24^2$ grid and separately-labelled multi-seed studies that
could not be pooled. Those constraints are gone. Every number in this manuscript now comes
from **one unattended 22.2-hour run on a single NVIDIA RTX 4090**: grid $48^2$, 1{,}200
epochs, five seeds for the full ten-phenomenon matrix and ten seeds for the three regime
representatives, preceded by a 205-test correctness gate.

That changed several conclusions, and the manuscript states each reversal rather than
quietly replacing the numbers.

### Conclusions that reversed

1. **Heat.** The previous draft reported a multi-scale conservative NCA as both more
   accurate and $107\times$ smaller than the FNO. At $48^2$ with ten seeds the FNO leads
   ($0.001447$ against $0.003699$), and the difference is Holm-significant at $n=80$. The
   scaling study identifies the cause directly: the winner changes between grid $24^2$ and
   $48^2$ (Kendall's $\tau=0.73$). **The accuracy half of that claim is withdrawn in the
   text**; the parameter claim stands and is restated in resource-relative form.

2. **Cahn--Hilliard.** Previously all emulators were reported as failing (errors $14$–$18$
   against an identity reference near $0.93$). With a converging teacher and the full
   protocol, the bounded conservative hybrid leads at $0.357$ and the FNO is twelfth of
   fourteen at $0.650$. The old result came from a teacher at $\Delta t=0.5$, above the
   fourth-order explicit bound of $0.231$, held together by the stepper's clip and showing
   an observed order of accuracy of zero. Experiments now use $\Delta t=0.02$; this is
   documented in the appendix.

3. **Bounding (A1).** Previously reported as a large stability fix with no accuracy
   benefit at the corrected teacher. It now buys both: CH error $0.515\to0.358$ and mass
   drift $19.9\to5.2\times10^{-5}$ at an identical backbone.

4. **The bounded-projection violation.** A previous claim that the uniform mass projection
   re-violates the bound on "a few percent of cells" is **withdrawn** — it was measured
   against the non-converging teacher. The failure remains provable and is exhibited by a
   constructed state in the test suite, but its empirical magnitude at this scale is not
   established. A7 now reports what is actually measured: the three projection variants are
   indistinguishable in accuracy and differ by five orders of magnitude in mass drift.

5. **A5 (spatial reach).** At $24^2$ the smallest stencil won on heat; at $48^2$ the widest
   does. Both are reported, because the disagreement is the point.

### Findings the physics-free controls produced

Adding ResNet and U-Net baselines at full and parameter-matched size changed the reading of
three phenomena: the U-Net wins wave, Gray--Scott, and Nagumo outright, and on Nagumo the
entire top three are physics-free. Without those controls the same data would have shown a
conservative NCA winning a field of conservative NCAs. The identity floor produced one
further result: on Navier--Stokes the plain NCA is on the wrong side of it in both tables.

### Axes that are *not* measured

Six stages of the run failed and one was skipped. The out-of-distribution study, the
guard-disabled long-horizon stability study, and the resolution-transfer sweep therefore
have no results at this scale. Reduced-scale results exist from earlier CPU runs, but
mixing backends inside one study is the error this paper's controls exist to prevent, so
they are **omitted rather than reported**. `table-restransfer.tex` and
`table-operators.tex` were deleted for this reason; Appendix "Evaluation Axes Not Measured
at This Scale" states the omission explicitly, and the generated claims audit marks them
`NOT_YET_MEASURED`.

## Structural changes

- Tables rewritten from the run: `table-regime2d`, `table-heatms` (now the ten-seed heat
  panel with bootstrap intervals and paired verdicts), `table-ablsummary` (now A4/A5/A1/A7
  with both phenomena per ablation), `table-eff` (grouped by parameter budget rather than
  ranked across budgets), `table-heat20`, `table-regime3d`, `table-axes`.
- New: `table-teacher` (the solver's own error, split spatial/temporal, with a convergence
  verdict per equation), `table-scaling` (Kendall's $\tau$ rank stability), `table-matched`
  (the well-posed PINN comparison, with the solver as a row).
- Removed: `table-restransfer`, `table-operators` (see above).
- `intervention-figure.tex` redrawn with the GPU A4 and A1 measurements, on logarithmic
  axes because the effects now span decades.
- Bibliography: added FINN (arXiv:2104.06010) and Karlbauer et~al.\ (arXiv:2111.11798),
  which are the direct prior art for conservative flux-form neural PDE models and are now
  cited as such — the manuscript explicitly does not claim that idea as new. The closest
  prior work overall, Saha and Wang (arXiv:2608.30328), is cited in the first related-work
  paragraph. Removed the ICLR-points entry, which was uncited and not about PDEs.
- `check_manuscript.py` added: a structural validator standing in for a LaTeX build in an
  environment without TeX. It checks every `\input`, `\ref`, citation key, environment
  balance, and table column count, and fails on five over-claiming phrases. It currently
  reports no problems.

## Format

- `main.tex` is the entry point; build with `latexmk -pdf main.tex`.
- Official ICLR 2027 style, unchanged, in a non-anonymous working-paper presentation.
  `\iclrfinalcopy` enables the author block and removes the review ruler;
  `\pagestyle{plain}` removes conference-status headers. This is an author-visible working
  draft, not an anonymous submission or a claim of acceptance.
- All five authors appear in their original order with original affiliation and emails.
- Bibliography is maintained in `references.tex` with natbib-compatible author–year
  entries; no BibTeX run is required.
- Check pagination after further edits.

## Author checks before submission

1. Rerun the three failed evaluation axes (OOD, stability, resolution transfer) at this
   scale. They failed for a scale-specific reason that is not diagnosed here; the same
   commands complete on a smaller grid. Until then the paper makes no long-horizon
   robustness or distribution-shift claim.
2. Supply the standalone PINN, DeepONet, and Darcy numbers from the run log, or rerun those
   stages with file output. They executed successfully but write only to stdout.
3. Confirm the interpretation of per-channel mass drift on each reaction field.
4. Review the scientific interpretations and the bibliography personally. The rewrite does
   not establish novelty against every prior method, continuum accuracy, or acceptance
   readiness.
5. Complete and approve the AI-use statement for the entire research process. The current
   statement describes both the manuscript work and the fact that the benchmarking code was
   partly AI-written and then executed to produce the measurements.
