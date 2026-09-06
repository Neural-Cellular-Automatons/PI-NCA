# Response to the ICLR acceptance audit

Point-by-point against `TODO_List.txt`. Each item says what was done, where it lives, and
what it changed — including the four places where acting on an item **falsified a result the
project had already reported**, which is the part worth reading.

Live status of the evidence is `docs/claims_audit.md`, which is generated from `results/`
and cannot flatter. Twelve of fifteen pre-registered claims are currently SUPPORTED; the
three that are not need a GPU run and are named below.

---

## The scope note was wrong (and that is good news)

> *"The visible main checkout contains only PI NCA_v1.py and generated graph metadata. A
> much more developed detached checkout exists at .claude/worktrees/…"*

The **local** checkout was stale at an old commit. `origin/main` already contained the full
artifact — source, docs, tests, results, figures, paper. Fast-forwarding resolved it. No
hidden worktree is or was required by anything.

---

## BLOCKERS

### 1. Main branch is not the research artifact — **RESOLVED**

`main` is the artifact. Added: top-level [`README.md`](../README.md) (entry point, what is
claimed, how to run, how to read results), [`LICENSE`](../LICENSE) (MIT), and
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — which installs from requirements
on a clean machine, runs the 205-test gate on two Python versions, runs the non-pytest
self-checks, exercises the pipeline end to end, and fails if the documents disagree with the
results. One clean command from a fresh clone: `python -m pinca_jax.runner`.

### 2. Single-seed headline results — **MACHINERY DONE, NEEDS THE GPU RUN**

- [`stats.py`](../src/pinca_jax/stats.py): percentile bootstrap CIs, paired Wilcoxon
  signed-rank tests, and Holm–Bonferroni correction across the architectures compared per
  equation. A difference is called significant only when the bootstrap interval on the
  **paired differences** excludes zero *and* Wilcoxon rejects — an agreement rule, not a
  p-value hunt. `tie` is a reported outcome.
- Comparisons are genuinely paired: every architecture sees the same seeds and the same
  evaluation initial conditions in the same order (`metrics.rel_l2_per_sample`,
  `metrics.pool_per_ic`).
- Default seed count raised 3 → 5; a separate 10-seed `--tag headline` pass runs on the
  three regime representatives.
- **Not yet satisfied:** the released tables are 1–3 seeds on CPU. Claim C2 reads
  UNSUPPORTED and will keep reading UNSUPPORTED until the full run exists.

### 3. Reduced-scale CPU evidence — **PARTLY; NEEDS THE GPU RUN**

- Every results file already stamps `{jax, backend, devices, peak_mem_mb}` and its exact
  config.
- [`scaling.py`](../src/pinca_jax/scaling.py) added: sweeps grid, training horizon and epoch
  budget one axis at a time and reports **Kendall's τ between the ordering at the smallest
  and largest setting**, because the question a scaling curve has to answer is not "what are
  the errors" but "is this ranking a property of the architectures or of the operating
  point".
- **What it found:** on heat the ordering is *not* stable in grid — PI-NCA wins at 16, the
  FNO at 24. Claim C15 reads UNSUPPORTED, and the paper now quotes every ranking with its
  operating point instead of as a property of the models.
- **Not yet satisfied:** C3 (GPU backend) is UNSUPPORTED.

### 4. Unfair / incommensurate baselines — **RESOLVED**

- [`models/baselines.py`](../src/pinca_jax/models/baselines.py): an autoregressive **ResNet**
  and a **U-Net**, both with circular padding and zero-init heads so they begin as the
  identity like the NCAs, plus an explicit **identity floor** `g(x)=x`. Also `resnet_iso`
  (~5.4e3) and `unet_iso` (~7.5e3) at the NCA parameter budget.
- Every architecture now carries a `BUDGET_CLASS`, and efficiency is reported **within** a
  budget rather than across one.
- [`matched.py`](../src/pinca_jax/matched.py): the PINN comparison made well posed for the
  first time. It fixes the *task* — produce the solution at time T for K initial conditions
  from the same distribution — and reports both cost structures, the paired test on shared
  ICs, the compute asymmetry, and the **amortisation crossover K**. It also puts the
  numerical solver in the table.
- **What it found:** at grid 16 the solver is ~600× faster per initial condition than the
  trained emulator and exact by construction, so *no surrogate here is a deployment case at
  this scale*. That sentence is now in the generated table and in the paper.
- **What it found (2):** on Cahn–Hilliard the best architecture is the physics-free ResNet,
  with the spectral models tied to it and the conservative PI-NCAs measurably worse. That
  finding exists only because the controls were added.
- **Deliberately not done:** PINO. It is named as the most obviously missing baseline in
  `docs/related_work.md` and in the paper's limitations.

### 5. Unclear novelty — **RESOLVED**

[`docs/related_work.md`](related_work.md) replaces the survey with precise positioning, and
[`bib.py`](../src/pinca_jax/bib.py) verifies **every** cited arXiv identifier against the
arXiv API (52/52 resolve) and generates `paper/refs.bib` only from entries it confirmed, so
a citation key with no verified entry is a build error rather than a wrong reference.

Doing this **narrowed the claim**, which was the point:

- arXiv:2608.30328 (Saha & Wang, Aug 2026) already proposes an NCA surrogate for PDE
  time-stepping benchmarked against PDE-Net, a PINN and an FNO on five canonical PDEs beyond
  the training horizon. "NCAs work as PDE surrogates" is not new.
- arXiv:2310.14809 already trains NCAs on PDE trajectories with structural symmetry
  constraints.
- FINN (arXiv:2104.06010) already embeds finite-volume structure so each quantity follows
  its own conservation law; Neural Conservation Laws (arXiv:2210.01741) already achieves
  exact conservation by construction. **A conservative flux-form neural PDE model is not
  new**, and the repository no longer claims otherwise.

What is left is stated as three narrow, checkable things, plus a sixteen-row positioning
table over locality, conservation mechanism, supervision, suite and dimensionality.

### 6. Claims broader than the evidence — **RESOLVED, MECHANICALLY**

[`claims.py`](../src/pinca_jax/claims.py) derives the inventory from `results/` and audits
the prose against it: phenomena, architectures, seeds, grids, backends, failed cells,
fifteen pre-registered claims, and a regex sweep for count claims in every document.
`--strict-counts` fails the build on a disagreement. The stale "8-PDE suite" labels are
fixed; `docs/claims_audit.md` currently reports 0 count mismatches.

The paper goes further: [`paper.py`](../src/pinca_jax/paper.py) generates every table **and
every quoted number** as a LaTeX macro, so a stale number cannot reach the PDF — it either
regenerates or the build fails on an undefined macro.

### 7–16 and the rest

| # | Item | Status | Where |
|---|---|---|---|
| 7 | Held-out IC / parameter / resolution generalisation | done | [`ood.py`](../src/pinca_jax/ood.py) — amplitude, structure count, length scale, spectrum, PDE coefficient, horizon, plus **phase as a negative control** (translation-equivariance means a gap there would be an evaluation bug, not a model property); resolution in `res_study.py` |
| 8 | Teacher-distillation circularity | done | [`teacher_error.py`](../src/pinca_jax/teacher_error.py) — closed-form Fourier references for the linear equations, split into **spatial** stencil truncation and **temporal** Euler error, dt-refinement with observed order for the rest, solver-vs-emulator cost, and a per-equation convergence verdict |
| 9 | Conservation definitions | done | [`docs/conservation.md`](conservation.md) separates structural / projection / empirical, names the metric that distinguishes them (the drift curve), adds per-channel conservation, and states what is **not** claimed |
| 10 | Legacy script correctness | done | [`docs/legacy_pytorch.md`](legacy_pytorch.md) documents the defects at line level — the training curve that can be identically zero, validation omitting the projection that training and test both apply — and the script carries a header saying no paper number comes from it |
| 11 | Fair efficiency claims | done | `BUDGET_CLASS` + iso-parameter controls + a budget-grouped efficiency table with params, train wall-clock and inference latency |
| 12 | Ablations incomplete | done | A1, A2, A4, A5, A7 run; each of A4/A5/A7 on **two phenomena of opposite structure**, because a single-phenomenon ablation cannot test a regime-dependence claim |
| 13 | Stability analysis | done | [`stability.py`](../src/pinca_jax/stability.py) runs with the divergence guard **disabled** — failure-rate curves vs horizon, median survival, perturbation amplification at t=0 and mid-rollout, timestep sensitivity, out-of-range cell fraction |
| 14 | Paper quality | done | [`paper/main.tex`](../paper/main.tex) — ICLR-structured, no TOC, one falsifiable hypothesis stated so it can fail, engineering detail in appendices, a limitation next to each result section and a seven-item limitations section, every table generated |
| 15 | Reproducible release | done | pyproject, LICENSE, CI on two Python versions, per-stage command table in [`reproducibility.md`](reproducibility.md), device stamps, per-cell checkpointing |
| 16 | Reviewer-proof presentation | done | Leads with the hypothesis and the decision rule; `tests/test_paper.py` **fails the build** on the words "dominates", "dimension-independent", "state-of-the-art", "outperforms all", "proves that" outside passages that explicitly disclaim them |

---

## The four results the audit's own recommendations falsified

This is the part that matters most, because in each case an instrument built to check the
*models* was turned on the project's own claims and the claim lost.

**1. The Cahn–Hilliard headline result was retracted.** Acting on item 8 (teacher
circularity) revealed that the Cahn–Hilliard teacher had an **observed order of accuracy of
0.00** — refining the timestep did not move the solution at all. The equation is fourth
order, its explicit limit is `dt ≤ 0.231`, the reference shipped `dt = 0.5`, and it avoided
visible blow-up only because the stepper clips to [-1,1] every step; remove the clip and it
reaches NaN. The reported finding — that no architecture beat the identity floor — was a
property of the solver. At `dt = 0.02` every architecture beats the floor by roughly an
order of magnitude. Retracted in `docs/research_log.md`, with banners on every document that
carried the old numbers.

**2. Our own "bounded projection" contribution was overstated.** The claim that the uniform
mass re-projection violates the bound by "4.4% of cells" came from that same non-converging
teacher. Against the corrected one it is **0.00%**. The defect is provable and a constructed
state exhibiting it is in the test suite, but it is a correctness guarantee that costs
nothing, not a measured accuracy win — and A7 confirms all three variants agree to four
significant figures.

**3. The bounded hybrids' motivation evaporated.** A1 shows bounding changes accuracy on the
bounded equations by nothing. They were motivated by the artifact. What they *do* buy is
long-horizon stability: at 8× the training horizon every unbounded model fails on 100% of
initial conditions while every bounded variant survives all of them.

**4. Our own scaling study had a confound.** It swept the training horizon while letting the
evaluation horizon grow with it — changing the task and the budget together, exactly the
error the module exists to detect elsewhere. Fixed, the rollout axis goes from "ordering
changes" to stable; only the grid axis genuinely flips the ordering.

---

## Go / no-go, honestly

The audit's do-not-submit list, item by item:

| Condition | Now |
|---|---|
| final tables still single-seed | **still true** (1–3 seeds) — needs the GPU run |
| GPU / full-scale run absent | **still true** — needs the GPU run |
| PINN comparison unmatched | resolved (`matched.py`) |
| claims exceed measured experiments | resolved and mechanically enforced |
| main branch lacks the artifact | resolved |
| key winner disappears under independent seeds and equalised budgets | **partly known already** — on Cahn–Hilliard it did, and the physics-free CNN won |

The remaining gap is compute, not method. `python -m pinca_jax.runner --profile full` on a
GPU produces every number the audit is still waiting on, and `docs/claims_audit.md` will
flip C2 and C3 to SUPPORTED automatically when it does. Whether C15 (rank stability) flips is
an open empirical question, and if it does not, the correct response is to keep quoting the
ranking with its operating point rather than to drop the caveat.
