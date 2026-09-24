# Response to the review: what was done, what it showed, what is left

One row per point raised. "Cleared" means the work is finished and the evidence is in the
repository; "cleared, GPU pending" means the experiment is built, tested and verified to
run, but the numbers that can go in the paper need the 4090; "paper-side" means it is a
wording or layout change in `paper_updated/`, which you said you are handling — for those
the row records only whether the current committed source still shows the problem.

Commits: `519b728` (experiments), `16a7d08` (screen results), `d205149` (old LaTeX
removal). Gate: 263 tests pass.

| # | Point raised | Status | Evidence |
|---|---|---|---|
| 1 | FNO + flux head, U-Net-S + flux head on heat / SW / CH | **cleared, GPU pending** | §1; `fno_flux`, `unet_iso_flux`, `resnet_iso_flux` |
| 2 | FINN as a baseline, at least on shallow water | **cleared, GPU pending** | §2; `models/finn.py` |
| 3 | "If promising great, else reframe as flux heads help small local models" | **evidence favours the reframe** | §1, §2 |
| 4 | Prop 2 applies to PI-NCA too — one architecture with a width setting | paper-side; support done | §5 |
| 5 | "Isolates the capacity of the joint perception" is wrong — it isolates width | paper-side; confirmed correct | §5 |
| 6 | Table 2 takes the best of three configurations per PDE | **cleared** | §3; pre-declared rule + `bounded_mc_nca` |
| 7 | Restrict the "stabiliser" claim to "relative to the unconstrained NCA" | paper-side; confirmed correct | §4 |
| 8 | Error-growth contradicts the stabiliser claim; get guard-off stability at 48² | **cleared, GPU pending** | §4; root cause found and fixed |
| 9 | 3-point LR sweep + 4× budget for FNO and ResNet on CH and SW, with loss curves | **cleared, GPU pending** | §6; `tuning.py` |
| 10 | Rank stability excludes the runner-up; add full ResNet/U-Net, ≥3 seeds, 96²/128² | **cleared, GPU pending** | §7; `scaling --tag wide` |
| 11 | Missing spaces in the abstract | fixed | verified absent from `main.tex` |
| 12 | The abstract defines PI-NCA twice | fixed | verified: one definition |
| 13 | Internal pipeline language | **3 instances remain** | §9 |
| 14 | Reference list not alphabetical | **still out of order** | §9 |
| 15 | Appendices numbered 8–16 rather than lettered | **still numbered** | §9 |
| 16 | Tesán et al. published — cite the journal version | **still an arXiv preprint** | §9 |
| 17 | "Wave, dispersive" — check it | **cleared, final** | §8; measured |
| 18 | Non-significance treated as equivalence; the U-Net tie is not tabulated | **cleared** (numbers); wording paper-side | §10 |
| 19 | `p = 3.1×10⁻¹⁴` does not say what it tests | identified; wording paper-side | §10 |
| 20 | "Above/below the floor" ambiguous | paper-side | §9 |
| 21 | "Small budget" includes MC-PI-NCA at 9,936 params | confirmed factual | §10 |
| 22 | "Within 30% of the solver" holds at 10 seeds, not 5 | **cleared** | §10 |
| 23 | No lumped-vs-per-field ablation backs Prop 2 | **cleared** | §11 |

Run everything that is GPU-pending with one command:

```bash
bash run_paper.sh --only fluxhead,declared,tuning,dispersion,scaling_wide,stability,ood
```

Roughly 8–10 h on a 4090, resumable, every stage non-fatal.

---

## §1 — Which part does the work: the automaton or the flux head?

The conservative flux head is now a one-line option on three backbones that are **not**
cellular automata. It costs under 1% of parameters in every case, so each pair isolates
the head:

| pair | params | what differs |
|---|---|---|
| `fno` / `fno_flux` | 592,897 / 592,922 | global spectral backbone, state head vs flux head |
| `unet_iso` / `unet_iso_flux` | 7,464 / 7,468 | multi-resolution backbone |
| `resnet_iso` / `resnet_iso_flux` | 5,364 / 5,376 | local backbone, per-layer weights |

`tests/test_flux_probes.py` asserts that each flux variant conserves every field for
arbitrary weights and that each non-flux sibling does not, so the pair cannot silently
stop measuring the head.

**Reduced-scale screen** (grid 24, 250 epochs, 2 seeds, CPU — indicative only, tag
`probe_cpu24`, never mixed into a paper table). Every entry is a paired test on shared
initial conditions, all at *p* ≤ 0.04:

| backbone | heat | shallow water | Cahn–Hilliard |
|---|---|---|---|
| FNO | **2.13× better** | **2.38× better** | 0.78× (worse) |
| ResNet-S | **1.65× better** | **2.99× better** | 0.97× (worse) |
| U-Net-S | **1.47× better** | **1.31× better** | 0.94× (worse) |

The head also transfers the conservation property: on heat the FNO's end-of-rollout mass
error falls from 1.5 to 2.4×10⁻⁴, the U-Net-S's from 5.1 to 3.8×10⁻⁴.

**Reading.** On conservative dynamics the flux head is the effect, and it is not a
property of the cellular automaton. On stiff fourth-order dynamics it costs accuracy on
all three backbones — the same trade the A4 ablation already shows for PI-NCA itself. At
this scale `fno_flux` beats PI-NCA on heat by 4.5×.

## §2 — FINN

`models/finn.py` implements a FINN-style facewise flux network (Praditia et al. 2021;
Karlbauer et al. 2022). PI-NCA learns a **cellwise** rule that emits a cell's outgoing
flux; FINN learns a **facewise** rule applied to the pair of states across each face,
split into a diffusive part proportional to the jump and an advective part carried by the
upwind cell. Both conserve by the same telescoping argument, so it is a like-for-like
conservative baseline. A test pins the facewise structure: perturbing one cell must move
exactly that cell and its four neighbours.

Screen result — FINN is the strongest conservative model at this scale, on all three
phenomena, at a matched parameter count (4,672 vs 4,576):

| phenomenon | FINN | PI-NCA | ratio | *p* |
|---|---|---|---|---|
| heat | 0.00483 | 0.0128 | 2.65× | 6.1e-5 |
| shallow water | 0.00318 | 0.00381 | 1.20× | 6.1e-5 |
| Cahn–Hilliard | 0.0710 | 0.178 | 2.51× | 3.1e-5 |

**Caveat that must be checked at full scale:** FINN is parameter-matched but **not
compute-matched** — four small MLPs per step against PI-NCA's one. Its measured latency is
1.18 ms/step against PI-NCA's 0.95 ms on the same CPU run, so the gap is real but modest;
the GPU run will settle whether accuracy-per-millisecond still favours it.

**Consequence for the paper (point 3).** On current evidence the defensible claim is the
one you named: *flux heads help, including on models that are not cellular automata*, with
PI-NCA as the smallest member of that family rather than the contribution. Full-scale
numbers may move the ratios, but three independent backbones and three phenomena all
point the same way.

## §3 — The pre-declared configuration rule

The rule is now fixed in advance and applied everywhere:

> **Width:** the wide setting. **Projection:** enabled if and only if the equation's state
> has a hard physical range (Allen–Cahn, Cahn–Hilliard).

`bounded_mc_nca` was added because that rule needs a wide-and-bounded cell that the
original run never trained; it is parameter-identical to MC-PI-NCA (the projection has no
parameters). `paper_updated/pinca_tests.py` prints the rank of the pre-declared
configuration on every phenomenon under **both** rules, so the cost of committing to one is
visible:

| phenomenon | compact rule | wide rule |
|---|---|---|
| shallow water | **1** / 11 | 2 / 11 |
| Cahn–Hilliard | **1** / 11 | needs `bounded_mc_nca` (GPU) |
| advection–diffusion | 2 | 3 |
| heat | 7 | 2 |
| wave | 6 | 8 |
| Navier–Stokes | 7 | 9 |
| Allen–Cahn | 8 | needs `bounded_mc_nca` (GPU) |
| Gray–Scott | 9 | 7 |
| Nagumo | 8 | 10 |
| FitzHugh–Nagumo | 10 | 8 |

**This is the sharpest cost of the review.** The published table's "first on two, second on
two" came from choosing per phenomenon. Under a single declared rule the compact setting
keeps both firsts but puts heat at 7th; the wide setting lifts heat to 2nd but has no
firsts pending the two bounded cells.

## §4 — Long-horizon stability, guard off, at 48²

**Root cause found.** Neither the stability driver nor the OOD driver had the
out-of-memory backoff the accuracy matrix has (`bench.run_with_oom_backoff`), so a single
allocation failure killed the whole stage. That is why six stages died in the GPU run. Both
drivers now retry at half the batch, and both were verified to complete at grid 48 here,
including the 384-step guard-off rollouts.

**Your reading of the error-growth column is confirmed.** On heat at ten seeds, PI-NCA's
error-growth ratio is 5.52 against the identity floor's 3.70 and the FNO's 2.11 — PI-NCA's
error does grow faster than everything except the standard NCA (22.87). The "stabiliser"
claim cannot rest on that column; it can only be stated relative to the unconstrained NCA
until the guard-off study is rerun.

## §5 — One architecture, one width setting

No experiment needed; recording the conclusion so it is not re-litigated. PI-NCA and
MC-PI-NCA share the per-field divergence, and on a single field they differ **only** in
width (32/64 vs 48/96 channels). Both perceive all fields jointly through the same 3×3
convolution, so a comparison between them isolates width, not "the capacity of the joint
perception". The two propositions belong to the architecture, not to the wide variant.

## §6 — Were the large baselines under-trained?

`src/pinca_jax/tuning.py`: three learning rates (3e-4, 1e-3, 3e-3) at the shared budget,
then the best learning rate at **four times** the budget, for the large baselines *and* the
PI-NCA models — sweeping only the baselines would replace one unfair comparison with
another. It records the training-loss curve (subsampled to 24 points) and a
`still-improving` ratio: the mean loss over the middle tenth of training divided by the
mean over the last tenth, so "it had converged" is shown rather than asserted. Output:
`results/tuning_<pde>.{json,md}`, with a table of each model's best setting and what that
does to the ordering.

## §7 — Rank stability including the runner-up

`scaling.py` gained `--tag`, so a second sweep cannot overwrite the one the paper reports.
The `scaling_wide` stage sweeps eight models — including the **full-size ResNet and U-Net**,
which is where the Cahn–Hilliard runner-up lives — at three seeds, over grids 48 → 96
(`full` profile: 48 → 128). Output: `results/scaling_<pde>_wide.json`.

## §8 — Wave dispersion: measured, finished

You were right, and it is now quantified. The continuum 2-D wave equation is exactly
non-dispersive; this discretisation is not. `src/pinca_jax/dispersion.py` computes the
closed-form discrete relation and fits the realised frequency from the solver itself; the
two agree to four decimals at every mode, so the error is the scheme's and not the fit's.

| mode | ω exact | ω discrete | ω measured | phase error |
|---|---|---|---|---|
| (1,0) | 0.0654 | 0.0654 | 0.0654 | −0.1% |
| (8,0) | 0.5236 | 0.5000 | 0.5000 | −4.5% |
| (16,0) | 1.0472 | 0.8661 | 0.8661 | −17.3% |
| (24,0) | 1.5708 | 1.0001 | 1.0001 | −36.3% |

Semi-discrete and fully discrete frequencies agree to four decimals at this timestep, so
essentially all of the dispersion comes from the 5-point Laplacian rather than the time
integrator. At the grid scale the phase speed is 0.32 against the exact 0.5. The correct
label is "non-dispersive continuum, dispersive discretisation".

## §9 — Paper-side items: what the committed source still shows

You said the wording changes are handled. Four are **not** yet in the committed source, so
they are listed with locations rather than assumed:

| item | state in `paper_updated/` |
|---|---|
| Abstract spacing (`at all.We`, `FNO.On`, `FNO.Overall`) | fixed — none present |
| Abstract defines PI-NCA twice | fixed — one definition |
| Internal pipeline language | **3 remain**: `appendix.tex:202` "the migration gate … kept in the registry"; `appendix.tex:285` "Skipped by the runner because a previous file was present"; `appendix.tex:333` "two defects found during the migration" |
| Reference list alphabetical | **not fixed** — Richardson precedes Praditia and Karlbauer in `references.tex` |
| Appendices lettered A, B, … | **not fixed** — no `\appendix`, so they are sections 8–16 |
| Tesán et al. journal version | **not fixed** — `references.tex:159` still cites arXiv:2507.08861 |
| "Above/below the floor" | **not fixed** — `main.tex:495` "still above the floor" |

## §10 — Statistics and wording corrections that needed numbers

**Non-significance is not equivalence (point 18).** `pinca_tests.py` now prints the
confidence interval of the paired difference and its half-width, so a tie is reported as a
bound:

| comparison | mean difference | 95% CI | resolution |
|---|---|---|---|
| heat: MC-PI-NCA vs U-Net | −3.15×10⁻⁴ | [−9.25×10⁻⁴, +3.87×10⁻⁴] | ±6.6×10⁻⁴ |
| shallow water: PI-NCA vs MC-PI-NCA | −3.32×10⁻⁴ | [−7.61×10⁻⁴, +7.47×10⁻⁵] | ±4.2×10⁻⁴ |

Neither supports "indistinguishable"; both support "any difference is smaller than the
sample size can resolve". The U-Net comparison now exists as a number, which it did not
before.

**Point 19.** `p = 3.1×10⁻¹⁴` is PI-NCA versus the identity floor on Navier–Stokes at ten
seeds (0.441 against 0.513, PI-NCA better on 94% of initial conditions).

**Point 22 — the 30% claim.** Confirmed: the FNO's heat error is **26.6%** above the
solver's own error at ten seeds (0.001447 vs 0.0011428) and **37.0%** at five seeds
(0.001566). The two sample sizes do not support the same sentence.

**Point 21 — the "small" budget class.** Confirmed factual: MC-PI-NCA sits in that class at
9,936 parameters against 5,364–7,464 for the `-S` baselines, i.e. 1.33–1.85× larger. The
class is a coarse bucket rather than a matched-budget claim; either the bucket boundary or
the wording needs to change, and the parameter column is already printed beside every row.

## §11 — Lumped versus per-field conservation (Prop 2)

`pi_nca_lumped` is the wrong constraint built deliberately: same trunk, unconstrained head,
and a projection restoring the sum over **all** fields and cells. Parameter-matched to
PI-NCA within 2%.

Measured on shallow water (three conserved fields), reduced-scale screen:

| model | rel-L2 | lumped total error | per-field errors |
|---|---|---|---|
| PI-NCA | 0.00403 | 9.9×10⁻⁵ | 1.8×10⁻⁴, 1.3×10⁻⁷, 9.5×10⁻⁷ |
| MC-PI-NCA | 0.00437 | 1.8×10⁻⁴ | 3.1×10⁻⁴, 1.3×10⁻⁷, 1.1×10⁻⁶ |
| lumped control | 0.0240 | 6.7×10⁻⁴ | **2.4, 7.0, 4.7** |

The lumped control keeps the total to 6.7×10⁻⁴ while the individual fields drift by
order-unity amounts, and it is **6× less accurate**. That is exactly the failure mode the
per-field proposition describes, now measured rather than argued.

---

## What the screen cannot settle

The screen is grid 24, 250 epochs, two seeds, CPU. Cahn–Hilliard in particular does not
reproduce its own regime at that scale: the bound projection is worth nothing there
(0.1784 against 0.1786 unprojected, a tie) where at 48² with the full budget it was worth
30%, and `finn_src` scores *better* than plain FINN, which contradicts the conservation
argument and is most likely a short-horizon artefact. Every Cahn–Hilliard conclusion has to
come from the GPU run.

## File index

| What | Where |
|---|---|
| Flux head on other backbones | `models/fno.py`, `models/baselines.py` (`flux=True`) |
| FINN-style baseline | `models/finn.py` |
| Lumped-conservation control | `models/flux_nca.py` (`LumpedConsNCA`) |
| Wide + bounded cell | `models/registry.py` (`bounded_mc_nca`) |
| LR / budget sweep | `src/pinca_jax/tuning.py` |
| Wave dispersion | `src/pinca_jax/dispersion.py`, `results/dispersion_wave.{json,md}` |
| OOM backoff fix | `src/pinca_jax/stability.py`, `src/pinca_jax/ood.py` |
| Rank-stability tagging | `src/pinca_jax/scaling.py` (`--tag`) |
| Pre-declared rule, equivalence tests, flux-head summary | `paper_updated/pinca_tests.py` |
| Correctness gate for all of it | `tests/test_flux_probes.py` (30 tests) |
| Screen results | `results/bench_*_probe_cpu24.{json,md}` |
| Method notes | `docs/review_experiments.md` |
