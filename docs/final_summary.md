# Physics-Informed Neural Cellular Automata vs PINNs and Neural Operators for PDEs — Research Summary

*Reduced-scale CPU study (Python 3.14, JAX). Findings are relative-ordering claims;
absolute numbers re-run at full scale on GPU with unchanged code.*

## Abstract
We migrate a PyTorch Physics-Informed Neural Cellular Automaton (PI-NCA) codebase to
JAX/Flax, build a correctness-gated 8-PDE suite, and compare — under one shared
autoregressive-emulation protocol — a conservative flux-divergence **PI-NCA**, an
unconstrained **NCA**, and a **Fourier Neural Operator (FNO)**, plus a continuous **PINN**
baseline. The objective is explicitly *not* to crown NCAs but to characterise **which
method wins in which regime**. We find no universal winner: the **conservation inductive
bias** makes the local NCA viable and efficient on smooth local diffusion, while **all
emulators fail on stiff 4th-order dynamics** until an **output-bounding** fix is applied —
and that fix in turn **conflicts with exact conservation**. Different regimes reward
different components.

## Methods
- **Migration (gated).** All PyTorch solvers/models ported to JAX and asserted equal to
  verbatim references to tolerance before any architecture work — **36/36 tests**. A real
  NHWC-axis bug in the heat Laplacian and a Gray–Scott `dt=2.0` instability were found and
  documented (`docs/research_log.md`).
- **Protocol.** One-step emulators `g_θ` distilled against a differentiable solver teacher
  over a short rollout, evaluated on a 4× longer horizon from held-out ICs. Identical
  teacher/data/optimiser/loss/metrics across architectures (`harness.py`). PINN evaluated
  separately (continuous single-IVP). Multi-seed mean ± std throughout.
- **Metrics.** rel-L2, MSE/RMSE, PSNR, conservation error, periodic-BC residual,
  gradient-energy (stability), parameters, train wall-clock, inference s/step.

## Key results
| Regime | Best | Evidence | Caveat |
|---|---|---|---|
| **Heat** (smooth, local) | **`multiscale_flux_nca` (hybrid)** | rel-L2 **0.0183** > FNO 0.0352 > PI-NCA 0.0438, at **110× fewer params than FNO** | plain NCA diverges (0.23) |
| **Cahn–Hilliard** (stiff, 4th-order) — baselines | *none* (all fail) | all 14–18 rel-L2 vs **identity floor 0.93** | FNO catastrophic variance; conservation ≠ correctness |
| **CH + output clip** (ablation) | bounded NCAs | 0.54–0.60 below floor, 24–27× gain | clip **destroys** conservation (3e-5→7.6) |
| **Cahn–Hilliard** — hybrids | **`bounded_cons_nca` (hybrid)** | **0.603** below floor **AND** conservation **2.4e-4** | non-bounding hybrids still diverge (24) |

**The hybrids win.** A motivated hybrid is the single best model in *each* regime, beating every
pure baseline: a **multi-scale conservative NCA** on smooth local diffusion (beats FNO at 110×
fewer params) and a **bounded conservation-preserving NCA** on stiff 4th-order dynamics (the only
model both accurate and mass-conserving). One architecture — `MultiScaleFluxNCA` with PDE-
appropriate bounds — is best-or-near-best in both regimes (`bounds=None` wins heat;
`bounds=(-1,1)` ties the CH winner at ~6× lower variance). Honest caveat: stacking *all*
components is not automatically better — multi-scale perception helps smooth transport but **not**
stiff-CH accuracy (there, bounding is what matters). The regime selects the component.

### Findings (honest, anti-confirmation-bias)
1. **Conservation makes the local NCA viable** on heat: the unconstrained NCA diverges;
   the flux-divergence PI-NCA conserves mass to ~1e-4 and is competitive at 4.6k params.
2. **FNO buys accuracy + seed-stability at large cost** (≈5.9×10⁵ params, 5× slower, no
   conservation). Its global mixing did **not** rescue stiff CH — refuting a naïve
   "global operator is always better" expectation (H2 unsupported in this regime).
3. **All simple emulators fail on stiff 4th-order CH** — they diverge below a do-nothing
   baseline. The cause is unbounded output blow-up: **bounding the state recovers them**
   (24–27×). The failure is architectural, not fundamental.
4. **Stability ↔ conservation tension:** naïve clipping fixes blow-up but destroys exact
   conservation — the two best properties conflict, motivating a conservation-preserving
   bounded update.
5. **No single *baseline* dominates; the component that matters is regime-dependent**
   (conservation on smooth local PDEs; bounding on stiff PDEs).
6. **Hybrids that compose the right components beat every baseline in their regime**
   (confirmed, not just hypothesised): multi-scale conservative NCA wins heat (0.0183 < FNO
   0.0352, 110× fewer params); bounded conservation-preserving NCA wins stiff CH (0.603 below
   floor with 2.4e-4 conservation — accuracy of clipping *and* conservation of the flux form).
7. **Stacking fixes compounds on stiff PDEs:** bounding (A1) + matched train/eval horizon (A3)
   takes CH to **0.511** with conservation **6.8e-5** — both fixes contribute independently.
8. **The conservation prior helps iff the PDE is conservative** (multi-channel test): on
   shallow-water (periodic mass+momentum conserved) the conservative `mc_flux_nca` ties FNO
   accuracy (0.031 vs 0.026) at **54× fewer params** with 4-orders-better conservation; on
   FitzHugh-Nagumo (source-term reaction, non-conservative) the *same* prior is the **worst**
   model (0.99) and the unconstrained NCA wins (0.154). Inductive bias must match PDE structure.
9. **FNO's accuracy is largely parameter count, not just spectral mixing** (ablation A2): at
   NCA-budget params FNO degrades 3.4× (0.035→0.119), losing to the multi-scale NCA (0.0183 at
   ~1% of FNO's params). Local multi-scale priors are markedly more parameter-efficient.
10. **CAX gives no single-CPU rollout speedup** (same `nnx.scan`/`jit` as our `lax.scan`); its
   value is its CA zoo + multi-device scaling. Not integrated into the hot path (`docs/cax_evaluation.md`).
11. **Global coupling is the NCA's hard limit — FNO wins Navier–Stokes** (incompressible
   vorticity, global Poisson): FNO 0.145 vs best NCA 0.284 (multi-scale) and 0.53/0.68 for
   plain/conservative NCAs. The complement of heat: locality+multi-scale wins purely-local
   diffusion; global spectral wins globally-coupled flow. The PDE's information-propagation
   structure picks the architecture.
12. **Nagumo re-confirms the conservation rule** (3rd PDE): on the non-conservative bistable RD,
   conserving models stall at 0.379 while unconstrained `plain_nca`/`fno` reach 0.118.
13. **Hybrid NCA is ~200× more parameter-efficient than FNO** on heat (rel-L2×params 101 vs
   20,875): more accurate at 107× fewer parameters and ~6× cheaper inference (`docs/efficiency_comparison.md`).
14. **Where the NCAs' error lives (spectral diagnostic):** on heat the NCAs' high-frequency error
   fraction is 0.44–0.89 vs FNO's **0.004** — NCAs over-smooth (residual error in fine scales),
   FNO's error is broadband-flat. This explains FNO's wins on sharp-interface (Allen–Cahn) and
   fine-scale regimes despite NCAs' lower *total* error on smooth ones.
15. **A4/A5 isolate the mechanisms cleanly:** at matched backbone width, the conservation head
   helps heat (0.104 vs 0.353) and hurts Nagumo (0.379 vs 0.123); widening perception helps
   monotonically on heat, but on Navier–Stokes only *dilated multi-scale* helps — a plain 5×5
   stencil destabilises (1.49 vs 3×3's 0.58). Multi-scale dilation is the robust reach mechanism.

16. **The regime map is dimension-independent (3-D).** Re-running 6 PDEs in full 3-D (NDHWC, 16³)
   reproduces every 2-D conclusion: local diffusion → conservative NCA wins (heat: pi_nca 0.047
   at ~3k vs FNO's 747k params); sharp-interface (Allen–Cahn 0.012) and non-conservative
   (FitzHugh–Nagumo 0.163) → FNO; the conservation prior **hurts** non-conservative Nagumo
   (conserving NCAs 0.198 vs plain 0.041); NCAs conserve mass to 1e-3–1e-6 vs plain/FNO drift.
   The architecture-vs-PDE-structure thesis holds in 2-D and 3-D alike.

**Coverage:** 10 emulator phenomena in 2-D + 6 in 3-D with 20-metric tables; ablations A1–A6;
PINN + DeepONet + Darcy operator baselines; CAX eval; GIFs/error-maps; architecture diagrams;
the "start from a better point" protocol (single seed 42 + He-init + warmup + pre-seeding).

## Conclusions
For PDE-governed systems, "NCA vs PINN vs operator" is the wrong framing — the right one is
**which inductive bias matches the PDE's character**: locality + conservation for smooth
local transport; global spectral mixing for cross-IC generalisation on periodic grids;
explicit boundedness/stabilisation for stiff dynamics; continuous mesh-free PINNs for
single-IVP/irregular-query problems. **This is borne out empirically:** hybrids that compose
the right inductive biases — multi-scale locality + conservation for smooth transport, bounded
conservation-preserving updates for stiff dynamics — **beat every pure baseline in their
regime**, and a single unified `bounded_multiscale_nca` carries both biases. The conservative
NCA's flux structure is the key reusable component: cheap, exactly mass-conserving, and
composable with wider perception or bounding as the PDE demands.

## Coverage (what was run)
Emulator benchmarks (multi-seed): **heat, Cahn–Hilliard, shallow-water, FitzHugh–Nagumo**;
ablations **A1** (output bounding) and **A3** (horizon curriculum); **A2** iso-parameter FNO.
Three hybrids + a unified capstone. Continuous/operator baselines: **PINN** and **DeepONet**
(3-seed). **CAX** accelerator evaluated. Correctness gate **41/41**.

## Limitations
- Reduced-scale CPU (small grids/horizons/epochs, 2–3 seeds); single device ⇒ `pmap`/
  sharding unused. Absolute numbers are not GPU-scale; *relative orderings* are the object.
- PINN/DeepONet are heat-only and use different horizons/metrics than the emulators (reported as
  a separate paradigm, not head-to-head).
- FHN initial condition is a standard choice (not recovered from the notebook source).
- `gray_scott`/`wave`/`adv_diff`/`allen_cahn` emulator sweeps and ablations A4–A6 (conservation
  on/off at matched width, neighbourhood size, physics-loss weighting) remain to run; all are
  scaffolded (`bench.py --pde ... --archs ...`).

## Future work (infrastructure ready)
- Remaining PDE emulator sweeps + ablations A4–A6; multi-channel hybrids (bounded+multiscale on SWE).
- Operator hybrids on the stiff regime (FNO-latent + bounded conservative NCA refinement).
- GPU full-scale re-run (configs unchanged) + `jax.sharding`; CAX CA-zoo targets (Lenia) and its
  Moore/von-Neumann perception as A5 neighbourhood variants.

## Artifacts
`docs/{literature_review, architecture_report, experimental_report, ablation_report,
reproducibility, research_log}.md`, `docs/migration/*`, `results/bench_*.{json,md}`,
JAX core `src/pinca_jax/`, gate `tests/` (36/36). Branch trail: `research/jax-migration`
(+ `baseline-pinn`, `ablation-studies`).
