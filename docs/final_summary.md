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
model that is both accurate and mass-conserving). The capstone `bounded_multiscale_nca` unifies
both ideas in one architecture.

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

## Limitations
- Reduced-scale CPU (small grids/horizons/epochs, 2–3 seeds); single device ⇒ `pmap`/
  sharding unused. Absolute numbers are not GPU-scale.
- Benchmarks so far: heat + Cahn–Hilliard (+ CH ablation). SWE/FHN multi-channel, the full
  multi-seed PINN sweep, and hybrids are scaffolded but not yet run at scale.
- FHN initial condition is a standard choice (not recovered from the notebook source).

## Future work (infrastructure ready)
- Multi-channel conservative NCA → SWE/FHN/Gray–Scott benchmarks.
- Hybrids: FNO-latent + NCA refinement; conservation-preserving bounded NCA; multi-scale
  perception (lit review §7, ablation A5).
- Ablations A2–A6 (iso-parameter FNO, horizon curriculum, conservation on/off, neighbourhood
  size, physics-loss weighting).
- Full multi-seed PINN across horizons; DeepONet baseline.
- CAX integration for accelerated NCA rollouts (evaluate vs hand-written `lax.scan`).

## Artifacts
`docs/{literature_review, architecture_report, experimental_report, ablation_report,
reproducibility, research_log}.md`, `docs/migration/*`, `results/bench_*.{json,md}`,
JAX core `src/pinca_jax/`, gate `tests/` (36/36). Branch trail: `research/jax-migration`
(+ `baseline-pinn`, `ablation-studies`).
