# PI-NCA Research Program — NCA vs PINN vs Operator Learning for PDEs

A rigorous, JAX-based comparison of Physics-Informed NCAs against PINNs and neural operators
for PDE-governed physical systems. The objective is **not** to prove NCAs superior, but to
characterize **the regimes where PINNs, NCAs, operators, and hybrids each win.**

## Deliverables (status)
| # | Deliverable | File(s) | Status |
|---|---|---|---|
| 1 | Migration report | `docs/migration/` (4 docs) | ✅ done (gate 36/36) |
| 2 | Literature review | `docs/literature_review.md` | ✅ done |
| 3 | Architecture report | `docs/architecture_report.md` | ✅ done |
| 4 | Experimental report | `docs/experimental_report.md` | ✅ heat + CH (more queued) |
| 5 | Ablation report | `docs/ablation_report.md` | ✅ A1 (A2–A6 queued) |
| 6 | Performance benchmarks | `results/bench_*.{json,md}` | ✅ heat, CH, CH-ablation |
| 7 | Reproducibility guide | `docs/reproducibility.md` + `environment.md` | ✅ done |
| 8 | Final paper-style summary | `docs/final_summary.md` | ✅ done |
| — | Running research log | `docs/research_log.md` | ✅ live |
| + | Master results (all tables) | `docs/master_results.md` | ✅ done |
| + | Efficiency comparison | `docs/efficiency_comparison.md` | ✅ done |
| + | Visual gallery (analytic/model/error) | `docs/figures.md` + `docs/figures/*.png` | ✅ 8 phenomena |
| + | CAX accelerator evaluation | `docs/cax_evaluation.md` | ✅ done |

**Phenomena benchmarked (2-D):** heat, Cahn–Hilliard, Allen–Cahn, shallow-water, Gray–Scott,
FitzHugh–Nagumo, Nagumo, advection–diffusion, wave, Navier–Stokes (emulators); Darcy (steady
operator); PINN + DeepONet (continuous/operator, heat).
**3-D:** heat, advection–diffusion, Allen–Cahn, Nagumo, Gray–Scott, FitzHugh–Nagumo
(`bench3d.py`, NDHWC 16³). **Gate: 56/56** (`python -m pytest tests/`).

**Protocol:** single fixed seed (42) + He-init + zero-init heads + LR warmup + pre-seeding —
the originals' "start from a better point" recipe (`docs/initialization_and_protocol.md`).
**Architecture diagrams:** `docs/architecture_diagrams.md`.

## Branch map (research trail)
```
main / claude/*            original PyTorch (PI NCA_v1.py)
research/jax-migration     foundation: lit review, JAX core (src/pinca_jax/), migration + correctness
  ├─ research/baseline-pinn
  ├─ research/baseline-nca
  ├─ research/physics-informed-nca
  ├─ research/fno-baseline
  ├─ research/nca-fno-hybrid
  ├─ research/operator-nca-hybrid
  ├─ research/ablation-studies
  └─ research/final-comparison   merges results
```

## Equation suite (shared across all architectures)
Heat / heterogeneous heat · Gray–Scott · Shallow-Water · FitzHugh–Nagumo · Cahn–Hilliard.
See `docs/literature_review.md §0` for formulations and per-branch provenance.

## Compute scope
CPU-only host this session ⇒ **reduced-scale** configs (small grids/steps/seeds) for an
end-to-end, reproducible methodology demonstration. Configs are written to re-run unchanged on
GPU (only numeric fields change). `pmap`/sharding are implemented but no-ops on one CPU.
See `docs/environment.md`.

## Quickstart
```bash
python -m pip install -r requirements-jax.txt   # see environment.md for Windows no-admin notes
python -m pytest tests/                          # migration correctness gate (added in Phase 1)
```

## Metrics (every architecture, multi-seed mean ± std)
L2 error · relative error · residual loss · BC satisfaction · generalization · stability ·
wall-clock train time · inference speed · memory · parameter count · FLOPs.
