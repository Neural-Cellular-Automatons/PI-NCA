# PI-NCA Research Program — NCA vs PINN vs Operator Learning for PDEs

A rigorous, JAX-based comparison of Physics-Informed NCAs against PINNs and neural operators
for PDE-governed physical systems. The objective is **not** to prove NCAs superior, but to
characterize **the regimes where PINNs, NCAs, operators, and hybrids each win.**

## Deliverables (status)
| # | Deliverable | File(s) | Status |
|---|---|---|---|
| 1 | Migration report | `docs/migration/` (per branch) | in progress |
| 2 | Literature review | `docs/literature_review.md` | ✅ draft |
| 3 | Architecture report | `docs/architecture_report.md` | pending |
| 4 | Experimental report | `docs/experimental_report.md` | pending |
| 5 | Ablation report | `docs/ablation_report.md` | pending |
| 6 | Performance benchmarks | `results/` | pending |
| 7 | Reproducibility guide | `docs/environment.md` + this file | ✅ draft |
| 8 | Final paper-style summary | `docs/final_summary.md` | pending |
| — | Running research log | `docs/research_log.md` | ✅ live |

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
