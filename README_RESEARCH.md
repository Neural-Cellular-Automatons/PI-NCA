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
**Research paper (detailed, LaTeX in .txt):** `docs/research_paper.txt` — "No Universal Winner:
When Physics-Informed Neural Cellular Automata Beat (and Lose to) PINNs and Neural Operators on
PDEs" (14 sections, 8 tables, 39 references; rename to `.tex` to compile).
**True-3D volume renders:** `docs/figures/<pde>_3d_volume.png` (`viz3d_volume.py`).

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
The published tables were produced on a **CPU-only host** ⇒ reduced-scale configs (small
grids/steps/seeds) for an end-to-end, reproducible methodology demonstration. Configs re-run
unchanged on GPU — only numeric fields change. `pmap`/sharding are implemented but no-ops on
one device. See `docs/environment.md`.

## Quickstart (CPU)
```bash
python -m pip install -r requirements-jax.txt   # see environment.md for Windows no-admin notes
python -m pip install -e .                      # src layout -> `python -m pinca_jax.*` works
python -m pytest tests/                          # migration correctness gate
```

## Quickstart (GPU — full benchmark run)
Headless, terminal-only; full instructions in **`docs/gpu_runbook.md`**.
Needs Python >= 3.12 (3.13 preferred). Windows hosts must run inside WSL2 — JAX publishes no
native-Windows GPU wheels.
```bash
bash setup_gpu.sh                                # builds .venv, installs jax[cuda12], verifies the GPU
source .venv/bin/activate
bash run_gpu.sh smoke                            # ~2 min wiring check
bash run_gpu.sh                                  # full run: gate -> 2D -> 3D -> res-study -> figures -> plots
```
Every driver prints its backend and stamps `{jax, backend, devices, peak_mem_mb}` into
`results/*.json` under `"device"`, so a GPU run is self-identifying after the fact.

## Benchmark plots
```bash
python -m pinca_jax.plots        # reads results/*.json only — no training, seconds
```
Writes `docs/figures/bench/`: accuracy / PSNR / conservation / train-time / throughput bars,
error-growth profiles, accuracy-vs-params Pareto, the regime map, the 3-D suite, ablations
A4/A5, and resolution-transfer heatmaps.

## Metrics (every architecture, multi-seed mean ± std)
L2 error · relative error · residual loss · BC satisfaction · generalization · stability ·
wall-clock train time · inference speed · memory · parameter count · FLOPs.
