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
main                       original PyTorch (PI NCA_v1.py)
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

## Quickstart

**Python 3.12, 3.13 or 3.14.** GPU requires Linux or WSL2 — JAX publishes no
native-Windows CUDA wheels.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # reference impl only
pip install -r requirements-gpu.txt      # or requirements-jax.txt for CPU
pip install -e .                         # src layout -> `python -m pinca_jax.*` resolves
python -m pytest tests/ -q               # correctness gate
```

## The benchmark run — one command

```bash
bash run_gpu.sh
```

Gate, uniform 2-D matrix, ablations, uniform 3-D matrix, resolution study, baselines,
trajectory capture, field figures, plots, and the regenerated report. Presets:
`--profile smoke` (~2 min wiring check) and `--profile bench` (measurements only).

It is **resumable** (results checkpoint after every model, so a crash costs one model,
not a night — just run it again), **OOM-tolerant** (a model that runs out of device
memory is retried at half the batch, then recorded as failed while the sweep
continues), and **GPU-only** (the drivers refuse to run on the CPU backend rather than
silently producing numbers that cannot be compared to a GPU run).

Full detail, including WSL2 setup and troubleshooting: **`docs/gpu_runbook.md`**.

## Uniform benchmark matrix

Every architecture runs on every phenomenon — the same competitor list in every table.
The flux-form models used to hardcode a 2-channel flux head, so they only applied to
single-channel fields; multi-field phenomena were measured with three models while
scalar ones got five. All models are now generic in the channel count (one flux pair per
field, per-channel divergence, per-field mass projection), and the bounded variants take
each PDE's measured physical range instead of a hardcoded [-1,1].

At C = 1 the numerics are unchanged, so prior results still stand —
`tests/test_uniform_matrix.py` asserts it and the PyTorch migration gate still passes.

## Benchmark plots and the report
```bash
python -m pinca_jax.plots        # reads results/*.json only — no training, seconds
python -m pinca_jax.report       # regenerates the report's tables from those results
python -m pinca_jax.md2pdf docs/PI-NCA_Architectures_and_Results.md
```
The report's results tables are **generated from the JSON**, so the document cannot
quote a smaller or older set of architectures than the benchmarks actually produced.
Writes `docs/figures/bench/`: accuracy / PSNR / conservation / train-time / throughput bars,
error-growth profiles, accuracy-vs-params Pareto, the regime map, the 3-D suite, ablations
A4/A5, and resolution-transfer heatmaps.

## Field figures — capture once, render forever
`pinca_jax.capture` trains one model per phenomenon and archives the raw solver/model
trajectories to `results/traj/*.npz` (2-D `(T+1,H,W)`, 3-D full volumes `(T+1,D,H,W)`).
Every montage, GIF and rotating 3-D render is then produced *from those files* — no
training, no GPU, plain numpy — so figures can be rebuilt or restyled later on any machine:
```bash
python -m pinca_jax.capture --dims both
python -m pinca_jax.viz3d_volume --npz results/traj/heat_3d.npz
```
`--max-mb` (default 64/phenomenon) strides the time axis to bound file size. See
`docs/gpu_runbook.md` §4b.

## Metrics (every architecture, multi-seed mean ± std)
L2 error · relative error · residual loss · BC satisfaction · generalization · stability ·
wall-clock train time · inference speed · memory · parameter count · FLOPs.
