# PI-NCA Research Program — status and provenance

> **Start at [README.md](README.md).** That is the entry point: what the project claims,
> how to run it, and how to read the results. This document is the research-programme
> record — deliverable status, the branch trail, and the scope of the compute — kept
> because the programme was specified as a list of deliverables and the list should stay
> auditable.

## NCA vs PINN vs Operator Learning for PDEs

A rigorous, JAX-based comparison of Physics-Informed NCAs against PINNs and neural operators
for PDE-governed physical systems. The objective is **not** to prove NCAs superior, but to
characterize **the regimes where PINNs, NCAs, operators, and hybrids each win.**

## Deliverables (status)
| # | Deliverable | File(s) | Status |
|---|---|---|---|
| 1 | Migration report | `docs/migration/` | done; asserted by the gate |
| 2 | Literature review | `docs/literature_review.md` | done (background) |
| 2b | **Related work + novelty positioning** | `docs/related_work.md` | done; every citation arXiv-verified |
| 3 | Architecture report | `docs/architecture_report.md`, `architecture_diagrams.md` | done |
| 4 | Experimental report | `docs/experimental_report.md` | done at the released scale |
| 5 | Ablation report | `docs/ablation_report.md` | A1/A2/A4/A5/A7 wired; A6 protocol documented |
| 6 | Performance benchmarks | `results/*.{json,md}` | uniform matrix, all phenomena |
| 7 | Reproducibility guide | `docs/reproducibility.md`, `environment.md`, `gpu_runbook.md` | done |
| 8 | Final paper-style summary | `docs/final_summary.md` | done |
| 9 | **Submission paper source** | `paper/main.tex` (+ generated tables) | done; every number generated |
| — | Running research log | `docs/research_log.md` | live |
| + | **Claims audit (generated)** | `docs/claims_audit.md` | live; `python -m pinca_jax.claims` |
| + | **Bibliography (arXiv-verified)** | `docs/bibliography.md`, `paper/refs.bib` | 52/52 resolve |
| + | **Conservation taxonomy** | `docs/conservation.md` | done |
| + | **Legacy script defects** | `docs/legacy_pytorch.md` | done, line-referenced |
| + | Master results (all tables) | `docs/master_results.md` | done |
| + | Efficiency comparison | `docs/efficiency_comparison.md` | superseded by the budget-class table |
| + | Visual gallery | `docs/figures.md` + `docs/figures/*.png` | 10 2-D + 6 3-D |
| + | CAX accelerator evaluation | `docs/cax_evaluation.md` | done; measured slower on both backends |

**Do not read the status column as a claim that the evidence is sufficient.**
`docs/claims_audit.md` is generated from `results/` and is the authority on what the
released numbers actually support; at the time of writing it correctly reports the seed
count, the backend and rank stability as unsupported.

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
Heat · advection–diffusion · wave · Allen–Cahn · Cahn–Hilliard · Gray–Scott · shallow water ·
FitzHugh–Nagumo · Nagumo · Navier–Stokes (2-D); six of them repeated in 3-D. Formulations,
parameters and the two documented stability overrides (Gray–Scott `dt=2.0`, Cahn–Hilliard
`dt=0.5` — both above their explicit limits) are in `paper/appendix.tex` §B and
`src/pinca_jax/equations/pdes.py`.

## Compute scope
Most published tables were produced on a **CPU-only host** ⇒ reduced-scale configs (small
grids, short horizons, few seeds) for an end-to-end, reproducible methodology demonstration.
Configs re-run unchanged on GPU — only the numeric fields change. `pmap`/sharding are
implemented but no-ops on one device. See `docs/environment.md`.

This is a real limitation, not a formality, and two of the pipeline's own checks exist to
keep it visible: `pinca_jax.claims` marks the GPU and seed-count claims UNSUPPORTED until a
full run exists, and `pinca_jax.scaling` reports whether the ranking even survives a change
of grid and horizon (on heat it currently does not, so no ranking here should be quoted
without its operating point).

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

## Metrics (every architecture, multi-seed, with intervals and paired tests)
Per-IC relative L2 with bootstrap CIs · MSE / RMSE / MAE / L∞ / PSNR / SSIM · high-frequency
error fraction · error-growth profile · per-channel mass conservation and its drift curve ·
periodic-BC residual · gradient energy · out-of-distribution degradation on seven held-out
axes · long-horizon failure rate with the divergence guard off · perturbation amplification ·
timestep sensitivity · rank stability under scale · parameter count, train wall-clock,
inference latency, throughput and peak memory.

Comparisons are **paired** (same seeds, same evaluation ICs, same order), tested with
Wilcoxon signed-rank and Holm–Bonferroni correction. `tie` is a reported outcome.
