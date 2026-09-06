# Reproducibility Guide

Everything needed to reproduce every number. See also [`environment.md`](environment.md)
for exact package versions and the Windows long-path workaround, and
[`gpu_runbook.md`](gpu_runbook.md) for the GPU run.

## 0. The short version

```bash
python -m pip install -r requirements-gpu.txt   # CUDA build of JAX; requirements-jax.txt for CPU
bash run_paper.sh --estimate                    # what it costs on your card
bash run_paper.sh                               # every number, table and figure in the paper
```

That is the whole thing. `run_paper.sh` puts `src/` on `PYTHONPATH` itself, so
`pip install -e .` is optional. It refuses to start on the CPU backend without
`--allow-cpu`, because CPU and GPU numbers in one table are worse than no table.

It is safe to interrupt: benchmarks checkpoint per (PDE, architecture) cell, and whole
stages are skipped when their outputs already exist, so re-running the same command
continues rather than restarting. `--force` recomputes; `--list-stages` prints the names
`--only` and `--skip` accept.

A minutes-long wiring check that touches every stage and every phenomenon and whose
numbers are meaningless by design:

```bash
bash run_paper.sh --profile smoke --allow-cpu
```

## 1. Install

```bash
python -m pip install -r requirements-jax.txt     # CPU: Linux/macOS, or Windows w/ long paths
python -m pip install -r requirements-gpu.txt     # GPU: jax[cuda12] — see docs/gpu_runbook.md
python -m pip install -e .                        # src layout -> `python -m pinca_jax.*`
```

Verified stack: jax 0.10.1, flax 0.12.7, optax 0.2.8, numpy, scipy, matplotlib, torch
(CPU, reference only — used by the migration tests, never by the pipeline).

## 2. Correctness gate (run this first)

```bash
python -m pytest tests/ -q
```

205 tests. It asserts the JAX ports equal the verbatim PyTorch references to tolerance, the
NHWC isotropy fix, structural conservation, the Gray–Scott `dt=2.0` and Cahn–Hilliard
`dt=0.5` instabilities, the statistical machinery (bootstrap coverage, Holm correction,
paired tests), the closed-form solver references, the bounded mass projection, and the
paper's structural consistency. **A benchmark run on a red gate is not evidence of
anything.**

## 3. The pipeline, stage by stage

`bash run_paper.sh` runs all of it in order. Each stage is also runnable alone
(`bash run_paper.sh --only bench2d`, or the module directly):

| Stage | Command | Writes |
|---|---|---|
| 2-D matrix | `python -m pinca_jax.bench_all --group all --seeds 5` | `results/bench_<pde>_full.{json,md}` |
| Headline (high seed count) | `... --pdes heat,cahn_hilliard,navier_stokes --seeds 10 --tag headline` | `results/bench_<pde>_headline.*` |
| Ablations A1/A4/A5/A7 | `python -m pinca_jax.bench_all --group ablation --seeds 5` | `results/bench_<pde>_A*.{json,md}` |
| Teacher error | `python -m pinca_jax.teacher_error` | `results/teacher_error.{json,md}` |
| OOD generalisation | `python -m pinca_jax.ood --pde heat --seeds 3` | `results/ood_<pde>.{json,md}` |
| Stability (guard OFF) | `python -m pinca_jax.stability --pde heat --seeds 3` | `results/stability_<pde>.{json,md}` |
| Scaling / rank stability | `python -m pinca_jax.scaling --pde heat` | `results/scaling_<pde>.{json,md}` |
| Matched PINN comparison | `python -m pinca_jax.matched --pde heat --k 32` | `results/matched_heat.{json,md}` |
| Resolution transfer | `python -m pinca_jax.res_study --pdes heat,allen_cahn,navier_stokes` | `results/bench_resolution_*.{json,md}` |
| 3-D matrix | `python -m pinca_jax.bench3d --grid 32` | `results/bench3d_<pde>.{json,md}` |
| Continuous baselines | `python -m pinca_jax.pinn_heat`, `.deeponet_heat`, `.darcy` | stdout + `results/` |
| Claims audit | `python -m pinca_jax.claims` | `docs/claims_audit.md` |
| Bibliography | `python -m pinca_jax.bib` | `docs/bibliography.{md,json}`, `paper/refs.bib` |
| Paper tables | `python -m pinca_jax.paper` | `paper/generated/*.tex` |
| Figures / plots / report | `python -m pinca_jax.capture`, `.viz`, `.plots`, `.report` | `docs/figures/`, `docs/*.md` |

## 4. Determinism and seeds

* All randomness flows from an explicit `jax.random.PRNGKey`. No global RNG anywhere in
  `src/pinca_jax/`.
* Every architecture in a comparison sees the **same seeds and the same evaluation initial
  conditions in the same order**. That is what makes the paired Wilcoxon tests legitimate;
  an unpaired comparison of two independent means would be the wrong test.
* Bootstrap resampling is seeded, so a reported confidence interval is reproducible from
  the same raw numbers.
* XLA reductions are not bitwise-deterministic across thread counts, so correctness is
  asserted to tolerances rather than bit-equality.

## 5. Reading the results honestly

Five things exist specifically to stop a favourable number from being believed:

1. **The identity floor.** `identity` (g(x)=x) is in every comparison. A model above it
   learned worse than nothing, and the tables say so rather than ranking only the
   survivors.
2. **Physics-free controls.** `resnet`, `unet`, and both at NCA-matched parameter counts.
   Without them a conservation result is confounded with "an ordinary CNN was never tried".
3. **Paired statistics.** Bootstrap CIs, Wilcoxon signed-rank on shared initial conditions,
   Holm–Bonferroni across the architectures compared per equation. `tie` means *not
   resolvable at this sample size*, not "equal".
4. **The teacher's own error.** `teacher_error` reports what the reference solver gets
   wrong, split into spatial and temporal truncation, and flags any solver that does not
   converge under timestep refinement. An architecture ranking on a non-converging teacher
   is ranking solver noise, and the generated table says so instead of printing a winner.
5. **Rank stability.** `scaling` reports Kendall's tau between the ordering at the smallest
   and largest grid, horizon and training budget. Where it is below 1, the headline ranking
   is conditional on its operating point and must be quoted with it.

### The divergence guard, and when it is off

`EmuConfig.safety_factor` (default 1.25) clamps every emulator rollout to the reference
solver's measured physical range (`harness.field_bounds`) widened by that factor. A healthy
model never touches it; a diverging one can no longer emit meaningless numbers such as a
negative PSNR (which only ever meant "MSE exceeded the signal range", i.e. blow-up). An
explicit `output_clip` still takes precedence, so the bounding ablation is unchanged, and a
failed model stays visibly failed — its rel-L2 sits at or above the identity floor.

**`pinca_jax.stability` disables it** (`safety_factor=0`, no `output_clip`). The guard is
right for accuracy tables and wrong for stability tables, because it turns a blow-up into a
saturated field and a merely-poor score. Every stability number is measured unguarded and
failures are counted.

## 6. Scale knobs (CPU ↔ GPU)

`EmuConfig` fields (`grid_size`, `rollout_steps`, `eval_steps`, `epochs`, `batch`, seeds)
are the only difference between the reduced-scale CPU presets and the full-scale GPU
presets; the code is identical. The runner's `--profile` sets them all at once.

Cost at the `paper` preset: ~1700 trainings — the 2-D matrix is 14 architectures × 10
phenomena × 5 seeds, the headline stage adds 14 × 3 × 10, and the ablations, OOD,
stability, scaling, 3-D, resolution and figure stages make up the rest. Do not guess how
long that takes on your hardware; `bash run_paper.sh --estimate` trains two real cells
(the cheapest and the most expensive architecture) and multiplies by the exact per-stage
cell counts. Every stage checkpoints, so an interrupted run resumes.

Every driver stamps `{jax, backend, devices, peak_mem_mb}` and the exact config into its
results JSON under `"device"`, so a GPU run can be told from a CPU one after the fact.

## 7. Verifying the documents against the results

```bash
python -m pinca_jax.claims --strict   # fails if prose and results/ disagree
python -m pinca_jax.bib --strict      # fails if any cited arXiv id does not resolve
python -m pytest tests/test_paper.py  # fails if the paper quotes an ungenerated number
```

`docs/claims_audit.md` is generated, never written by hand. It lists what was measured,
which pre-registered claims are SUPPORTED / UNSUPPORTED / NOT_YET_MEASURED, every failed
matrix cell, and any count in the prose that disagrees with `results/`.

## 8. Artifact map

| Artifact | Path |
|---|---|
| Paper source (ICLR-structured) | `paper/main.tex`, `paper/appendix.tex` |
| Paper tables and quoted numbers (generated) | `paper/generated/` (`pinca_jax.paper`) |
| Claims audit (generated) | `docs/claims_audit.md` (`pinca_jax.claims`) |
| Bibliography (arXiv-verified) | `docs/bibliography.md`, `paper/refs.bib` (`pinca_jax.bib`) |
| Related work and novelty positioning | `docs/related_work.md` |
| Conservation taxonomy | `docs/conservation.md` |
| Legacy PyTorch script defects | `docs/legacy_pytorch.md` |
| Literature review (background) | `docs/literature_review.md` |
| Migration reports | `docs/migration/` |
| Architecture report + diagrams | `docs/architecture_report.md`, `docs/architecture_diagrams.md` |
| Experimental + ablation reports | `docs/experimental_report.md`, `docs/ablation_report.md` |
| Benchmarks (raw) | `results/*.json`, `results/*.md` |
| Research log (trail) | `docs/research_log.md` |
| JAX core | `src/pinca_jax/` |
| Correctness gate | `tests/` |
| Benchmark plots | `docs/figures/bench/*.png` (`pinca_jax.plots`) |
| GPU runbook | `docs/gpu_runbook.md` + `run_gpu.sh` |
