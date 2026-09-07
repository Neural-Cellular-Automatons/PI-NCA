# PI-NCA: which inductive bias matches which PDE regime?

Neural cellular automata, physics-informed variants, neural operators, and ordinary CNN
surrogates, compared under one harness on ten 2-D and six 3-D PDE families.

The question is **not** whether NCAs beat PINNs. It is **which structural prior pays off
in which regime**, and the answer this repository supports is that there is no universal
winner. Locality plus a conservative flux update wins where the dynamics are local and
mass-conserving; global spectral mixing wins where the solution has sharp interfaces or
long-range coupling; and on at least one equation a physics-free CNN beats every model
with a physics prior. Which of those hold at which scale is itself measured -- the ranking
is not stable under changes of grid and horizon, so it is always quoted with its operating
point. Every statement above is produced by a script here and checked against the raw
results by `python -m pinca_jax.claims`.

## One command

On a machine with an NVIDIA GPU (Linux or WSL2 — JAX has no native-Windows CUDA):

```bash
bash run_paper.sh
```

That produces **every number, table and figure in the paper**: the correctness gate, the
2-D matrix, the ablations, the multi-seed headline comparison, the teacher-error study,
the out-of-distribution study, the stability stress test, the scaling study, the 3-D
matrix, the resolution study, the baselines, the field figures, the plots, the claims
audit, the verified bibliography, the generated LaTeX tables, and the paper PDF if a LaTeX
toolchain is installed. Nothing else needs running afterwards.

Find out what it costs on your card first — it trains two real cells and projects:

```bash
bash run_paper.sh --estimate
```

It is safe to interrupt. Benchmarks checkpoint per (PDE, architecture) cell and whole
stages are skipped when their outputs already exist, so re-running the same command after
a crash, a Ctrl-C or a reboot continues instead of starting over. `--force` recomputes.
`--list-stages` prints the stage names that `--only` and `--skip` accept.

Windows without WSL2, or no GPU: `run_paper.bat`, and add `--allow-cpu` to get a run whose
numbers are explicitly not comparable with GPU numbers.

## Setup

```bash
python -m pip install -r requirements-gpu.txt   # GPU (CUDA build of JAX); or requirements-jax.txt for CPU
python -m pytest tests/ -q                      # correctness gate; must be green first
```

`pip install -e .` is optional — the launchers put `src/` on `PYTHONPATH` themselves, so a
fresh clone runs without it. `bash setup_gpu.sh` does the whole setup; see
[docs/gpu_runbook.md](docs/gpu_runbook.md).

A wiring check that touches every stage and every phenomenon, whose numbers are
meaningless by design (~20 minutes on a laptop CPU, a few on a GPU):

```bash
bash run_paper.sh --profile smoke --allow-cpu
```

## What is measured

| Stage | Module | Output |
|---|---|---|
| Correctness gate | `tests/` | JAX ports asserted equal to the verbatim PyTorch references |
| 2-D matrix | `pinca_jax.bench_all` | every architecture x every phenomenon, mean +/- std + bootstrap CIs + Holm-corrected paired tests |
| Headline comparison | `bench_all --tag headline` | 10 seeds on the three regime representatives |
| Ablations A1-A7 | `bench_all --group ablation` | conservation, receptive field, spectral mixing, bounds, projection |
| OOD generalisation | `pinca_jax.ood` | held-out amplitude / structure count / length scale / spectrum / phase / PDE coefficient / horizon |
| Stability stress | `pinca_jax.stability` | failure-rate curves with the divergence guard **disabled** |
| Teacher error | `pinca_jax.teacher_error` | the solver's own discretisation error, split spatial vs temporal, plus a convergence verdict per equation |
| Scaling / rank stability | `pinca_jax.scaling` | does the ranking survive a change of grid, horizon and training budget? |
| Matched PINN comparison | `pinca_jax.matched` | same equation, same ICs, same horizon; both cost structures and the amortisation crossover |
| Resolution transfer | `pinca_jax.res_study` | train at one grid, evaluate at others |
| 3-D matrix | `pinca_jax.bench3d` | the same comparison in NDHWC |
| Continuous baselines | `pinn_heat`, `deeponet_heat`, `darcy` | PINN / DeepONet / steady operator |
| Claims audit | `pinca_jax.claims` | every count and headline claim checked against `results/` |

## Reading the results honestly

Five things in this repository exist specifically to stop a favourable-looking number from
being believed. Each has already changed a conclusion:

* **An identity floor.** `identity` (g(x) = x) is in every comparison. A model above it
  learned something worse than nothing, and the tables say so.
* **Physics-free controls.** A residual CNN and a U-Net, at full size and at
  parameter-matched size. Without them, "the conservation prior helps" is confounded
  with "we never tried an ordinary CNN".
* **Paired statistics.** Architectures are compared on the *same* initial conditions with
  Wilcoxon signed-rank tests and Holm-Bonferroni correction. `tie` in a results table
  means the difference is not resolvable at that sample size -- it is reported, not
  rounded away.
* **The teacher is checked before it is trusted.** `teacher_error` refines the timestep and
  reports an observed order of accuracy. A solver that does not converge is not a target,
  and the generated table says so rather than printing a winner. This caught a real one:
  the Cahn-Hilliard reference shipped a timestep above its fourth-order stability limit and
  was held together by a clip, giving an observed order of zero. Every ranking distilled
  from it was ranking solver noise, and the finding it produced is retracted in
  `docs/research_log.md`.
* **The guard is switched off for stability claims.** `EmuConfig.safety_factor` clamps
  rollouts so a diverging model cannot emit a meaningless metric. That is right for
  accuracy tables and wrong for stability tables, so `pinca_jax.stability` disables it
  and counts failures instead.

Background reading: [`docs/related_work.md`](docs/related_work.md) for what is and is not
new here, [`docs/conservation.md`](docs/conservation.md) for the three different things the
word "conservation" is used for, and [`docs/research_log.md`](docs/research_log.md) for the
trail including the retractions.

`docs/claims_audit.md` is generated, not written: it lists what was measured, which
pre-registered claims are currently SUPPORTED / UNSUPPORTED / NOT_YET_MEASURED, and any
count in the prose that disagrees with `results/`.

## Layout

```
src/pinca_jax/          the implementation (this is the canonical code)
  equations/            reference PDE solvers (the teachers) + finite-difference operators
  models/               NCA, flux NCA, FNO, hybrids, CNN/U-Net baselines, ablation probes
  harness.py            train + evaluate any emulator; the shared protocol
  stats.py              bootstrap CIs, paired tests, multiple-comparison control
  ood.py stability.py teacher_error.py   the studies that test the claims
  scaling.py matched.py                  rank stability; the matched PINN comparison
  claims.py bib.py paper.py              generate the audit, the bibliography, the paper
  runner.py             one command that runs the lot
tests/                  the correctness gate (205 tests)
results/                raw JSON + generated Markdown tables
docs/                   reports, figures, runbook, paper source
paper/                  the submission source; every table and number generated
PI NCA_v1.py            LEGACY. The original PyTorch script, kept for provenance only.
```

`PI NCA_v1.py` is not part of the pipeline and is not maintained; see
[docs/legacy_pytorch.md](docs/legacy_pytorch.md) for its known defects and why the JAX
path replaced it.

## Reproducing a specific number

Every results JSON carries a `device` stamp (JAX version, backend, devices, peak memory)
and the exact `config` used. `docs/reproducibility.md` maps each table in the paper to the
command that produces it. Run `python -m pinca_jax.claims --strict` to fail loudly if the
documents and the results have drifted apart.

## Status

`docs/claims_audit.md` is the live version and is generated, so it cannot flatter. The
honest summary: the machinery and the breadth are in place; the released numbers are
reduced-scale and CPU-only, and the audit correctly reports the seed count, the backend and
rank stability as not yet supported. The remaining gap to a submission is compute, not
method -- `bash run_paper.sh` on a GPU produces every number the audit is currently
waiting on.

## License

MIT. See [LICENSE](LICENSE).
