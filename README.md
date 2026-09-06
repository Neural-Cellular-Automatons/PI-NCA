# PI-NCA: which inductive bias matches which PDE regime?

Neural cellular automata, physics-informed variants, neural operators, and ordinary CNN
surrogates, compared under one harness on ten 2-D and six 3-D PDE families.

The question is **not** whether NCAs beat PINNs. It is **which structural prior pays off
in which regime**, and the answer this repository supports is that there is no universal
winner: locality plus a conservative flux update wins where the dynamics are local and
mass-conserving, global spectral mixing wins where the solution has sharp interfaces or
long-range coupling, and on some phenomena nothing measured here beats a do-nothing
identity baseline by a statistically resolvable margin. Every one of those statements is
produced by a script in this repository and checked against the raw results.

## Quick start

```bash
python -m pip install -r requirements-jax.txt   # CPU
python -m pip install -e .
python -m pytest tests/ -q                      # correctness gate; must be green first
```

One command runs everything:

```bash
python -m pinca_jax.runner --profile smoke --allow-cpu   # ~minutes, wiring check only
```

```bash
python -m pinca_jax.runner --profile full                # GPU, overnight, the real numbers
```

`--profile smoke` exists to prove every stage is wired up; its numbers are meaningless.
Real numbers need a GPU, and the runner refuses to start on CPU without `--allow-cpu`
precisely so a CPU run cannot be mistaken for one. See [docs/gpu_runbook.md](docs/gpu_runbook.md).

## What is measured

| Stage | Module | Output |
|---|---|---|
| Correctness gate | `tests/` | JAX ports asserted equal to the verbatim PyTorch references |
| 2-D matrix | `pinca_jax.bench_all` | every architecture x every phenomenon, mean +/- std + bootstrap CIs + Holm-corrected paired tests |
| Headline comparison | `bench_all --tag headline` | 10 seeds on the three regime representatives |
| Ablations A1-A7 | `bench_all --group ablation` | conservation, receptive field, spectral mixing, bounds, projection |
| OOD generalisation | `pinca_jax.ood` | held-out amplitude / structure count / length scale / spectrum / phase / PDE coefficient / horizon |
| Stability stress | `pinca_jax.stability` | failure-rate curves with the divergence guard **disabled** |
| Teacher error | `pinca_jax.teacher_error` | the solver's own discretisation error, split spatial vs temporal, plus cost |
| Resolution transfer | `pinca_jax.res_study` | train at one grid, evaluate at others |
| 3-D matrix | `pinca_jax.bench3d` | the same comparison in NDHWC |
| Continuous baselines | `pinn_heat`, `deeponet_heat`, `darcy` | PINN / DeepONet / steady operator |
| Claims audit | `pinca_jax.claims` | every count and headline claim checked against `results/` |

## Reading the results honestly

Four things in this repository exist specifically to stop a favourable-looking number
from being believed:

* **An identity floor.** `identity` (g(x) = x) is in every comparison. A model above it
  learned something worse than nothing, and the tables say so.
* **Physics-free controls.** A residual CNN and a U-Net, at full size and at
  parameter-matched size. Without them, "the conservation prior helps" is confounded
  with "we never tried an ordinary CNN".
* **Paired statistics.** Architectures are compared on the *same* initial conditions with
  Wilcoxon signed-rank tests and Holm-Bonferroni correction. `tie` in a results table
  means the difference is not resolvable at that sample size -- it is reported, not
  rounded away.
* **The guard is switched off for stability claims.** `EmuConfig.safety_factor` clamps
  rollouts so a diverging model cannot emit a meaningless metric. That is right for
  accuracy tables and wrong for stability tables, so `pinca_jax.stability` disables it
  and counts failures instead.

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
  ood.py stability.py teacher_error.py   the three studies that test the claims
  runner.py             one command that runs the lot
tests/                  the correctness gate (181 tests)
results/                raw JSON + generated Markdown tables
docs/                   reports, figures, runbook, paper source
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

See `docs/claims_audit.md` for the live version. The honest summary: the machinery and the
breadth are in place, and the remaining gap to a submission is compute -- the multi-seed,
full-resolution GPU numbers -- not method.

## License

MIT. See [LICENSE](LICENSE).
