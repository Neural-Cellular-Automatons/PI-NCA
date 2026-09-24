# Review experiments: what each one asks, and how to run it

Seven experiments, added in response to a review of the manuscript. Each answers one
question that the paper's current evidence cannot. None changes the existing pipeline:
they are opt-in stages, and the default `run_paper.sh` costs exactly what it did before.

```bash
# everything below, in one command, on a GPU box
bash run_paper.sh --only fluxhead,declared,tuning,dispersion,scaling_wide,stability,ood
```

Individually:

```bash
python -m pinca_jax.bench_all --archs "$(python - <<'P'
from pinca_jax.runner import FLUX_PROBE_ARCHS; print(FLUX_PROBE_ARCHS)
P
)" --pdes heat,cahn_hilliard,navier_stokes --tag probe --seeds 5 --epochs 1200 --grid 48
python -m pinca_jax.bench_all --archs mc_flux_nca,bounded_mc_nca --tag declared --seeds 5
python -m pinca_jax.tuning --pde cahn_hilliard          # and --pde shallow_water
python -m pinca_jax.dispersion                          # no training; seconds
python -m pinca_jax.scaling --pde heat --archs ... --grids 48,96 --seeds 3 --tag wide
python -m pinca_jax.stability --pde heat --grid 48 --seeds 1
python -m pinca_jax.ood --pde heat --grid 48
```

---

## 1. Which part of PI-NCA does the work: the automaton, or the flux head?

PI-NCA changes two things at once relative to an ordinary surrogate. It is a single shared
local rule *and* its head predicts a flux whose divergence advances the state. The head
is now available on three other backbones, so the two can be separated:

| model | backbone | head | params (C=1) |
|---|---|---|---|
| `fno` / `fno_flux` | global spectral | state / **flux** | 592,897 / 592,922 |
| `unet_iso` / `unet_iso_flux` | multi-resolution | state / **flux** | 7,464 / 7,468 |
| `resnet_iso` / `resnet_iso_flux` | local, per-layer weights | state / **flux** | 5,364 / 5,376 |

The flux head costs under 1% of the parameters in every case, so each pair isolates the
head alone. `tests/test_flux_probes.py` asserts that every flux variant conserves each
field for arbitrary weights and that every non-flux sibling does not, because otherwise
the pair would not be measuring what it claims.

## 2. FINN: the published conservative baseline

[`models/finn.py`](../src/pinca_jax/models/finn.py) implements a FINN-style facewise flux
network \[Praditia et al. 2021; Karlbauer et al. 2022\]. The difference from PI-NCA is
where the learned function sits: PI-NCA learns a **cellwise** rule that emits a cell's
outgoing flux, FINN learns a **facewise** rule applied to the pair of states across each
face, splitting the flux into a diffusive part proportional to the jump and an advective
part carried by the upwind cell. Both are conservative by the same telescoping argument,
so this is a like-for-like baseline rather than an unconstrained one.

Caveats recorded up front: this is a single re-usable module applied unchanged to every
phenomenon, not a reproduction of the authors' equation-specific models, so it is labelled
FINN-*style*. It is parameter-matched to PI-NCA (4,672 against 4,576) but **not**
compute-matched — it evaluates four small MLPs per step against PI-NCA's one — so a fair
reading needs the latency column as well as the parameter column.

## 3. Lumped versus per-field conservation

The per-field proposition argues that conserving one lumped total is the wrong constraint
for a multi-field system. `pi_nca_lumped` is that wrong constraint, built deliberately:
the same trunk, an unconstrained head, and a projection that restores the sum over all
fields and cells. It conserves the lumped total to round-off and lets individual fields
drift, which is the property the proposition is about, and it is parameter-matched to
PI-NCA to within 2%.

## 4. A pre-declared configuration rule

The manuscript's summary table reported, per phenomenon, the best of several PI-NCA
configurations against single-configuration baselines. That is a selection over variants.
The rule is now fixed in advance and applied everywhere:

> **Width:** the wide setting. **Projection:** enabled if and only if the equation's state
> has a hard physical range (Allen--Cahn, Cahn--Hilliard).

`bounded_mc_nca` exists because that rule needs the wide-and-bounded cell, which the
original run never trained. `paper_updated/pinca_tests.py` prints the resulting rank on
every phenomenon under both the wide and the compact rule, so the cost of committing to
one is visible rather than hidden.

## 5. Are the large baselines under-trained?

On Cahn--Hilliard the 592,897-parameter FNO scores worse than a 6,784-parameter
unconstrained NCA under the shared recipe, which is a reason to doubt the recipe rather
than the FNO. [`tuning.py`](../src/pinca_jax/tuning.py) sweeps three learning rates and
then trains the best setting for four times as long, for the large baselines *and* the
PI-NCA models, and records the training-loss curve plus a `still-improving` ratio (middle
tenth of training over the last tenth) so convergence is shown rather than asserted.

## 6. Rank stability including the actual runner-up

The paper's sweep compares five models and excludes the full-size ResNet and U-Net — and
on Cahn--Hilliard the full ResNet is the runner-up. `scaling_wide` includes them, uses
three seeds, and extends the grid axis past the benchmark's own resolution. It writes
`results/scaling_<pde>_wide.json`, so the sweep the paper currently reports is untouched.

## 7. Long-horizon stability and out-of-distribution transfer, at scale

Both studies failed in the GPU run and the paper reports them as not measured. The cause
was mundane: neither driver had the out-of-memory backoff that the accuracy matrix has,
so a single allocation failure killed the stage. Both now retry at half the batch
(`bench.run_with_oom_backoff`), and both complete at grid 48 locally. Until they are rerun
on the GPU the paper's claim about the flux form acting as a stabiliser has to stay
restricted to the comparison against the unconstrained NCA, since the only measured proxy
is the error-growth ratio inside the guarded protocol.

---

## Numerical dispersion of the wave solver — already measured

This one needs no training and is finished: `results/dispersion_wave.{json,md}`.

The continuum 2-D wave equation is exactly non-dispersive. The scheme the benchmark
distils from is not, and the measurement matches the closed-form discrete relation at
every mode, so the error is the scheme's rather than the fit's:

| mode | \|k\| | ω exact | ω discrete (closed form) | ω measured | phase error |
|---|---|---|---|---|---|
| (1,0) | 0.131 | 0.0654 | 0.0654 | 0.0654 | −0.1% |
| (4,0) | 0.524 | 0.2618 | 0.2588 | 0.2588 | −1.1% |
| (8,0) | 1.047 | 0.5236 | 0.5000 | 0.5000 | −4.5% |
| (16,0) | 2.094 | 1.0472 | 0.8661 | 0.8661 | −17.3% |
| (24,0) | 3.142 | 1.5708 | 1.0001 | 1.0001 | −36.3% |
| (24,24) | 4.443 | 2.2214 | 1.4145 | 1.4145 | −36.3% |

The semi-discrete and fully discrete frequencies agree to four decimals at this timestep,
so essentially all of the dispersion comes from the 5-point Laplacian rather than from the
time integrator. Short waves travel too slowly: at the grid scale the phase speed is 0.32
against the exact 0.5.

The defensible description of the phenomenon is therefore "non-dispersive continuum,
dispersive discretisation", and the dispersion is strongest exactly where a smooth-biased
model has least to lose.

---

## Reduced-scale screen (CPU, not comparable to the GPU numbers)

To have an answer before the GPU run, the flux-head and conservation controls were also
trained at grid 24 for 250 epochs with two seeds on the CPU, written under the tag
`probe_cpu24`. These are **indicative only** — a fifth of the grid area, a fifth of the
training, and a different backend from every number in the paper — and they must not be
mixed into any table. What they suggest is recorded here so the GPU run can confirm or
overturn it.

See `results/bench_*_probe_cpu24.md` for the tables as they stand.
