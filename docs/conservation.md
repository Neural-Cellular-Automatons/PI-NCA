# Conservation: three different things that get called the same word

"The model conserves mass" is used in this literature for at least three claims of very
different strength, and a paper that does not distinguish them can report the weakest one
while sounding like it made the strongest. This document defines the three, says which
architecture here provides which, and names the metric that distinguishes them, so that
every conservation claim in the results tables can be traced to a specific mechanism.

## The three kinds

### 1. Structural conservation (exact, by construction)

The update is parameterised so that the conserved quantity cannot change, whatever the
weights are. In this repository that is the flux-divergence update
(`physics.divergence_flux_update`, `physics.multichannel_divergence_update`): the network
emits a flux field $f$ and the state increment is its discrete divergence,

```
u'_c = u_c + (roll(fx_c, +1, W) - fx_c) + (roll(fy_c, +1, H) - fy_c)
```

On a periodic grid this telescopes to exactly zero when summed, so `sum(u') == sum(u)` per
channel up to floating-point error, for **any** weights, at initialisation, mid-training,
and on out-of-distribution inputs. It cannot be broken by bad training.

Provided by: `pi_nca`, `mc_flux_nca`, `multiscale_flux_nca`, `abl_flux`, and the local
stream of `spectral_flux_nca`.

Caveat that must be stated: this conserves the quantity of the **discretised** system on a
**periodic** domain. It says nothing about the continuum conservation law, and it does not
survive a non-periodic boundary without an explicit boundary-flux term, which this
repository does not implement.

### 2. Projection (exact after the fact, by correction)

The update is unconstrained, and the total is restored afterwards by adding a correction.
Two variants here, and the difference between them matters:

* `conserve_energy_per_channel` — adds the same offset $\delta = (M_{\text{target}} - \sum
  u)/N$ to every cell. Exact in mass, but it moves every cell, and if the state was just
  clipped to a physical range it moves clipped cells straight back outside that range.
* `conserve_energy_bounded` — distributes the deficit in proportion to each cell's
  remaining headroom to the bound. Exact in mass **and** inside the bound whenever the
  target is feasible for the box; when it is not, the field saturates and
  `projection_residual` reports the mass error rather than the bound being violated.

Projection is strictly weaker than structural conservation: it fixes the global total
without any claim about *where* the mass went, so a model can move mass across the domain
arbitrarily and still pass. It is also applied at inference, so it must be applied
identically in training, validation and test — a mismatch there is exactly the defect
documented in [`legacy_pytorch.md`](legacy_pytorch.md), where the original script's
validation loop omitted a projection that training and the final test both applied.

Provided by: `bounded_cons_nca`, `bounded_multiscale_nca`, and the global stream of
`spectral_flux_nca` (`conserve=True`).

### 3. Empirical conservation (measured, not guaranteed)

The model has no conservation mechanism at all and the mass error simply happens to be
small on the evaluation set. This is the weakest claim and the easiest to mistake for one
of the others, because on a short rollout of a smooth field an unconstrained residual
model can show a mass error that looks like round-off.

The way to tell them apart is not the end-of-rollout number but the **drift curve**:
structural conservation is flat in time at machine precision, projection is flat because it
is corrected every step, and empirical conservation drifts. `metrics.conservation_drift`
returns the whole curve for exactly this reason, and `conservation_drift_max` /
`conservation_drift_final` are reported per cell in the benchmark JSON.

Provided by: nothing, by design. `plain_nca`, `abl_residual`, `resnet`, `unet` and `fno`
have no conservation structure, and their drift curves are what the conservation columns in
the results tables are measuring.

## Which metric answers which question

| Question | Metric | Where |
|---|---|---|
| Did the total change over the rollout? | `conservation_err` | every bench table |
| Did each field's total change, or only the lumped sum? | `conservation_error_per_channel` | `_per_channel_cons_err` in bench JSON |
| Is it flat by construction, corrected, or drifting? | `conservation_drift` (whole curve) | `conservation_drift` in bench JSON |
| Did the projection actually restore the mass? | `projection_residual` | `physics.py`, used by the bounded models |
| Did enforcing the bound break the bound? | out-of-range cell fraction | `results/stability_*.md` |

**Per-channel is not optional for multi-field states.** `conservation_error` sums over all
channels, so a model that adds mass to one field and removes the same amount from another
scores a perfect zero. `tests/test_rigor.py::test_conservation_error_per_channel_is_not_lumped`
constructs exactly that state and asserts the lumped metric misses it while the per-channel
one catches it. Every conservation claim about shallow water, Gray–Scott or FitzHugh–Nagumo
must therefore be read from the per-channel column.

## The tension with boundedness, and how it is resolved

Stiff bounded fields (Cahn–Hilliard, Allen–Cahn) need the state clipped to its physical
range to remain stable, and conservative models need mass restored after the clip. Composing
the obvious versions of the two gives a model that is neither: the clip enforces the bound,
and the uniform projection immediately breaks it again. The size of the violation is not
negligible — measured on Cahn–Hilliard with the divergence guard disabled it is a few
percent of cells, in the one model family whose stated property is being simultaneously
bounded and conserving.

`conserve_energy_bounded` resolves it by distributing the deficit in proportion to headroom
rather than uniformly. Mass is restored exactly, no cell can cross the bound because the
total increment is capped by the total headroom by construction, the map is differentiable,
and it is a single vectorised pass. Ablation **A7** measures the cost of the naive choice
against it (`abl_proj_none` / `abl_proj_uniform` / `abl_proj_headroom`) rather than
asserting the fix is free.

Feasibility is the honest limit: a target mass outside $[N\ell, Nh]$ cannot be reached
inside the box at all, and in that case the projection saturates and reports a nonzero
`projection_residual`. That is the correct behaviour — the alternative is to silently
violate the bound — but it does mean the "bounded AND conserving" claim is conditional on
feasibility, and models are evaluated with that residual recorded rather than assumed zero.

## What is deliberately not claimed

* **No claim about the continuum.** All conservation here is of the discretised quantity on
  a periodic grid.
* **No claim about momentum or energy.** Only mass (the channel sum) is conserved
  structurally. Shallow water conserves its three channel sums separately, which is the
  right discrete statement for that system on a periodic domain, but the model has no
  structural guarantee about the physical momentum or total energy.
* **No claim that conservation implies accuracy.** Ablation A4 exists precisely because it
  does not: it switches the conservation head on and off at matched backbone width, and on
  non-conservative dynamics (Nagumo, FitzHugh–Nagumo) imposing conservation is the wrong
  prior and costs accuracy. That is a finding, and it is reported as one.
