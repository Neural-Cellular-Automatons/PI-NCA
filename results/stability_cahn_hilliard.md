### Stability under stress - cahn_hilliard (grid=16, seeds=[42], n_ic=6)

Divergence guard **disabled** (`safety_factor=0`, no `output_clip`): a rollout that blows up is counted as a failure rather than clamped into a plausible-looking number. A run is failed once it is non-finite or its amplitude exceeds 10x the teacher's physical range.

| architecture | params | fail rate @ horizon | median survival (steps) | out-of-range cells | perturb amp (t=0) | perturb amp (mid) | rel-L2 @ dt/2 | rel-L2 @ 2dt |
|---|---|---|---|---|---|---|---|---|
| plain_nca | 6784 | 100% | 25/48 | 1.43% | 2.06x | 1.07x | 2.85 | 0.867 |
| bounded_multiscale_nca | 5520 | 0% | 48/48 | 4.36% | 2.22x | 1.04x | 0.966 | 0.853 |
| identity | 1 | 0% | 48/48 | 0.00% | 1x | 1x | 0.932 | 0.935 |

**Failure rate vs horizon** (fraction of initial conditions failed).

| architecture | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|
| plain_nca | 0% | 0% | 0% | 100% | 100% | 100% |
| bounded_multiscale_nca | 0% | 0% | 0% | 0% | 0% | 0% |
| identity | 0% | 0% | 0% | 0% | 0% | 0% |

`rel-L2 @ dt/2` and `@ 2dt` re-score the emulator against a teacher run at a different timestep to the same physical time. The emulator has no dt input, so a large change here means the learned map encodes the training timestep rather than the operator.
