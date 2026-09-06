### Stability under stress - cahn_hilliard (grid=16, seeds=[42], n_ic=8)

Divergence guard **disabled** (`safety_factor=0`, no `output_clip`): a rollout that blows up is counted as a failure rather than clamped into a plausible-looking number. A run is failed once it is non-finite or its amplitude exceeds 10x the teacher's physical range.

| architecture | params | fail rate @ horizon | median survival (steps) | out-of-range cells | perturb amp (t=0) | perturb amp (mid) | rel-L2 @ dt/2 | rel-L2 @ 2dt |
|---|---|---|---|---|---|---|---|---|
| plain_nca | 6784 | 100% | 25/96 | 1.81% | 1.95x | 1.04x | 3.04 | 0.864 |
| pi_nca | 4576 | 100% | 24/96 | 1.86% | 2.51x | 1.14x | 3.12 | 0.85 |
| multiscale_flux_nca | 5520 | 100% | 23/96 | 1.86% | 2.01x | 1.02x | 4.05 | 0.862 |
| bounded_multiscale_nca | 5520 | 0% | 96/96 | 0.00% | 2.13x | 1.02x | 0.954 | 0.849 |
| fno | 592897 | 100% | 39/96 | 0.83% | 3.53x | 1.17x | 1.21 | 0.809 |
| resnet | 74336 | 100% | 24/96 | 1.81% | 2.16x | 1.09x | 3.38 | 0.84 |
| unet | 265104 | 100% | 27/96 | 1.42% | 1.24x | 0.793x | 2.28 | 0.878 |

**Failure rate vs horizon** (fraction of initial conditions failed).

| architecture | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_nca | 0% | 0% | 12% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| pi_nca | 0% | 0% | 25% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| multiscale_flux_nca | 0% | 0% | 75% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| bounded_multiscale_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| fno | 0% | 0% | 0% | 0% | 62% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| resnet | 0% | 0% | 25% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| unet | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

`rel-L2 @ dt/2` and `@ 2dt` re-score the emulator against a teacher run at a different timestep to the same physical time. The emulator has no dt input, so a large change here means the learned map encodes the training timestep rather than the operator.
