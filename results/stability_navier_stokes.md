### Stability under stress - navier_stokes (grid=16, seeds=[42], n_ic=8)

Divergence guard **disabled** (`safety_factor=0`, no `output_clip`): a rollout that blows up is counted as a failure rather than clamped into a plausible-looking number. A run is failed once it is non-finite or its amplitude exceeds 10x the teacher's physical range.

| architecture | params | fail rate @ horizon | median survival (steps) | out-of-range cells | perturb amp (t=0) | perturb amp (mid) | rel-L2 @ dt/2 | rel-L2 @ 2dt |
|---|---|---|---|---|---|---|---|---|
| plain_nca | 6784 | 0% | 96/96 | 0.00% | 1.03x | 1.01x | 0.156 | 0.146 |
| pi_nca | 4576 | 0% | 96/96 | 0.00% | 0.932x | 0.944x | 0.163 | 0.142 |
| multiscale_flux_nca | 5520 | 0% | 96/96 | 0.00% | 1.05x | 0.982x | 0.177 | 0.12 |
| bounded_multiscale_nca | 5520 | 0% | 96/96 | 0.00% | 1.05x | 0.982x | 0.177 | 0.12 |
| fno | 592897 | 0% | 96/96 | 0.00% | 1x | 1x | 0.16 | 0.144 |
| resnet | 74336 | 0% | 96/96 | 0.00% | 1.01x | 0.998x | 0.182 | 0.113 |
| unet | 265104 | 0% | 96/96 | 0.00% | 1x | 0.998x | 0.186 | 0.094 |

**Failure rate vs horizon** (fraction of initial conditions failed).

| architecture | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| pi_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| multiscale_flux_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| bounded_multiscale_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| fno | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| resnet | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| unet | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

`rel-L2 @ dt/2` and `@ 2dt` re-score the emulator against a teacher run at a different timestep to the same physical time. The emulator has no dt input, so a large change here means the learned map encodes the training timestep rather than the operator.
