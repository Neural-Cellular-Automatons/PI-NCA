### Stability under stress - heat (grid=16, seeds=[42], n_ic=8)

Divergence guard **disabled** (`safety_factor=0`, no `output_clip`): a rollout that blows up is counted as a failure rather than clamped into a plausible-looking number. A run is failed once it is non-finite or its amplitude exceeds 10x the teacher's physical range.

| architecture | params | fail rate @ horizon | median survival (steps) | out-of-range cells | perturb amp (t=0) | perturb amp (mid) | rel-L2 @ dt/2 | rel-L2 @ 2dt |
|---|---|---|---|---|---|---|---|---|
| plain_nca | 6784 | 0% | 96/96 | 0.00% | 1.03x | 0.96x | 0.147 | 0.272 |
| pi_nca | 4576 | 0% | 96/96 | 0.00% | 1.4x | 1.06x | 0.181 | 0.21 |
| multiscale_flux_nca | 5520 | 0% | 96/96 | 0.00% | 1.13x | 1.01x | 0.214 | 0.205 |
| bounded_multiscale_nca | 5520 | 0% | 96/96 | 0.00% | 1.13x | 1.01x | 0.214 | 0.205 |
| fno | 592897 | 0% | 96/96 | 0.00% | 1.21x | 1.08x | 0.194 | 0.221 |
| resnet | 74336 | 0% | 96/96 | 0.00% | 1.19x | 1.07x | 0.192 | 0.215 |
| unet | 265104 | 0% | 96/96 | 0.00% | 1.19x | 1.08x | 0.186 | 0.232 |

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
