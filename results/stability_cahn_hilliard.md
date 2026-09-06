### Stability under stress - cahn_hilliard (grid=16, seeds=[42], n_ic=8)

Divergence guard **disabled** (`safety_factor=0`, no `output_clip`): a rollout that blows up is counted as a failure rather than clamped into a plausible-looking number. A run is failed once it is non-finite or its amplitude exceeds 10x the teacher's physical range.

| architecture | params | fail rate @ horizon | median survival (steps) | out-of-range cells | perturb amp (t=0) | perturb amp (mid) | rel-L2 @ dt/2 | rel-L2 @ 2dt |
|---|---|---|---|---|---|---|---|---|
| plain_nca | 6784 | 100% | 58/96 | 0.00% | 1.82x | 1.25x | 0.894 | 0.228 |
| pi_nca | 4576 | 100% | 62/96 | 0.00% | 1.8x | 1.24x | 0.844 | 0.227 |
| multiscale_flux_nca | 5520 | 100% | 60/96 | 0.00% | 1.8x | 1.24x | 0.887 | 0.226 |
| bounded_multiscale_nca | 5520 | 0% | 96/96 | 0.00% | 1.8x | 1.24x | 0.887 | 0.226 |
| abl_proj_uniform | 5520 | 0% | 96/96 | 0.00% | 1.8x | 1.24x | 0.887 | 0.226 |
| abl_proj_headroom | 5520 | 0% | 96/96 | 0.00% | 1.8x | 1.24x | 0.887 | 0.226 |
| fno | 592897 | 100% | 73/96 | 0.00% | 1.79x | 1.26x | 0.82 | 0.227 |
| resnet | 74336 | 100% | 66/96 | 0.00% | 1.79x | 1.26x | 0.856 | 0.227 |
| unet | 265104 | 100% | 69/96 | 0.00% | 1.78x | 1.26x | 0.934 | 0.229 |
| identity | 1 | 0% | 96/96 | 0.00% | 1x | 1x | 0.41 | 0.388 |

**Failure rate vs horizon** (fraction of initial conditions failed).

| architecture | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| pi_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 88% | 100% | 100% | 100% | 100% |
| multiscale_flux_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100% | 100% | 100% | 100% | 100% |
| bounded_multiscale_nca | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| abl_proj_uniform | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| abl_proj_headroom | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| fno | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 25% | 100% | 100% | 100% |
| resnet | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 25% | 100% | 100% | 100% | 100% |
| unet | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 25% | 88% | 100% | 100% | 100% |
| identity | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

`rel-L2 @ dt/2` and `@ 2dt` re-score the emulator against a teacher run at a different timestep to the same physical time. The emulator has no dt input, so a large change here means the learned map encodes the training timestep rather than the operator.
