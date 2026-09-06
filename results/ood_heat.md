### OOD generalisation - heat (grid=16, eval_steps=12, seeds=[42])

rel-L2 at the end of the rollout, pooled over held-out ICs (mean [95% bootstrap CI]); `div` = fraction of ICs whose rel-L2 exceeds 1 (worse than predicting nothing).

| axis | plain_nca | identity |
|---|---|---|
| in_dist | 0.169 [0.16,0.19] | 0.382 [0.36,0.41] |
| phase | 0.169 [0.16,0.19] | 0.382 [0.36,0.41] |
| amp_half | 0.179 [0.16,0.2] | 0.382 [0.36,0.41] |
| amp_2x | 0.161 [0.15,0.17] | 0.337 [0.3,0.37] |
| amp_4x | 0.268 [0.19,0.35] | 0.265 [0.21,0.33] |
| blobs_few | 0.189 [0.16,0.21] | 0.405 [0.38,0.43] |
| blobs_many | 0.131 [0.12,0.14] | 0.294 [0.27,0.31] |
| scale_narrow | 1.28 [1.2,1.4] div=83% | 1.21 [1.1,1.3] div=100% |
| scale_wide | 0.104 [0.096,0.11] | 0.0871 [0.082,0.093] |
| spectrum_rough | 0.202 [0.19,0.22] | 0.395 [0.37,0.42] |
| coeff_half | 0.101 [0.095,0.11] | 0.204 [0.19,0.21] |
| coeff_2x | 0.408 [0.36,0.45] | 0.666 [0.61,0.72] |
| horizon_2x | 0.214 [0.19,0.24] | 0.657 [0.6,0.71] |
| horizon_4x | 0.213 [0.2,0.22] | 1.02 [0.92,1.1] div=67% |

**Degradation ratio** (axis rel-L2 / in-dist rel-L2); 1.0 = perfect transfer.

| axis | plain_nca | identity |
|---|---|---|
| phase | 1.00x | 1.00x |
| amp_half | 1.06x | 1.00x |
| amp_2x | 0.95x | 0.88x |
| amp_4x | 1.59x | 0.69x |
| blobs_few | 1.12x | 1.06x |
| blobs_many | 0.77x | 0.77x |
| scale_narrow | 7.56x | 3.17x |
| scale_wide | 0.61x | 0.23x |
| spectrum_rough | 1.19x | 1.03x |
| coeff_half | 0.60x | 0.53x |
| coeff_2x | 2.41x | 1.75x |
| horizon_2x | 1.26x | 1.72x |
| horizon_4x | 1.26x | 2.66x |
