### Teacher (solver) error and cost - grid 48, horizon 48 steps

`vs_analytic` = rel-L2 against the closed-form periodic solution (linear constant-coefficient PDEs). `self_convergence` = rel-L2 between the production dt and a dt/8 reference over the same physical time; the observed order should match the scheme's design order.

| PDE | estimate | teacher rel-L2 | spatial | temporal | observed order | verdict | solver ms/step | fastest emulator ms/step | emulator vs solver |
|---|---|---|---|---|---|---|---|---|---|
| heat | vs_analytic | 1.143e-03 | 1.807e-03 | 6.956e-04 | - | converging | 1.640 | 3.855 (plain_nca) | **2x slower** |
| wave | vs_analytic | 6.144e-04 | 4.786e-04 | 8.200e-04 | - | converging | 3.786 | 4.227 (plain_nca) | **1x slower** |
| adv_diff | vs_analytic | 4.721e-03 | 4.800e-03 | 1.033e-03 | - | converging | 3.272 | 4.611 (plain_nca) | **1x slower** |
| allen_cahn | self_convergence | 1.653e-04 | - | - | 0.97 | converging | 3.634 | 6.438 (plain_nca) | **2x slower** |
| gray_scott | self_convergence | 4.971e-03 | - | - | 1.03 | converging | 1.700 | 1.262 (plain_nca) | 1.35x faster |
| shallow_water | self_convergence | 9.705e-07 | - | - | -0.52 | round-off limited | 4.918 | 4.719 (plain_nca) | 1.04x faster |
| cahn_hilliard | self_convergence | 4.407e-02 | - | - | 0.94 | converging | 3.081 | 6.275 (plain_nca) | **2x slower** |
| fitzhugh_nagumo | self_convergence | 6.012e-03 | - | - | 1.00 | converging | 4.060 | 6.781 (plain_nca) | **2x slower** |
| nagumo | self_convergence | 2.818e-03 | - | - | 1.00 | converging | 3.806 | 6.363 (plain_nca) | **2x slower** |
| navier_stokes | self_convergence | 2.507e-02 | - | - | 1.54 | converging | 1.148 | 1.276 (plain_nca) | **1x slower** |

**How to read an emulator's rel-L2 against this table.** An emulator error far above the teacher error is limited by learning: the target is not the binding constraint and architecture comparisons are meaningful. An emulator error approaching the teacher error has saturated the target, and further ranking there measures the teacher's own discretisation error rather than model quality.

`round-off limited` means the Cauchy differences reached machine precision, so the order estimate is undefined -- the teacher is fine, the estimator is not. `NOT CONVERGING` means refining the timestep did not reduce the difference at all.

> **The reference solver is faster than every emulator on: heat, wave, adv_diff, allen_cahn, cahn_hilliard, fitzhugh_nagumo, nagumo, navier_stokes.** At this grid size a learned surrogate is not a deployment case on these equations; it is a measurement harness. Surrogates pay off where the solver does not fit -- much larger grids, much longer horizons, stiff timestep restrictions, or differentiability through an entire rollout.
