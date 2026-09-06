### Teacher (solver) error and cost - grid 16, horizon 12 steps

`vs_analytic` = rel-L2 against the closed-form periodic solution (linear constant-coefficient PDEs). `self_convergence` = rel-L2 between the production dt and a dt/8 reference over the same physical time; the observed order should match the scheme's design order.

| PDE | estimate | teacher rel-L2 | spatial | temporal | observed order | verdict | solver ms/step | fastest emulator ms/step | emulator vs solver |
|---|---|---|---|---|---|---|---|---|---|
| heat | vs_analytic | 1.344e-02 | 2.034e-02 | 7.198e-03 | - | converging | 0.040 | 0.885 (plain_nca) | **22x slower** |
| wave | vs_analytic | 5.680e-03 | 5.866e-03 | 1.635e-03 | - | converging | 0.025 | 0.617 (plain_nca) | **25x slower** |
| adv_diff | vs_analytic | 2.646e-02 | 2.727e-02 | 1.799e-03 | - | converging | 0.078 | 0.635 (plain_nca) | **8x slower** |
| allen_cahn | self_convergence | 8.000e-04 | - | - | 1.04 | converging | 0.035 | 0.946 (plain_nca) | **27x slower** |
| gray_scott | self_convergence | 1.122e-02 | - | - | 1.02 | converging | 0.108 | 0.879 (plain_nca) | **8x slower** |
| shallow_water | self_convergence | 3.466e-07 | - | - | -0.56 | round-off limited | 0.165 | 0.526 (plain_nca) | **3x slower** |
| cahn_hilliard | self_convergence | 2.287e-02 | - | - | 0.90 | converging | 0.090 | 0.723 (plain_nca) | **8x slower** |
| fitzhugh_nagumo | self_convergence | 1.850e-02 | - | - | 1.00 | converging | 0.053 | 0.772 (plain_nca) | **14x slower** |
| nagumo | self_convergence | 1.025e-03 | - | - | 0.99 | converging | 0.015 | 0.777 (plain_nca) | **50x slower** |
| navier_stokes | self_convergence | 1.037e-02 | - | - | 1.15 | converging | 0.249 | 0.792 (plain_nca) | **3x slower** |

**How to read an emulator's rel-L2 against this table.** An emulator error far above the teacher error is limited by learning: the target is not the binding constraint and architecture comparisons are meaningful. An emulator error approaching the teacher error has saturated the target, and further ranking there measures the teacher's own discretisation error rather than model quality.

`round-off limited` means the Cauchy differences reached machine precision, so the order estimate is undefined -- the teacher is fine, the estimator is not. `NOT CONVERGING` means refining the timestep did not reduce the difference at all.

> **The reference solver is faster than every emulator on: heat, wave, adv_diff, allen_cahn, gray_scott, shallow_water, cahn_hilliard, fitzhugh_nagumo, nagumo, navier_stokes.** At this grid size a learned surrogate is not a deployment case on these equations; it is a measurement harness. Surrogates pay off where the solver does not fit -- much larger grids, much longer horizons, stiff timestep restrictions, or differentiability through an entire rollout.
