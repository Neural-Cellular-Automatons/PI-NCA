### Teacher (solver) error and cost - grid 16, horizon 12 steps

`vs_analytic` = rel-L2 against the closed-form periodic solution (linear constant-coefficient PDEs). `self_convergence` = rel-L2 between the production dt and a dt/8 reference over the same physical time; the observed order should match the scheme's design order.

| PDE | estimate | teacher rel-L2 | spatial | temporal | observed order | verdict | solver ms/step | fastest emulator ms/step | emulator vs solver |
|---|---|---|---|---|---|---|---|---|---|
| heat | vs_analytic | 1.344e-02 | 2.034e-02 | 7.198e-03 | - | converging | 0.011 | 0.408 (plain_nca) | **36x slower** |
| wave | vs_analytic | 5.680e-03 | 5.866e-03 | 1.635e-03 | - | converging | 0.014 | 0.416 (plain_nca) | **30x slower** |
| adv_diff | vs_analytic | 2.646e-02 | 2.727e-02 | 1.799e-03 | - | converging | 0.012 | 0.431 (plain_nca) | **36x slower** |
| allen_cahn | self_convergence | 8.000e-04 | - | - | 1.04 | converging | 0.039 | 0.363 (plain_nca) | **9x slower** |
| gray_scott | self_convergence | 1.122e-02 | - | - | 1.02 | converging | 0.011 | 0.423 (plain_nca) | **38x slower** |
| shallow_water | self_convergence | 3.466e-07 | - | - | -0.56 | round-off limited | 0.073 | 0.362 (plain_nca) | **5x slower** |
| cahn_hilliard | self_convergence | 5.603e-01 | - | - | -0.06 | **NOT CONVERGING** | 0.035 | 0.416 (plain_nca) | **12x slower** |
| fitzhugh_nagumo | self_convergence | 1.850e-02 | - | - | 1.00 | converging | 0.016 | 0.461 (plain_nca) | **29x slower** |
| nagumo | self_convergence | 1.025e-03 | - | - | 0.99 | converging | 0.014 | 0.364 (plain_nca) | **26x slower** |
| navier_stokes | self_convergence | 1.037e-02 | - | - | 1.15 | converging | 0.200 | 0.397 (plain_nca) | **2x slower** |

**How to read an emulator's rel-L2 against this table.** An emulator error far above the teacher error is limited by learning: the target is not the binding constraint and architecture comparisons are meaningful. An emulator error approaching the teacher error has saturated the target, and further ranking there measures the teacher's own discretisation error rather than model quality.

`round-off limited` means the Cauchy differences reached machine precision, so the order estimate is undefined -- the teacher is fine, the estimator is not. `NOT CONVERGING` means refining the timestep did not reduce the difference at all.

> **Teacher does not converge on: cahn_hilliard.** Refining the timestep does not reduce the solver's own change, so the distillation target on these equations is not a converged solution. Any architecture ranking on them is measuring the solver's instability rather than model quality, and should not be reported as a result. This is the most likely explanation wherever no architecture beats the do-nothing identity floor.

> **The reference solver is faster than every emulator on: heat, wave, adv_diff, allen_cahn, gray_scott, shallow_water, cahn_hilliard, fitzhugh_nagumo, nagumo, navier_stokes.** At this grid size a learned surrogate is not a deployment case on these equations; it is a measurement harness. Surrogates pay off where the solver does not fit -- much larger grids, much longer horizons, stiff timestep restrictions, or differentiability through an entire rollout.
