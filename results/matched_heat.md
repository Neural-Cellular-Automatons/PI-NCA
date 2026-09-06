### Matched comparison - heat, 4 held-out initial conditions

Task: produce the solution at T = 8 solver steps for all 4 initial conditions, drawn from the same distribution. Same PDE, same ICs, same horizon, same reference, same metric. The emulator (`multiscale_flux_nca`) trains once and rolls out 4 times; the PINN trains once **per initial condition**. Grid 12.

| | rel-L2 mean | 95% CI | median | params | fixed cost (s) | per-IC cost (s) | total for K=4 (s) |
|---|---|---|---|---|---|---|---|
| emulator (`multiscale_flux_nca`) | 1.605e-01 | [1.50e-01, 1.75e-01] | 1.561e-01 | 5520 | 2.8 | 0.0584 | 3.0 |
| PINN (per IC) | 3.179e-01 | [2.92e-01, 3.52e-01] | 3.098e-01 | 14209 | 0 | 6.0 | 23.8 |
| numerical solver | 0 (reference) | - | - | - | 0 | 0.0000 | 0.000 |

Paired test (same ICs): mean difference -1.574e-01 [-1.78e-01, -1.35e-01], Wilcoxon p = 1, emulator wins on 100% of initial conditions -> **not resolvable**.

**Crossover: K = 1.** Below that many initial conditions the per-IC PINN is cheaper in total wall-clock; above it the emulator's one-off training is amortised. This is the number a practitioner needs, and it is why a single-IC accuracy comparison between the two paradigms is not informative on its own.

**Compute asymmetry.** The PINN received 7.91x the emulator's total wall-clock on this task. That is stated rather than equalised, because the two budgets are not interchangeable -- the PINN's is K independent runs, the emulator's is one -- but it does settle the direction: the emulator's advantage here is not bought with extra compute.

**Against the solver, both surrogates lose at this scale.** The numerical solver produces the reference in 0.038 ms per initial condition; the emulator takes 1541x that per initial condition and is less accurate, and the PINN is slower still. A learned surrogate pays off only where the solver does not fit -- much larger grids, much longer horizons, stiff timestep restrictions, or the need to differentiate through the whole rollout. On a 12x12 periodic grid with an explicit stepper, none of those apply, so this regime is a measurement harness rather than a deployment case. No claim in this repository should be read as 'replace the solver'.

The solver row is included deliberately. It is the reference, so its error is zero by construction here -- `pinca_jax.teacher_error` measures what it actually gets wrong against closed-form solutions -- but its *cost* is real, and any claim that a surrogate is worthwhile has to clear the solver, not only the other surrogate.
