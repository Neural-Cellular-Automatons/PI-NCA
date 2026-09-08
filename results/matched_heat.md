### Matched comparison - heat, 32 held-out initial conditions

Task: produce the solution at T = 48 solver steps for all 32 initial conditions, drawn from the same distribution. Same PDE, same ICs, same horizon, same reference, same metric. The emulator (`multiscale_flux_nca`) trains once and rolls out 32 times; the PINN trains once **per initial condition**. Grid 32.

| | rel-L2 mean | 95% CI | median | params | fixed cost (s) | per-IC cost (s) | total for K=32 (s) |
|---|---|---|---|---|---|---|---|
| emulator (`multiscale_flux_nca`) | 1.529e-02 | [1.44e-02, 1.65e-02] | 1.477e-02 | 5520 | 27.2 | 0.0550 | 28.9 |
| PINN (per IC) | 1.235e-01 | [1.18e-01, 1.29e-01] | 1.267e-01 | 14209 | 0 | 389.8 | 12474.3 |
| numerical solver | 0 (reference) | - | - | - | 0 | 0.0189 | 0.604 |

Paired test (same ICs): mean difference -1.082e-01 [-1.14e-01, -1.03e-01], Wilcoxon p = 4.66e-10, emulator wins on 100% of initial conditions -> **emulator better**.

**Crossover: K = 1.** Below that many initial conditions the per-IC PINN is cheaper in total wall-clock; above it the emulator's one-off training is amortised. This is the number a practitioner needs, and it is why a single-IC accuracy comparison between the two paradigms is not informative on its own.

**Compute asymmetry.** The PINN received 431.10x the emulator's total wall-clock on this task. That is stated rather than equalised, because the two budgets are not interchangeable -- the PINN's is K independent runs, the emulator's is one -- but it does settle the direction: the emulator's advantage here is not bought with extra compute.

**Against the solver, both surrogates lose at this scale.** The numerical solver produces the reference in 18.884 ms per initial condition; the emulator takes 3x that per initial condition and is less accurate, and the PINN is slower still. A learned surrogate pays off only where the solver does not fit -- much larger grids, much longer horizons, stiff timestep restrictions, or the need to differentiate through the whole rollout. On a 32x32 periodic grid with an explicit stepper, none of those apply, so this regime is a measurement harness rather than a deployment case. No claim in this repository should be read as 'replace the solver'.

The solver row is included deliberately. It is the reference, so its error is zero by construction here -- `pinca_jax.teacher_error` measures what it actually gets wrong against closed-form solutions -- but its *cost* is real, and any claim that a surrogate is worthwhile has to clear the solver, not only the other surrogate.
