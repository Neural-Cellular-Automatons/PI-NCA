# Efficiency Comparison — Hybrid NCA vs FNO vs PINN/DeepONet

Reduced-scale CPU; all measured under the shared harness (heat unless noted), 3 seeds.
"Efficiency" = accuracy delivered per unit of parameters / compute / data.

## Measured cost & accuracy (heat, grid 24, emulator track)
| model | class | rel-L2 | params | infer (s/step) | train wall (s) | data |
|---|---|---|---|---|---|---|
| **multiscale_flux_nca** | hybrid NCA | **0.0183** | **5 520** | 7.5e-4 | ~55 | solver |
| pi_nca | conservative NCA | 0.0438 | 4 576 | **5.7e-4** | ~60 | solver |
| spectral_flux_nca | hybrid NCA+FNO | 0.0199 | 134 225 | 1.4e-3 | ~55 | solver |
| fno | FNO (standard) | 0.0352 | 592 897 | 4.4e-3 | ~71 | solver |
| fno_small | FNO (iso-param) | 0.119 | 8 433 | 7.3e-4 | ~11 | solver |
| DeepONet | operator | 0.075 @T | 126 593 | (single fwd) | ~9 | solver |
| PINN | continuous, single-IVP | 0.208 @T | 14 209 | (single fwd) | ~88 | **none** |

> PINN/DeepONet solve a different task (single-IVP / fixed-T operator) and are scored on a
> different horizon than the emulators — included for paradigm-level efficiency context, not a
> head-to-head accuracy race.

## Parameter efficiency (the headline)
**Param-cost of accuracy = rel-L2 × params** (lower is better):

| model | rel-L2 × params | relative to best |
|---|---|---|
| **multiscale_flux_nca** | **101** | 1× (best) |
| pi_nca | 200 | 2.0× |
| fno_small | 1 003 | 9.9× |
| spectral_flux_nca | 2 672 | 26× |
| fno (standard) | 20 875 | **207×** |

→ The **hybrid multi-scale NCA is ~200× more parameter-efficient than a standard FNO** on heat:
it is both *more accurate* (0.0183 vs 0.0352) and **107× smaller** (5.5k vs 593k params).

## Compute-scaling (analytic, per step)
| model | time / step | receptive field / step |
|---|---|---|
| NCA / hybrid NCA | **O(N · P)** (N cells, P params; 1×1-conv-dominated) | local (multi-scale: dilations widen it) |
| FNO | O(L · (N log N · W + modes²·W²)) | global (1 layer) |
| PINN | O(N_col · cost(∇²u_φ)) per opt step; O(1) per query | continuous (any x,t) |
| DeepONet | O(branch + trunk) per query | global (operator) |

NCA inference is **linear in cells** and the cheapest per step measured (5.7–7.5e-4 s/step);
FNO pays FFT + large spectral weights (4.4e-3 s/step, ~6–8× slower) for its global receptive field.

## Data efficiency
- **PINN: zero training data** — the equation *is* the supervision (mesh-free). Most data-efficient,
  least accurate here, solves only one IVP.
- **NCA / FNO / DeepONet:** need solver trajectories/pairs (distillation / operator regression).
- Among data-driven models, the **hybrid NCA needs the fewest parameters to fit the same data**.

## Verdict
For PDE *emulation*, the **hybrid multi-scale conservative NCA dominates the efficiency frontier**:
best accuracy on local/smooth dynamics at ~1% of FNO's parameters, cheapest inference, and small
training cost. FNO buys a global receptive field (essential on globally-coupled PDEs — see the
Navier–Stokes result) at a large parameter/compute premium. PINNs win only the *data* axis
(none required) but are the least accurate and do not generalise across initial conditions.
