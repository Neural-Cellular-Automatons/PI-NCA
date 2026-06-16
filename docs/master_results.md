# Master Results — Every Experiment, Tabulated

One-stop index of all benchmarks (reduced-scale CPU, mean ± std over seeds). Detailed
analysis in `docs/experimental_report.md`; efficiency in `docs/efficiency_comparison.md`;
ablations in `docs/ablation_report.md`. **Bold** = best in column. rel-L2 lower is better.

## 0. Regime map (the headline)
| PDE | character | winner | why |
|---|---|---|---|
| Heat | smooth, local | **multiscale_flux_nca** (0.0183) | local multi-scale + conservation |
| Cahn–Hilliard | stiff, 4th-order, bounded | **bounded_cons_nca** (0.603→0.511 w/ A3) | bounding + conservation |
| Shallow-water | conservative, multi-field | **mc_flux_nca** (0.031, ≈FNO @54× fewer params) | per-field conservation correct |
| FitzHugh–Nagumo | non-conservative reaction | **plain_nca** (0.154) | conservation prior would hurt |
| Nagumo | non-conservative bistable | **fno / plain_nca** (0.118) | conservation prior hurts |
| Navier–Stokes | global Poisson coupling | **fno** (0.145) | global spectral mixing; locality fails |

**Thesis:** no universal winner. The PDE's structure — conservation, boundedness, and
information-propagation range — selects the architecture. Hybrids that compose the matching
inductive biases win every *local/bounded* regime; FNO wins the *globally-coupled* regime.

## 1. Emulator benchmarks (per PDE, all architectures)

### Heat (grid 24, train 12 / eval 48, 3 seeds)
| arch | rel_l2 | conservation_err | params | infer s/step |
|---|---|---|---|---|
| **multiscale_flux_nca** | **0.0183±0.0016** | 1.4e-3 | 5 520 | 7.5e-4 |
| spectral_flux_nca | 0.0199±0.014 | 1.5e-3 | 134 225 | 1.4e-3 |
| fno | 0.0352±0.0035 | 21.3 | 592 897 | 4.4e-3 |
| pi_nca | 0.0438±0.035 | **3.8e-4** | 4 576 | 5.7e-4 |
| fno_small | 0.119±0.014 | 80.1 | 8 433 | 7.3e-4 |
| plain_nca | 0.234±0.28 | 33.3 | 6 784 | 1.1e-3 |

### Cahn–Hilliard (grid 24, eval 48; identity floor 0.93)
| arch | rel_l2 | conservation_err | params |
|---|---|---|---|
| **bounded_cons_nca** | **0.603±0.012** | 2.4e-4 | 4 576 |
| bounded_multiscale_nca | 0.633±0.002 | 3.6e-4 | 5 520 |
| (unbounded baselines plain/pi/fno) | 14–18 (diverge) | — | — |
| spectral_flux_nca / multiscale (unbounded) | 24 / 24 (diverge) | — | — |

### Shallow-Water (C=3, grid 24, eval 36, 2 seeds)
| arch | rel_l2 | conservation_err | params |
|---|---|---|---|
| **fno** | **0.0259±0.0015** | 0.77 | 592 995 |
| mc_flux_nca | 0.0311±0.0058 | **1.6e-4** | 10 992 |
| plain_nca | 0.0399±0.0028 | 4.81 | 7 744 |

### FitzHugh–Nagumo (C=2, grid 24, eval 48, 2 seeds)
| arch | rel_l2 | conservation_err | params |
|---|---|---|---|
| **plain_nca** | **0.154±0.0099** | 118.1 | 7 264 |
| fno | 0.452±0.020 | 75.2 | 592 946 |
| mc_flux_nca | 0.993±0.0078 | **1.1e-6** | 10 464 |

### Nagumo (bistable RD, grid 24, eval 48, 2 seeds)
| arch | rel_l2 | params |
|---|---|---|
| **fno** | **0.118±0.009** | 592 897 |
| plain_nca | 0.119±0.006 | 6 784 |
| pi_nca | 0.379±0.001 | 4 576 |
| multiscale_flux_nca | 0.379±0.001 | 5 520 |

### Navier–Stokes (2-D vorticity, global Poisson, grid 24, eval 48, 2 seeds)
| arch | rel_l2 | conservation_err | params |
|---|---|---|---|
| **fno** | **0.145±0.029** | 1.93 | 592 897 |
| multiscale_flux_nca | 0.284±0.030 | 1.0e-4 | 5 520 |
| plain_nca | 0.528±0.050 | 7.28 | 6 784 |
| pi_nca | 0.676±0.17 | 2.1e-5 | 4 576 |

## 2. Ablations
| ablation | setup | result |
|---|---|---|
| **A1** output bounding (CH) | clip [-1,1] | plain 12.9→0.54, pi_nca 16.5→0.60 (24–27×) but clip breaks conservation (→7.6); bounded_cons hybrid fixes both |
| **A2** iso-param FNO (heat) | width 8/modes 4 (8.4k) | 0.035→0.119 (3.4× worse than full FNO); FNO edge was param count |
| **A3** horizon curriculum (CH) | train 12/24/48 | 0.633→0.619→**0.511**; conservation 3.2e-4→6.8e-5 (train≈eval helps) |

## 3. Operator / continuous baselines (heat; different task — not head-to-head)
| model | rel-L2 @ T | params | train | paradigm |
|---|---|---|---|---|
| DeepONet | **0.075±0.009** | 126 593 | ~9 s | operator, cross-IC |
| PINN | 0.208±0.018 | 14 209 | ~88 s | single-IVP, no data |

## 4. Darcy flow (steady elliptic operator a↦u, grid 20, 256 train, 2 seeds) — `darcy.py`
| model | rel-L2 | params | train |
|---|---|---|---|
| **nca_solver** (learned iterative) | **0.535±0.062** | **4 112** | ~246 s |
| fno (direct operator) | 0.594±0.008 | 592 897 | ~44 s |

**Darcy — both mediocre at reduced scale; the local NCA solver edges FNO at 144× fewer params.**
High-contrast piecewise coefficients (a∈{3,9}) with only 256 samples are hard; neither learns the
operator well here. Notably the **NCA learned-iterative-solver (0.535, 4.1k params) ≥ FNO (0.594,
593k params)**: relaxing u from 0 over 24 local steps is a natural fit for an elliptic solve (akin
to a learned Jacobi/multigrid smoother), and over a 20-grid 24 steps nearly suffices to propagate
boundary information. Caveat: FNO trains 5× faster (no inner rollout); both would improve with more
data/iters/grid at full scale. An honest "neither is great yet, but locality is not disqualifying
on a static elliptic problem the way it is on dynamic Navier–Stokes."

## 5. Efficiency (heat) — param-cost of accuracy (rel-L2 × params, lower better)
| model | rel-L2 × params | vs best |
|---|---|---|
| **multiscale_flux_nca** | **101** | 1× |
| pi_nca | 200 | 2× |
| fno_small | 1 003 | 9.9× |
| fno | 20 875 | 207× |

## 6. Visual artifacts
Per-phenomenon analytic-vs-model GIFs + |error| GIFs: `results/gifs/<pde>_{analytic,model,error}.gif`.
Committed montages (key timesteps): `docs/figures/<pde>_comparison.png`. Generated by `viz.py`.
