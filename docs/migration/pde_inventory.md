# PDE Inventory — verified solver specs extracted from the notebooks

Source-of-truth for migrating the notebook branches (extracted directly from the
PyTorch notebook source, **not** assumed). The `new-update-of-physics-informed-models`
notebook is an **8-PDE edition**. Shared differential operators:

```
laplacian(f) = roll(f,1,-1)+roll(f,-1,-1)+roll(f,1,-2)+roll(f,-1,-2) - 4f   # periodic 5-pt
grad_x(f)    = (roll(f,-1,-1) - roll(f,1,-1)) * 0.5                          # central diff
grad_y(f)    = (roll(f,-1,-2) - roll(f,1,-2)) * 0.5
```

| # | PDE | State | Update (per step) | Params | Integrator |
|---|---|---|---|---|---|
| 1 | Heat | u | `u + dt·α·∇²u` | α=0.5, dt=0.1 | Euler |
| 2 | Wave | (u,v) | `v += dt·c²·∇²u; u += dt·v` | c=0.5, dt=0.05 | symplectic Euler |
| 3 | Advection–Diffusion | u | `u + dt·(D∇²u − v_x ∂ₓu − v_y ∂_yu)` | D=0.1, v_x=0.3, v_y=0.2, dt=0.08 | Euler |
| 4 | Allen–Cahn | u | `u + dt·(ε²∇²u + u − u³)` | ε²=0.01, dt=0.04 | Euler |
| 5 | Gray–Scott | (u,v) | `uvv=u v²; du=D_u∇²u−uvv+F(1−u); dv=D_v∇²v+uvv−(F+k)v` | D_u=0.2, D_v=0.05, F=0.035, k=0.065, dt=2.0 | Euler |
| 6 | Shallow-Water | (h,hu,hv) | conservative flux RHS (below) | g=1.0, dt=0.05 | **RK4** |
| 7 | Cahn–Hilliard | u | `μ=u³−u−ε²∇²u; u += dt·∇²μ; clamp(−1,1)` | ε²=0.01, dt=0.5 | Euler |
| 8 | FitzHugh–Nagumo | (u,v) | `du=D_u∇²u+(u−u³/3−v)/τ; dv=D_v∇²v+ε(u+a−bv)` | D_u=0.5, D_v=0.1, a=0.7, b=0.8, τ=12.5, ε=0.08, dt=0.1 | Euler |

**Shallow-Water RHS** (state = h, hu, hv; u=hu/h, v=hv/h):
```
dh  = −(∂ₓ(hu)         + ∂_y(hv))
dhu = −(∂ₓ(hu·u + ½g h²) + ∂_y(hu·v))
dhv = −(∂ₓ(hu·v)         + ∂_y(hv·v + ½g h²))
```
integrated with RK4; positivity asserted (`h>0`).

## Other branches
- `PI-NCA-Gray-Heat-Equation` notebook: despite the name, the implemented solver is the
  **heat** equation (conv-based Laplacian, same DeepFluxNCA), evaluated at 64²/128² for
  400/2000/10000 steps. CONFIG: train_size 64, test_size 128, batch 8, lr 1e-3, α=0.5,
  dt=0.1, min/max curriculum steps 40/150. (Gray–Scott proper lives in the v3 notebook, #5.)
- `Heterogenous-simulations`: heterogeneous heat `∂ₜu = ∇·(α(x)∇u)` + ref `2407.06151`
  (to read for the spatially-varying-coefficient setup).

## NCA in v3 notebook
"Multi-Scale Flux-Divergence" NCA + two-phase training; PSNR/MSE/energy/rel-L2 metrics;
long-horizon eval. This directly informs the multi-scale-NCA hybrid (lit review §7.3).

## Migration plan (correctness-gated, same pattern as Heat)
Port each solver to `src/pinca_jax/equations/` as a pure `step`/`rollout`, then assert
equivalence to these PyTorch formulas to tolerance on random inputs before any architecture
change. Multi-channel states (wave, GS, SWE, FHN) use NHWC with C=state-dim.
