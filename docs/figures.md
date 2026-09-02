# Visual Gallery — Analytic vs Model + Error Maps

For each phenomenon: a 3-row montage (top = analytic/solver, middle = trained model,
bottom = |error|) at five timesteps t = 0, T/4, T/2, 3T/4, T (T = 64 steps, grid 32).
Animated GIFs (analytic / model / error) are in `results/gifs/<pde>_{analytic,model,error}.gif`
(generated locally; not committed). Regenerate with `python -m pinca_jax.viz --pde <name>`.

Each model is the regime-appropriate default (`viz.DEFAULT_ARCH`): multi-scale conservative
NCA for local/smooth PDEs, bounded conservative NCA for Cahn–Hilliard, multi-channel
conservative NCA for SWE/Gray–Scott, plain NCA for FitzHugh–Nagumo, FNO for Navier–Stokes.

> Note: these are single-run illustrations at the viz config (grid 32, 120 epochs, 64-step
> rollout) — qualitative, not the multi-seed benchmark numbers in `docs/master_results.md`.

| phenomenon | model shown | final-frame rel-err | montage |
|---|---|---|---|
| Heat | multiscale_flux_nca | 0.022 | ![heat](figures/heat_comparison.png) |
| Allen–Cahn | multiscale_flux_nca | 0.027 | ![allen_cahn](figures/allen_cahn_comparison.png) |
| Shallow-water | mc_flux_nca | 0.045 | ![swe](figures/shallow_water_comparison.png) |
| Navier–Stokes | fno | 0.221 | ![ns](figures/navier_stokes_comparison.png) |
| FitzHugh–Nagumo | plain_nca | 0.305 | ![fhn](figures/fitzhugh_nagumo_comparison.png) |
| Nagumo | multiscale_flux_nca | 0.465 | ![nagumo](figures/nagumo_comparison.png) |
| Gray–Scott | mc_flux_nca | 0.498 | ![gs](figures/gray_scott_comparison.png) |
| Cahn–Hilliard | bounded_cons_nca | 0.720 | ![ch](figures/cahn_hilliard_comparison.png) |

**Reading the gallery.** The easy local regimes (heat, Allen–Cahn, shallow-water) show
near-perfect analytic/model agreement with faint structured error. The hard regimes show where
single-run models lag: Cahn–Hilliard (stiff; the 64-step viz horizon exceeds the bounded model's
comfort zone), Gray–Scott/Nagumo (the conserving default fights the non-conservative reaction),
and Navier–Stokes (global coupling — even FNO accumulates visible error over 64 steps). These
match the quantitative regime map in `docs/master_results.md`.

## 3-D gallery — true volume renders (`viz3d_volume.py`)
These are **genuine 3-D renderings** (matplotlib `Axes3D`): each volume is drawn as a
perspective cube with a depth-shaded volumetric point cloud, so the 3-D structure is visible
(not a flat slice). Rows = analytic / model / |error|, columns = timesteps. Rotating + evolving
animated GIFs: `results/gifs/<pde>_3d_volume.gif` (local). Committed montages:

| 3-D phenomenon | model shown | volume montage |
|---|---|---|
| Heat | FluxNCA3D (PI-NCA) | ![heat3dv](figures/heat_3d_volume.png) |
| Advection–diffusion | FNO3D | ![adv3dv](figures/adv_diff_3d_volume.png) |
| Allen–Cahn | FNO3D | ![ac3dv](figures/allen_cahn_3d_volume.png) |
| Nagumo | NCA3D | ![nag3dv](figures/nagumo_3d_volume.png) |
| Gray–Scott | NCA3D | ![gs3dv](figures/gray_scott_3d_volume.png) |
| FitzHugh–Nagumo | FNO3D | ![fhn3dv](figures/fitzhugh_nagumo_3d_volume.png) |

### Complementary cross-sections (mid-depth z-slice, `viz3d.py`)
The `*_3d_comparison.png` montages show the mid-depth slice (z = D/2) over time as a quantitative
cross-section to read the error field cell-by-cell — a 2-D view *into* the 3-D volume, paired with
the volume renders above. Both mirror the dimension-independent regime map
(`docs/master_results.md §5b`).
