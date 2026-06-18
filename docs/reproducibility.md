# Reproducibility Guide

Everything needed to reproduce the results. See also `docs/environment.md` (exact
package versions + the Windows orbax/MAX_PATH workaround).

## 1. Install
```bash
python -m pip install -r requirements-jax.txt     # Linux/macOS, or Windows w/ long paths
# Windows, no admin: see docs/environment.md for the --no-deps sequence
```
Verified stack: jax 0.10.1 (CPU), flax 0.12.7, optax 0.2.8, cax 0.3.3, torch 2.12.0 (CPU, reference only).

## 2. Correctness gate (run first)
```bash
python -m pytest tests/ -q          # 36 tests: migration correctness + suite + metrics
```
This asserts the JAX ports equal the verbatim PyTorch references to tolerance, the
2-D isotropy fix, conservation, the Gray-Scott dt=2.0 instability finding, and a
training smoke. **Must be green before trusting any benchmark.**

## 3. Reproduce a benchmark (multi-seed, mean±std)
```bash
python -m pinca_jax.bench --pde heat --seeds 3 --epochs 200 --grid 24 --rollout 12 --eval 48
python -m pinca_jax.bench --pde cahn_hilliard --seeds 3 --epochs 200 --grid 24
```
Writes `results/bench_<pde>.json` (machine-readable) and `results/bench_<pde>.md`
(table with per-column winners). Tables are pasted into `docs/experimental_report.md`.

Available PDEs: `heat, wave, adv_diff, allen_cahn, gray_scott, shallow_water,
cahn_hilliard, fitzhugh_nagumo`. Multi-channel PDEs run the channel-applicable
architectures only (the scalar flux PI-NCA is C=1).

## 3b. Reproduce the 3-D benchmarks
```bash
python -m pytest tests/test_pde3d_correctness.py -q   # 3-D correctness gate (11 tests)
python -m pinca_jax.bench3d --grid 16 --epochs 120    # detailed 3-D tables -> results/bench3d_*.md
```
3-D uses NDHWC `(B,D,H,W,C)`, reduced scale 16³, single seed 42. The 3-D modules mirror the
2-D ones (`*3d.py` / `*_3d`); same configs scale to larger grids on GPU.

## 4. Reproduce the PINN baseline
```bash
python -m pinca_jax.pinn_heat        # 2-D periodic heat IVP, rel-L2 vs solver at t=T
```

## 5. Determinism & seeds
- All randomness flows from `jax.random.PRNGKey(seed)`; no global RNG.
- Benchmarks sweep `seeds = range(N)` and report mean ± std (`metrics.aggregate`).
- XLA CPU reductions are not bitwise-deterministic across thread counts → correctness
  is asserted to tolerances, not bit-equality.

## 6. Scale knobs (CPU ↔ GPU)
`EmuConfig` / `HeatNCAConfig` / `PINNConfig` fields (`grid_size`, `rollout_steps`,
`eval_steps`, `epochs`, `batch`, seeds) are the *only* difference between the
reduced-scale CPU presets and full-scale GPU presets. Increase them on GPU; code is
unchanged. `pmap`/sharding paths are no-ops on a single device (this host).

## 7. Artifact map
| Artifact | Path |
|---|---|
| Literature review | `docs/literature_review.md` |
| Migration reports | `docs/migration/` (`README`, `01_heat_pinca_main`, `pde_inventory`) |
| Architecture report | `docs/architecture_report.md` |
| Experimental report | `docs/experimental_report.md` |
| Ablation report | `docs/ablation_report.md` *(in progress)* |
| Benchmarks (raw) | `results/bench_*.json`, `results/bench_*.md` |
| Research log (trail) | `docs/research_log.md` |
| Final summary | `docs/final_summary.md` *(in progress)* |
| JAX core | `src/pinca_jax/` |
| Correctness gate | `tests/` |
