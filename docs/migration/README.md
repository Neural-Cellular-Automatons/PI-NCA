# Migration Reports — PyTorch → JAX

One report per active implementation branch. Each documents the source, the JAX
port, the **numerical-correctness evidence**, and any behavioural differences.

| # | Source branch | Content | JAX port | Correctness | Status |
|---|---|---|---|---|---|
| 01 | `main` | Heat PI-NCA (`PI NCA_v1.py`) | `src/pinca_jax/` | `tests/` 7/7 pass | ✅ done |
| 02 | `Heterogenous-simulations` | heterogeneous heat notebook + ref `2407.06151` | `equations/heat_hetero.py` (planned) | planned | ◻ todo |
| 03 | `PI-NCA-Gray-Heat-Equation` | Gray–Scott RD notebook | `equations/gray_scott.py` (planned) | planned | ◻ todo |
| 04 | `new-update-of-physics-informed-models` | SWE / FHN / Cahn–Hilliard notebook | `equations/{swe,fhn,cahn_hilliard}.py` (planned) | planned | ◻ todo |

## Migration principles (applied to every branch)
1. **Correctness gate first.** Port the *differentiable reference solver* and any
   neural module, then assert equivalence to the PyTorch original to tolerance
   (`atol=1e-5, rtol=1e-4`) on random inputs — *before* touching architecture.
2. **Functional + pure.** No global RNG; explicit `PRNGKey`. Steppers expose a
   single-step fn so rollouts use `jax.lax.scan`.
3. **Modern-JAX idioms.** `jit` whole train steps; `vmap` over batch/seeds;
   `lax.scan` for rollouts; `jax.checkpoint` for BPTT memory; `optax` optimisers.
4. **Honest hardware notes.** `pmap`/`sharding` paths are written where they would
   help on multi-device, but are **no-ops on this single-CPU host** — documented,
   not silently omitted.
5. **Reproducibility.** Reduced-scale (CPU) and full-scale (GPU) presets differ
   only in numeric config fields (`src/pinca_jax/configs.py`).
