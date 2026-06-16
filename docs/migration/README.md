# Migration Reports — PyTorch → JAX

One report per active implementation branch. Each documents the source, the JAX
port, the **numerical-correctness evidence**, and any behavioural differences.

| # | Source branch | Content | JAX port | Correctness | Status |
|---|---|---|---|---|---|
| 01 | `main` | Heat PI-NCA (`PI NCA_v1.py`) | `src/pinca_jax/` | ✅ in 25/25 gate | ✅ done |
| 02 | `Heterogenous-simulations` | heterogeneous heat notebook + ref `2407.06151` | `equations/heat_hetero.py` (planned) | planned | ◻ todo |
| 03 | `PI-NCA-Gray-Heat-Equation` | "Gray–Scott" notebook (= multi-res heat) | covered by `equations/pdes.py:heat` | ✅ | ✅ done |
| 04 | `new-update-of-physics-informed-models` | **8-PDE** suite (heat, wave, adv-diff, Allen-Cahn, Gray-Scott, SWE, Cahn-Hilliard, FHN) | `equations/pdes.py` (registry) | ✅ 8/8 step+rollout | ✅ done |

> **Correction (Phase 2):** the Phase-1 heat Laplacian used last-two-axes differencing,
> wrong for the NHWC training arrays (degenerate 1-D Laplacian in the training teacher).
> Fixed via `equations/operators.py` (NHWC spatial axes) + a 2-D isotropy regression test.
> See research log "Phase 2".

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
