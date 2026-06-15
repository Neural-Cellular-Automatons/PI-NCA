# Migration 01 — Heat PI-NCA (`main` / `PI NCA_v1.py`) → JAX

**Status:** ✅ Complete and numerically verified (`tests/` 7/7 pass).

## Source
`PI NCA_v1.py` (PyTorch): a fused, conservative Physics-Informed NCA that learns
a local update imitating a differentiable 2-D heat solver, trained by truncated
BPTT with AMP and `torch.compile`. Key components:
- `HeatEquationSolver` — fixed 5-point-Laplacian conv, circular padding, explicit Euler.
- `DeepFluxNCA` — perceive (3×3 circular conv) → 1×1-conv MLP → **flux head** →
  discrete-divergence update (finite-volume / conservation form), flux head zero-initialised.
- `conserve_energy` — projects total mass onto the initial total.

## JAX port (files)
| PyTorch | JAX (`src/pinca_jax/`) | Notes |
|---|---|---|
| `HeatEquationSolver.step/k_steps` | `equations/heat.py` (`heat_step`, `rollout`, `rollout_trajectory`) | Laplacian via `jnp.roll` (== circular conv for a symmetric kernel); rollouts via `lax.scan`. |
| `DeepFluxNCA` | `models/flux_nca.py` | Flax `linen`, NHWC, `padding="CIRCULAR"`, zero-init flux head preserved. |
| `conserve_energy`, divergence update | `physics.py` | Divergence factored out for reuse by hybrids. Mass-deficit divided by true `H*W` (general; == reference when square). |
| training loop (AMP, truncated BPTT, Adam, StepLR) | `train_nca.py` + `configs.py` + `data.py` | `jit` train step, `optax.adamw`, `lax.scan` rollout, vectorised `make_blobs`. |

## Correctness evidence
`python -m pytest tests/` → **7 passed**:
- `test_laplacian_matches_torch_circular_conv` (2 grid shapes) — JAX Laplacian == PyTorch circular conv.
- `test_heat_step_and_rollout_match_torch` — single step **and** 25-step `lax.scan` rollout match `k_steps`.
- `test_flux_nca_matches_torch_with_nonzero_head` — weights ported torch→flax
  (kernel transpose `(O,I,kH,kW)→(kH,kW,I,O)`); with a **non-zero** flux head the
  full perceive→MLP→divergence path matches (atol 1e-5).
- `test_divergence_update_conserves_mass` — net mass change of the divergence update ≈ 0 (periodic).
- `test_conserve_energy_hits_target` — projection hits the target total exactly.

## End-to-end
`python -m pinca_jax.train_nca --smoke` → loss `1.11e-1 → 7.22e-3` (15.4×) in **2.0 s** on CPU.

## Behavioural differences (documented, intentional)
- **Laplacian implementation**: `roll`-based vs conv. Mathematically identical for
  the symmetric kernel; floating-point reduction order differs (asserted to 1e-5, not bit-equal).
- **`conserve_energy` denominator**: reference used `W**2` (square-only); port uses
  `H*W` (correct for any shape; identical when square).
- **AMP / `torch.compile`** have no JAX analogue needed: XLA fuses the `jit`-ed step;
  mixed precision is a config choice, deliberately off for the correctness baseline.
- **Optimizer**: `optax.adamw` (decoupled weight decay) ≈ torch `Adam(weight_decay=...)`
  which is L2-coupled. For exact parity one can use `optax.add_decayed_weights` +
  `optax.adam`; adamw chosen as the modern default (noted; affects only fine training dynamics).
- **Curriculum on rollout length**: reference grows `steps` per epoch; the port fixes
  `rollout_steps` (static → one XLA compile / hot cache). Curriculum is a config knob
  to re-enable with bucketed step counts.

## Hardware honesty
Single CPU device → `pmap`/`sharding` would be no-ops; not used here. `jit`, `vmap`,
`lax.scan` provide the realised speedups. Wall-clock numbers are CPU, reduced-scale.
