# CAX Accelerator Evaluation

**Question (from the mandate):** can the CAX accelerator (Cellular Automata Accelerated
in JAX, arXiv:2410.02651) improve performance, and should we integrate it?

## What CAX is
An `nnx`-based library of CA primitives: `ConvPerceive` / `MoorePerceive` /
`VonNeumannPerceive`, `NCAUpdate` / `MLPUpdate` / `ResidualUpdate`, and an abstract
`ComplexSystem` whose `__call__(state, num_steps=K)` drives a multi-step rollout via
`nnx.scan` wrapped in `nnx.jit`. It also ships a zoo of complex systems (Lenia, Flow-Lenia,
Game of Life, Boids, elementary CA).

## Key observation
CAX's rollout driver is **`nnx.scan` + `nnx.jit`** — structurally identical to our
hand-written `jax.lax.scan` rollout (`harness.py`, `equations/*.rollout`). Both compile to
the same XLA scan. There is no separate kernel or hardware path that would make CAX
intrinsically faster on the same device.

## Measurement (`python -m pinca_jax.cax_eval`, single CPU, grid 32, 64 steps, batch 8)
| rollout | ms/step |
|---|---|
| our `lax.scan` `DeepFluxNCA` (C=1) | ~1.17 |
| CAX `ComplexSystem` NCA (C=16 + hidden) | ~2.42 |

**Caveat (honest):** this is *not* a param-matched comparison — the CAX NCA is a standard
multi-channel NCA (16 state channels, 48 perception, 128 hidden) doing more work per step
than our scalar conservative flux NCA, so the 2× gap is mostly model size, not framework
overhead. The point stands regardless: **CAX gives no rollout speedup on a single CPU**.

## Re-measurement (2026-07, after the training loop became a single `lax.scan`)
`python -m pinca_jax.cax_eval` on the same CPU host: our `lax.scan` rollout **0.638 ms/step**
vs CAX `ComplexSystem` **1.202 ms/step** — CAX is **1.88x slower**. Training now runs *all*
epochs inside one jitted `lax.scan` (`harness.train_emulator`), so the rollout is already a
single fused XLA program; CAX's `nnx.scan` cannot improve on that, it is the same mechanism.

## Decision
- **Not integrated into the hot path.** Our `lax.scan` core is fully under our control,
  correctness-gated (36+ tests), and uses Flax `linen` consistently with the conservative
  flux/divergence physics layers; swapping to CAX's `nnx` NCA would not speed up CPU
  rollouts and would fork the architecture.
- **Where CAX *would* be beneficial (documented for future GPU work):** (1) multi-device /
  large-grid scaling, where its `nnx.scan` composes with `jax.sharding` the same way ours
  would — no advantage but no disadvantage; (2) reusing its CA zoo (Lenia/Flow-Lenia) as
  additional emulation targets or pretrained perception kernels; (3) its `MoorePerceive` /
  `VonNeumann` perception stencils as ready-made ablation variants for neighbourhood-size
  studies (ablation A5).
- A working CAX NCA subclass is kept in `cax_eval.py` as a reference integration point.

## Reproduce
```bash
python -m pinca_jax.cax_eval
```
