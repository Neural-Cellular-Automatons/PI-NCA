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

## Decision: auto-selected on GPU, never on CPU
CAX is now wired in behind a backend switch (`src/pinca_jax/cax_backend.py`), not forked
into the models:

* `rollout(step_fn, x0, steps)` drives the rollout with CAX's `ComplexSystem`
  (`nnx.scan` + `nnx.jit`) when `jax.default_backend() == "gpu"` and `cax` imports,
  and with `jax.lax.scan` otherwise.
* `PINCA_CAX=1` / `PINCA_CAX=0` forces either backend (the tests use this).
* Trajectory rollouts (`collect=True`) always use `lax.scan`: CAX's driver returns only
  the final state, and the metrics need every frame.
* `harness._emu_rollout` / `_emu_traj` both route through it, so training, evaluation,
  capture and the figures all inherit the policy from one place.

**Equivalence is asserted, not assumed** (`tests/test_cax_backend.py`): the two backends
agree to 1e-6 on a rollout, the CAX path is differentiable (checked against the analytic
gradient of `sum(x s^n)`), and a 40-epoch heat training run gives bit-identical results
either way (loss 2.129980e-03, rel-L2 7.424576e-02). On CPU the CAX path is measurably
slower (15.2 s vs 14.6 s for that run; 1.202 vs 0.638 ms/step in the microbenchmark),
which is precisely why the default is GPU-only.

**Still unmeasured:** whether CAX beats `lax.scan` on an actual GPU. No GPU was available
here. The switch means a GPU run picks it up automatically; if it turns out slower there
too, `PINCA_CAX=0` disables it without touching any model code.

## Reproduce
```bash
python -m pinca_jax.cax_eval
```
