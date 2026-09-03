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

## Measurement (`python -m pinca_jax.cax_eval`, grid 32, 64 steps, batch 8)
| rollout | CPU ms/step | GPU ms/step (RTX 3050) |
|---|---|---|
| our `lax.scan` `DeepFluxNCA` (C=1) | ~1.17 | ~0.066 |
| CAX `ComplexSystem` NCA (C=16 + hidden) | ~2.42 | ~0.165 |
| ratio (CAX/ours) | ~2.1x | ~2.5x |

**Caveat (honest):** this is *not* a param-matched comparison — the CAX NCA is a standard
multi-channel NCA (16 state channels, 48 perception, 128 hidden) doing more work per step
than our scalar conservative flux NCA, so the gap is mostly model size, not framework
overhead. The point stands regardless: **CAX gives no rollout speedup, on CPU or GPU** —
the ratio is the same order of magnitude on both backends, confirming the "same XLA scan,
no separate kernel path" reasoning above rather than a CPU-specific artifact.

**GPU gotcha found while reproducing this:** at this grid/step/batch scale both rollouts
finish in single-digit milliseconds, so `cuda_timer` logs `Delay kernel timed out:
measured time has sub-optimal accuracy` — a benign warmup-kernel warning, not a
correctness issue. It does not change the conclusion; a larger grid or step count would
clear it if tighter timing precision were ever needed here.

## Re-measurement (2026-07, after the training loop became a single `lax.scan`)
`python -m pinca_jax.cax_eval` on the same CPU host: our `lax.scan` rollout **0.638 ms/step**
vs CAX `ComplexSystem` **1.202 ms/step** — CAX is **1.88x slower**. Training now runs *all*
epochs inside one jitted `lax.scan` (`harness.train_emulator`), so the rollout is already a
single fused XLA program; CAX's `nnx.scan` cannot improve on that, it is the same mechanism.

## Decision
- **Not integrated into the hot path, on either backend.** Our `lax.scan` core is fully
  under our control, correctness-gated (119+ tests), and uses Flax `linen` consistently
  with the conservative flux/divergence physics layers; swapping to CAX's `nnx` NCA would
  not speed up the rollout on CPU *or* GPU (measured above) and would fork the
  architecture for no benefit.
- **Removed from the GPU pipeline.** `cax` is no longer listed in `requirements-gpu.txt` —
  a real GPU run installs one fewer dependency it was never using. `cax_eval.py` degrades
  gracefully without it (catches the import failure, prints `CAX path error`, and still
  reports the `lax.scan` number); install `cax==0.4.3` yourself only to re-run the CAX
  side of the comparison.
- The multi-device/sharding, CA-zoo, and alternative-perception-stencil angles noted in
  an earlier draft of this doc were speculative "future GPU work" items gated on actually
  having a GPU to test on. Now that we do, the core finding (same XLA scan, no
  intrinsic speedup) held on GPU exactly as it did on CPU, which removes the motivation
  to chase those angles — they'd inherit the same "no framework-level advantage" ceiling.
- A working CAX NCA subclass is kept in `cax_eval.py` as a reference integration point,
  in case a future CAX release adds something (e.g. a genuinely different kernel/hardware
  path) worth re-measuring.

## Reproduce
```bash
pip install cax==0.4.3   # not installed by default — see "Removed from the GPU pipeline"
python -m pinca_jax.cax_eval
```
