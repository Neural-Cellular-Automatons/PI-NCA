# `PI NCA_v1.py` is a legacy reference, not the implementation

The original PyTorch script is kept at the repository root for provenance: it is the
artifact the JAX package was migrated from, and the migration reports in
[`docs/migration/`](migration/) assert numerical equality against it. It is **not** part
of the pipeline, is not exercised by the correctness gate, and no number in the paper
comes from running it.

This document exists so that is a stated fact rather than something a reader has to infer,
and so the specific defects found during migration are on the record instead of being
quietly fixed in the rewrite.

## Defects found in the original script

Each is a real behaviour of the code as committed, with the line reference. Two of them
would materially mislead a reader of that script's own output.

### 1. The plotted training curve can be identically zero (lines 164-196)

The truncated-BPTT loop accumulates `acc_loss` and, at every truncation boundary, steps
the optimiser and then resets the accumulator:

```python
if (t + k) % CONFIG["truncate_depth"] == 0:
    ...
    acc_loss = torch.tensor(0.0, device=device)
...
train_loss.append(acc_loss.item())     # line 196
```

`train_loss` is appended *after* the loop. When the sampled rollout length is an exact
multiple of `truncate_depth` -- which happens whenever `steps % truncate_depth == 0`, a
substantial fraction of epochs given `steps` is drawn uniformly -- the last thing that
happened to `acc_loss` was the reset, so the recorded value is `0.0`. The training curve
plotted at line 269 therefore mixes real losses with zeros, and the apparent trend is an
artifact of which rollout lengths happened to be drawn.

The correct quantity is the mean loss over the epoch's truncation windows. The JAX harness
returns exactly that, from a single `lax.scan` over epochs, with one device sync at the
end (`harness.train_emulator`).

### 2. Validation evaluates a different model from training and test (lines 198-207)

Training applies the conservation projection inside the rollout:

```python
pred = model(pred)
pred = conserve_energy(pred, target_sum)   # line 176
```

and so does the final test (line 227). Validation does not:

```python
for _ in range(500):
    vt = solver.step(vt)
    vp = model(vp)                          # line 205 -- no conserve_energy
```

The validation number is therefore measuring the unprojected model while both the training
objective and the reported test result measure the projected one. For a model whose entire
premise is the conservation projection, this is not a small inconsistency: validation and
test are different architectures.

In the JAX package there is one rollout path (`harness._emu_traj`) used by training,
evaluation, capture and figures, so this class of mismatch cannot recur -- and the
projection is part of the module (`models/hybrids.py`) rather than something a call site
has to remember to apply.

### 3. Deprecated AMP API with no fallback (lines 11, 145, 173)

`from torch.cuda.amp import autocast, GradScaler` and bare `with autocast():` are the
pre-2.0 spellings; current PyTorch wants `torch.amp.autocast("cuda")` /
`torch.amp.GradScaler("cuda")` and warns on the old ones. More importantly the script has
no CPU path: `GradScaler` and CUDA autocast make the script GPU-only in practice, with no
graceful degradation.

### 4. `torch.compile` with no compatibility fallback (line 140)

```python
solver.k_steps = torch.compile(solver.k_steps)
```

is unconditional. On a platform or PyTorch build where Inductor is unavailable (Windows
without a supported toolchain, older drivers), this raises at import time rather than
falling back to eager. The JAX package uses `jax.jit`, which has no equivalent
availability cliff, and the one place a backend choice is made
(`env.require_gpu`) fails with an explicit message rather than a compiler traceback.

### 5. Global RNG state (lines 17-18)

```python
torch.manual_seed(42)
np.random.seed(42)
```

Global seeding makes reproducibility depend on execution order: adding a single call that
consumes randomness shifts every subsequent draw. `np.random.uniform` at line 127 and
`np.random.randint` at line 161 both draw from that global state inside the training loop.
The JAX package threads an explicit `PRNGKey` everywhere and never touches a global RNG,
which is why a seed in `EmuConfig` reproduces a run exactly regardless of what else ran
first.

### 6. No CLI, no config file, no checkpointing

`CONFIG` is a module-level dict edited in place. There is no way to run a sweep without
editing the source, no record in the output of which configuration produced it, and a
crash loses the whole run. The JAX drivers take arguments, stamp `{config, seeds, device}`
into every results JSON, and checkpoint per matrix cell so an interrupted sweep resumes.

## What was preserved from it

The defects above are process problems, not physics problems. The parts that encode the
actual method were migrated verbatim and are asserted equal to the original to tolerance
by `tests/test_migration_correctness.py`:

* the flux-divergence conservative update,
* the `conserve_energy` mass projection (generalised to per-channel; identical at C=1),
* the He/kaiming initialisation and zero-initialised head,
* the Gaussian-blob initial-condition generator,
* the heat solver and its parameters.

The "start from a better point" training recipe -- warmup, zero-init head, pre-seeding on
developed states -- is also taken from this script and its notebook successors, and is
documented in [`docs/initialization_and_protocol.md`](initialization_and_protocol.md).

## If you want to run it anyway

It needs PyTorch with CUDA and will emit deprecation warnings. Nothing in this repository
depends on its output, and its numbers should not be compared with the JAX tables:
different framework, different precision defaults, different seeding, no shared harness.
