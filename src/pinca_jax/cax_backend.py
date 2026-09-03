"""Rollout backend: CAX on GPU, jax.lax.scan everywhere else.

CAX (arXiv:2410.02651) drives a cellular-automaton rollout with `nnx.scan` inside
`nnx.jit`. That is the same mechanism as our hand-written `jax.lax.scan`, and on this
CPU host CAX measured *slower* (1.202 vs 0.638 ms/step, see docs/cax_evaluation.md).
It is therefore selected only where it might plausibly pay off -- an actual GPU -- and
never on CPU, which is exactly the policy asked for.

Selection order:
  1. PINCA_CAX=1 / =0 forces the backend on or off (used by the tests).
  2. otherwise: CAX iff the JAX backend is "gpu" *and* cax imports.

Both paths are pure `state -> state` rollouts and must agree numerically;
tests/test_cax_backend.py asserts that, including through a gradient.
"""
from __future__ import annotations

import os

import jax


def _env_override():
    v = os.environ.get("PINCA_CAX")
    if v is None:
        return None
    return v.strip().lower() not in ("0", "false", "no", "")


def use_cax() -> bool:
    """True when the CAX rollout should be used."""
    forced = _env_override()
    try:
        import cax  # noqa: F401
    except Exception:
        return False                      # not installed -> never
    if forced is not None:
        return forced
    return jax.default_backend() == "gpu"


def _cax_rollout(step_fn, x0, steps):
    """Final state after `steps` applications of step_fn, driven by CAX."""
    from flax import nnx
    from cax.core.cs import ComplexSystem

    class _Wrapped(ComplexSystem):
        # ComplexSystem is abstract: it supplies the jitted nnx.scan driver via
        # __call__(state, num_steps=...); a subclass only has to define _step.
        def __init__(self, fn):
            self.fn = fn

        def _step(self, state, input=None, *, sow=False):
            return self.fn(state)

    return _Wrapped(step_fn)(x0, num_steps=steps)


def rollout(step_fn, x0, steps, collect: bool = False):
    """Roll `step_fn` forward `steps` times.

    collect=False -> final state (CAX-eligible).
    collect=True  -> stacked trajectory (always lax.scan; CAX returns only the
                     final state, and the trajectory is what the metrics need).
    """
    if not collect and use_cax():
        return _cax_rollout(step_fn, x0, steps)

    def body(x, _):
        y = step_fn(x)
        return y, (y if collect else None)

    xf, ys = jax.lax.scan(body, x0, xs=None, length=steps)
    return ys if collect else xf
