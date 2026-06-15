"""Differentiable reference PDE solvers (JAX).

Each solver exposes a pure `step(state, params)->state` plus a `rollout` built on
`jax.lax.scan`, so it can act as a differentiable teacher (for NCA distillation),
a ground-truth generator (for operator training), and a metric oracle.
"""
from . import heat

__all__ = ["heat"]
