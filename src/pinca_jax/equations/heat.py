"""2-D heat / diffusion equation — differentiable JAX solver.

Migrated from the PyTorch `HeatEquationSolver` in `PI NCA_v1.py`.

PyTorch reference:
    kernel = [[0,1,0],[1,-4,1],[0,1,0]]               # 5-point Laplacian
    laplace = Conv2d(1,1,3, padding=1, padding_mode="circular", bias=False)
    step(u) = u + (alpha*dt) * laplace(u)

JAX port: a circular-padded convolution with a *symmetric* fixed kernel is
*exactly* the roll-based 5-point stencil, so we implement the Laplacian with
`jnp.roll` (no conv op, fully fused under XLA, trivially `vmap`/`scan`-able).
Equivalence to the PyTorch conv is asserted in tests/test_migration_correctness.py.

State convention: arrays are (..., H, W) — the Laplacian acts on the last two
axes, so leading axes (batch, channels) broadcast for free.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def laplacian_periodic(u: jax.Array) -> jax.Array:
    """Periodic 5-point Laplacian on the last two axes.

    Identical to a circular-padded conv2d with kernel [[0,1,0],[1,-4,1],[0,1,0]].
    """
    return (
        jnp.roll(u, 1, axis=-1)
        + jnp.roll(u, -1, axis=-1)
        + jnp.roll(u, 1, axis=-2)
        + jnp.roll(u, -1, axis=-2)
        - 4.0 * u
    )


def heat_step(u: jax.Array, alpha_dt: float) -> jax.Array:
    """One explicit-Euler heat step: u + (alpha*dt) * Laplacian(u)."""
    return u + alpha_dt * laplacian_periodic(u)


def rollout(u0: jax.Array, alpha_dt: float, n_steps: int) -> jax.Array:
    """Roll the solver forward `n_steps`, returning the final state.

    Uses lax.scan so the trajectory has O(1) Python overhead and is fully
    differentiable end-to-end (used as a teacher in NCA distillation).
    """
    def body(u, _):
        return heat_step(u, alpha_dt), None

    u_final, _ = jax.lax.scan(body, u0, xs=None, length=n_steps)
    return u_final


def rollout_trajectory(u0: jax.Array, alpha_dt: float, n_steps: int) -> jax.Array:
    """Like `rollout` but stacks every intermediate state: (n_steps, ...)."""
    def body(u, _):
        u_next = heat_step(u, alpha_dt)
        return u_next, u_next

    _, traj = jax.lax.scan(body, u0, xs=None, length=n_steps)
    return traj
