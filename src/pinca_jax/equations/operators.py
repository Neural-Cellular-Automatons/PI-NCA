"""Periodic finite-difference operators (JAX), NHWC convention.

All states in this package are channels-last: (B, H, W, C). Spatial axes are
**axis 1 (H / y)** and **axis 2 (W / x)**; the channel axis is last and is never
differenced. This matches the Flax conv convention used by the models and the
divergence update in physics.py.

Operators mirror the notebook reference exactly (see docs/migration/pde_inventory.md):
    laplacian(f) = roll(f,1,y)+roll(f,-1,y)+roll(f,1,x)+roll(f,-1,x) - 4f
    grad_x(f)    = (roll(f,-1,x) - roll(f,1,x)) * 0.5     # central difference
    grad_y(f)    = (roll(f,-1,y) - roll(f,1,y)) * 0.5
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_Y = 1  # height axis
_X = 2  # width axis


def laplacian(f: jax.Array) -> jax.Array:
    """Periodic 5-point Laplacian on spatial axes (1,2) of an NHWC array."""
    return (
        jnp.roll(f, 1, axis=_Y)
        + jnp.roll(f, -1, axis=_Y)
        + jnp.roll(f, 1, axis=_X)
        + jnp.roll(f, -1, axis=_X)
        - 4.0 * f
    )


def grad_x(f: jax.Array) -> jax.Array:
    """Central-difference x-derivative (width axis)."""
    return (jnp.roll(f, -1, axis=_X) - jnp.roll(f, 1, axis=_X)) * 0.5


def grad_y(f: jax.Array) -> jax.Array:
    """Central-difference y-derivative (height axis)."""
    return (jnp.roll(f, -1, axis=_Y) - jnp.roll(f, 1, axis=_Y)) * 0.5
