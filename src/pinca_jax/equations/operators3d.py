"""Periodic finite-difference operators in 3-D (JAX), NDHWC convention.

States are channels-last 5-D: (B, D, H, W, C). Spatial axes are 1 (D/z), 2 (H/y),
3 (W/x); channel axis is last and never differenced. Direct extension of operators.py.

    laplacian_3d(f) = Σ_{a∈{1,2,3}} [roll(f,1,a)+roll(f,-1,a)] − 6f     # 7-point stencil
    grad_x/y/z      = central difference on axis 3 / 2 / 1
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_Z, _Y, _X = 1, 2, 3  # depth, height, width axes


def laplacian_3d(f: jax.Array) -> jax.Array:
    return (
        jnp.roll(f, 1, _Z) + jnp.roll(f, -1, _Z)
        + jnp.roll(f, 1, _Y) + jnp.roll(f, -1, _Y)
        + jnp.roll(f, 1, _X) + jnp.roll(f, -1, _X)
        - 6.0 * f
    )


def grad_x(f):
    return (jnp.roll(f, -1, _X) - jnp.roll(f, 1, _X)) * 0.5


def grad_y(f):
    return (jnp.roll(f, -1, _Y) - jnp.roll(f, 1, _Y)) * 0.5


def grad_z(f):
    return (jnp.roll(f, -1, _Z) - jnp.roll(f, 1, _Z)) * 0.5
