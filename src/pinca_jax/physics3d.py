"""Conservation operators in 3-D (NDHWC). 3-D analogues of physics.py."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def total_mass(u):
    return u.sum(axis=(1, 2, 3, 4), keepdims=True)


def conserve_energy(u, target_sum):
    n_cells = u.shape[1] * u.shape[2] * u.shape[3]
    return u + (target_sum - total_mass(u)) / n_cells


def divergence_flux_update(x, flux):
    """Scalar 3-D conservative update. flux: (B,D,H,W,3) = (f_x,f_y,f_z)."""
    fx, fy, fz = flux[..., 0:1], flux[..., 1:2], flux[..., 2:3]
    dx = (jnp.roll(fx, 1, 3) - fx) + (jnp.roll(fy, 1, 2) - fy) + (jnp.roll(fz, 1, 1) - fz)
    return x + dx


def multichannel_divergence_update(x, flux):
    """Per-field 3-D divergence. x:(B,D,H,W,C); flux:(B,D,H,W,3C)."""
    B, D, H, W, C = x.shape
    f = flux.reshape(B, D, H, W, C, 3)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    dx = (jnp.roll(fx, 1, 3) - fx) + (jnp.roll(fy, 1, 2) - fy) + (jnp.roll(fz, 1, 1) - fz)
    return x + dx
