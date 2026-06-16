"""Conservation operators shared across models (JAX).

Migrated from `PI NCA_v1.py`:
- `conserve_energy` (mass/energy projection onto a fixed total).
- the discrete-divergence flux update used by DeepFluxNCA, exposed standalone so
  hybrid models can reuse the conservation structure.

Channels-last (NHWC) to match Flax convs: arrays are (B, H, W, C).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def total_mass(u: jax.Array) -> jax.Array:
    """Sum over spatial+channel axes (H, W, C), keepdims → (B,1,1,1)."""
    return u.sum(axis=(1, 2, 3), keepdims=True)


def conserve_energy(u: jax.Array, target_sum: jax.Array) -> jax.Array:
    """Project `u` so its total mass equals `target_sum`.

    PyTorch reference distributed the deficit equally over cells, dividing by
    `u.shape[-1]**2` (valid only for square H==W in NCHW). Here, in NHWC, we
    divide by the true number of spatial cells H*W so it is correct for any shape
    (identical to the reference when H==W).
    """
    n_cells = u.shape[1] * u.shape[2]
    diff = (target_sum - total_mass(u)) / n_cells
    return u + diff


def divergence_flux_update(x: jax.Array, flux: jax.Array) -> jax.Array:
    """Apply a discrete divergence of a 2-channel flux field as a state increment.

    `flux[..., 0]` is the x-flux (width axis=2), `flux[..., 1]` the y-flux
    (height axis=1). Backward-difference divergence, matching the PyTorch:
        dx = (roll(fx,1,W) - fx) + (roll(fy,1,H) - fy)
    Returns `x + dx`. This makes the update a discrete conservation law: the
    net increment summed over a periodic grid is exactly zero (telescoping),
    so mass is conserved up to floating point.
    """
    fx = flux[..., 0:1]
    fy = flux[..., 1:2]
    dx = (jnp.roll(fx, 1, axis=2) - fx) + (jnp.roll(fy, 1, axis=1) - fy)
    return x + dx


def multichannel_divergence_update(x: jax.Array, flux: jax.Array) -> jax.Array:
    """Per-channel discrete-divergence update for multi-field states.

    x: (B,H,W,C); flux: (B,H,W,2C) as [fx_0,fy_0, fx_1,fy_1, ...]. Each channel gets
    its own conservative flux-divergence increment, so EACH channel's total sum is
    conserved on a periodic grid (telescoping). Physically correct on a periodic
    domain for conserved quantities (e.g. shallow-water mass+momentum); a deliberately
    *wrong* prior for reaction systems with source terms (FitzHugh-Nagumo) — which is
    itself a test of when the conservation bias helps vs hurts.
    """
    B, H, W, C = x.shape
    f = flux.reshape(B, H, W, C, 2)
    fx, fy = f[..., 0], f[..., 1]  # (B,H,W,C)
    dx = (jnp.roll(fx, 1, axis=2) - fx) + (jnp.roll(fy, 1, axis=1) - fy)
    return x + dx
