"""Initial-condition generators for the 8-PDE suite (vectorised JAX, NHWC).

Ported from the notebook `make_state`/`make_gaussian_blobs`
(docs/migration/pde_inventory.md). Where the notebook used per-sample Python
loops (gray_scott seed boxes), we use an equivalent vectorised construction at
fixed seed/blob counts (documented; reduced-scale faithful). FitzHugh-Nagumo's
IC was not in the captured source, so a standard small-perturbation excitable IC
is used (flagged below).

All generators take an explicit PRNGKey and return NHWC `(batch, H, W, C)`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .equations.operators import laplacian  # noqa: F401  (kept for parity utilities)


def gaussian_blobs(key, batch, size, n_min=3, n_max=5, amp_lo=5.0, amp_hi=10.0,
                   sigma_frac=0.08):
    """Periodic sum of up to n_max Gaussian blobs (random count in [n_min,n_max])."""
    nb = n_max
    sig = size * sigma_frac
    kx, ky, ka, kc = jax.random.split(key, 4)
    cx = jax.random.randint(kx, (batch, nb), 0, size).astype(jnp.float32)
    cy = jax.random.randint(ky, (batch, nb), 0, size).astype(jnp.float32)
    amps = jax.random.uniform(ka, (batch, nb), minval=amp_lo, maxval=amp_hi)
    cnt = jax.random.randint(kc, (batch,), n_min, nb + 1)
    mask = (jnp.arange(nb)[None] < cnt[:, None]).astype(jnp.float32)
    amps = amps * mask

    coords = jnp.arange(size, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")  # (size,size)
    dx = jnp.abs(xx[None, None] - cx[:, :, None, None]); dx = jnp.minimum(dx, size - dx)
    dy = jnp.abs(yy[None, None] - cy[:, :, None, None]); dy = jnp.minimum(dy, size - dy)
    blob = amps[:, :, None, None] * jnp.exp(-(dx ** 2 + dy ** 2) / (2 * sig ** 2))
    g = blob.sum(axis=1)  # (batch,size,size)
    return g[..., None]   # NHWC


def _box_seeds(key, batch, size, n_seeds=6, r_lo=2, r_hi=8):
    """Vectorised random square patches (for gray_scott), as a [0,1] mask field."""
    kc, kr = jax.random.split(key)
    cx = jax.random.randint(kc, (batch, n_seeds), r_hi, size - r_hi).astype(jnp.float32)
    cy = jax.random.randint(jax.random.fold_in(kc, 1), (batch, n_seeds), r_hi, size - r_hi).astype(jnp.float32)
    r = jax.random.randint(kr, (batch, n_seeds), r_lo, r_hi).astype(jnp.float32)
    coords = jnp.arange(size, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
    inx = jnp.abs(xx[None, None] - cx[:, :, None, None]) < r[:, :, None, None]
    iny = jnp.abs(yy[None, None] - cy[:, :, None, None]) < r[:, :, None, None]
    box = (inx & iny).any(axis=1)  # (batch,size,size)
    return box[..., None].astype(jnp.float32)


def make_state(key, pde_name, batch, size):
    """Dispatch per-PDE IC → NHWC (batch,H,W,C). C matches REGISTRY[pde].channels."""
    if pde_name in ("heat", "adv_diff"):
        return gaussian_blobs(key, batch, size)

    if pde_name == "wave":
        u = gaussian_blobs(key, batch, size, amp_lo=2.0, amp_hi=5.0, sigma_frac=0.10)
        return jnp.concatenate([u, jnp.zeros_like(u)], axis=-1)

    if pde_name == "allen_cahn":
        k1, k2 = jax.random.split(key)
        phase = jnp.sign(jax.random.normal(k1, (batch, size, size, 1)))
        return phase + jax.random.normal(k2, phase.shape) * 0.05

    if pde_name == "gray_scott":
        box = _box_seeds(key, batch, size)
        u = jnp.where(box > 0, 0.5, 1.0)
        v = jnp.where(box > 0, 0.3, 0.0)
        return jnp.concatenate([u, v], axis=-1)

    if pde_name == "shallow_water":
        h = 1.0 + gaussian_blobs(key, batch, size, n_min=2, n_max=4,
                                 amp_lo=0.15, amp_hi=0.5, sigma_frac=0.10)
        z = jnp.zeros_like(h)
        return jnp.concatenate([h, z, z], axis=-1)

    if pde_name == "cahn_hilliard":
        noise = jax.random.normal(key, (batch, size, size, 1)) * 0.4
        # 3x3 periodic box smoothing (== notebook's circular conv with ones/9).
        sm = sum(jnp.roll(jnp.roll(noise, dy, 1), dx, 2)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
        return jnp.clip(sm, -0.99, 0.99)

    if pde_name == "nagumo":
        # bistable RD: smoothed random field in (0,1)
        noise = jax.random.normal(key, (batch, size, size, 1)) * 0.5
        sm = sum(jnp.roll(jnp.roll(noise, dy, 1), dx, 2)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
        return jax.nn.sigmoid(3.0 * sm)

    if pde_name == "navier_stokes":
        # random low-wavenumber vorticity field, zero mean (decaying turbulence IC)
        w = gaussian_blobs(key, batch, size, n_min=3, n_max=6,
                           amp_lo=-3.0, amp_hi=3.0, sigma_frac=0.10)
        k2 = jax.random.split(key)[0]
        signs = jnp.sign(jax.random.normal(k2, (batch, 1, 1, 1)))
        w = w * signs
        return w - w.mean(axis=(1, 2, 3), keepdims=True)

    if pde_name == "fitzhugh_nagumo":
        # NOTE: not captured from notebook source — standard excitable IC:
        # small random perturbation of both fields near the rest state.
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (batch, size, size, 1)) * 0.1
        v = jax.random.normal(k2, (batch, size, size, 1)) * 0.1
        return jnp.concatenate([u, v], axis=-1)

    raise KeyError(pde_name)
