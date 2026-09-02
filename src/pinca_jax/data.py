"""Initial-condition generators (JAX). Migrated/vectorised from `make_state`."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def make_blobs(key: jax.Array, batch: int, size: int, n_blobs: int,
               sigma_frac: float, amp_low: float, amp_high: float) -> jax.Array:
    """Periodic sum-of-Gaussian-blobs initial conditions → (batch, size, size, 1).

    Vectorised version of the PyTorch `make_state`: random centres, random
    amplitudes, fixed number of blobs (jit-friendly), periodic (toroidal) distance.
    """
    sigma = size * sigma_frac
    k_c, k_a = jax.random.split(key)
    centres = jax.random.randint(k_c, (batch, n_blobs, 2), 0, size).astype(jnp.float32)
    amps = jax.random.uniform(k_a, (batch, n_blobs), minval=amp_low, maxval=amp_high)

    coords = jnp.arange(size, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(coords, coords, indexing="ij")  # (size, size)

    def one_sample(cs, a):  # cs:(n_blobs,2) a:(n_blobs,)
        def one_blob(c, amp):
            cy, cx = c[1], c[0]  # matches torch: cx,cy = randint; centre indexed (cx,cy)
            dx = jnp.minimum(jnp.abs(xx - cx), size - jnp.abs(xx - cx))
            dy = jnp.minimum(jnp.abs(yy - cy), size - jnp.abs(yy - cy))
            return amp * jnp.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
        return jax.vmap(one_blob)(cs, a).sum(axis=0)

    grid = jax.vmap(one_sample)(centres, amps)  # (batch, size, size)
    return grid[..., None]  # NHWC
