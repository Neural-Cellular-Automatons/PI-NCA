"""3-D initial-condition generators (NDHWC). 3-D analogues of ic.py."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def gaussian_blobs_3d(key, batch, size, n_min=3, n_max=5, amp_lo=5.0, amp_hi=10.0,
                      sigma_frac=0.10):
    nb = n_max
    sig = size * sigma_frac
    kc, ka, kn = jax.random.split(key, 3)
    centres = jax.random.randint(kc, (batch, nb, 3), 0, size).astype(jnp.float32)
    amps = jax.random.uniform(ka, (batch, nb), minval=amp_lo, maxval=amp_hi)
    cnt = jax.random.randint(kn, (batch,), n_min, nb + 1)
    mask = (jnp.arange(nb)[None] < cnt[:, None]).astype(jnp.float32)
    amps = amps * mask

    coords = jnp.arange(size, dtype=jnp.float32)
    zz, yy, xx = jnp.meshgrid(coords, coords, coords, indexing="ij")  # (size,size,size)

    def one_sample(cs, a):
        def one_blob(c, amp):
            dz = jnp.minimum(jnp.abs(zz - c[0]), size - jnp.abs(zz - c[0]))
            dy = jnp.minimum(jnp.abs(yy - c[1]), size - jnp.abs(yy - c[1]))
            dx = jnp.minimum(jnp.abs(xx - c[2]), size - jnp.abs(xx - c[2]))
            return amp * jnp.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * sig ** 2))
        return jax.vmap(one_blob)(cs, a).sum(0)

    grid = jax.vmap(one_sample)(centres, amps)  # (batch,size,size,size)
    return grid[..., None]


def make_state(key, pde_name, batch, size):
    if pde_name in ("heat", "adv_diff"):
        return gaussian_blobs_3d(key, batch, size)
    if pde_name == "allen_cahn":
        k1, k2 = jax.random.split(key)
        phase = jnp.sign(jax.random.normal(k1, (batch, size, size, size, 1)))
        return phase + jax.random.normal(k2, phase.shape) * 0.05
    if pde_name == "nagumo":
        noise = jax.random.normal(key, (batch, size, size, size, 1)) * 0.5
        return jax.nn.sigmoid(3.0 * noise)
    if pde_name == "gray_scott":
        # random cubic seeds of v
        k = jax.random.normal(key, (batch, size, size, size, 1))
        v = (jax.nn.sigmoid(8.0 * (k - 1.0)) * 0.5)
        u = 1.0 - v
        return jnp.concatenate([u, v], axis=-1)
    if pde_name == "fitzhugh_nagumo":
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (batch, size, size, size, 1)) * 0.1
        v = jax.random.normal(k2, (batch, size, size, size, 1)) * 0.1
        return jnp.concatenate([u, v], axis=-1)
    raise KeyError(pde_name)
