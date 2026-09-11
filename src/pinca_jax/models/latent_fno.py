"""Latent-space FNO emulator: encode, evolve spectrally in the latent, decode.

This is the architecture a JEPA-style objective needs, and it is useful on its own. The
three stages are exposed as separate methods so the same weights can be driven three
ways, which is the point of building it:

* ``__call__`` -- encode, predict, decode every step. This is a one-step field map
  ``s_t -> s_{t+1}``, so it satisfies the emulator interface every other architecture in
  ``models/registry.py`` satisfies and drops into the shared harness, the paired
  statistics, and the identity floor unchanged.
* ``encode`` / ``predict`` / ``decode`` -- called individually, they allow a rollout that
  encodes once, iterates the predictor ``K`` times on the downsampled latent grid, and
  decodes once. On a ``p``-fold patch embedding that is ``p^2`` fewer spatial positions
  per step. Whether it is faster in wall-clock, and what it costs in accuracy, is
  measured rather than asserted (``pinca_jax.jepa``).
* ``predict`` alone is what a self-supervised latent objective trains, with no decoder in
  the loop at all.

The two rollout modes are **not the same model applied two ways**. Per-step decoding
matches the teacher at every step; latent rollout only matches it at the end, because
``decode(encode(x))`` is not the identity. They are therefore measured as separate
entries and never pooled into one accuracy column.

Design choices worth stating:

* The encoder is a strided circular convolution and the decoder is a pointwise
  convolution followed by a pixel shuffle. Pixel shuffle rather than a transposed
  convolution because it is exactly shape-inverting and does not produce the
  checkerboard artefacts a transposed convolution does on a periodic field.
* The predictor is the existing ``FNO2d`` applied to the latent grid, not a new spectral
  implementation. Reusing it means the latent predictor and the full-resolution FNO
  baseline differ only in what they are applied to, which is what makes the comparison
  between them mean anything.
* The decoder head is zero-initialised, so the model begins as the identity map exactly
  like every other emulator here. Without that, a comparison at epoch zero would be
  measuring initialisation rather than learning.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .fno import FNO2d

_HE = nn.initializers.he_normal()


def largest_patch(size: int, wanted: int) -> int:
    """Largest patch <= `wanted` that divides `size`; 1 if none does.

    The benchmark grids are 12, 16, 24, 32 and 48, all divisible by 4, but silently
    cropping a grid that does not divide would corrupt every rollout metric downstream,
    so the patch degrades instead.
    """
    for p in range(min(wanted, size), 0, -1):
        if size % p == 0:
            return p
    return 1


class LatentFNOEmulator(nn.Module):
    out_channels: int = 1
    latent_dim: int = 32
    patch: int = 4
    width: int = 32
    modes: int = 6          # latent grid is size/patch, so few modes are available
    depth: int = 4

    def setup(self):
        p = self.patch
        self.enc = nn.Conv(self.latent_dim, (p, p), strides=(p, p), padding="CIRCULAR",
                           kernel_init=_HE, name="encoder")
        self.pred = FNO2d(out_channels=self.latent_dim, width=self.width,
                          modes=self.modes, depth=self.depth, residual=True,
                          name="predictor")
        # Pointwise to p*p*C, then pixel shuffle back to full resolution. Zero-init so
        # the whole model starts as the identity in field space.
        self.dec = nn.Conv(self.out_channels * p * p, (1, 1), name="decoder",
                           use_bias=False, kernel_init=nn.initializers.zeros)

    # -- stages, callable individually via nn.Module.apply(..., method=...) -----------
    def encode(self, x: jax.Array) -> jax.Array:
        """(B,H,W,C) -> (B,H/p,W/p,latent_dim)."""
        self._check_divisible(x)
        return self.enc(x)

    def predict(self, z: jax.Array) -> jax.Array:
        """Latent -> latent. One step of spectral evolution; the only stage a latent
        self-supervised objective trains."""
        return self.pred(z)

    def decode(self, z: jax.Array) -> jax.Array:
        """(B,H/p,W/p,latent_dim) -> (B,H,W,C) field *increment* (not a state)."""
        p = self.patch
        B, Hl, Wl, _ = z.shape
        flat = self.dec(z)                                     # (B,Hl,Wl,C*p*p)
        # pixel shuffle: (B,Hl,Wl,p,p,C) -> (B,Hl*p,Wl*p,C)
        shuf = flat.reshape(B, Hl, Wl, p, p, self.out_channels)
        shuf = shuf.transpose(0, 1, 3, 2, 4, 5)
        return shuf.reshape(B, Hl * p, Wl * p, self.out_channels)

    def _check_divisible(self, x):
        H, W = x.shape[1], x.shape[2]
        if H % self.patch or W % self.patch:
            raise ValueError(
                f"grid {H}x{W} is not divisible by patch {self.patch}; construct with "
                f"patch=largest_patch({H}, {self.patch}) rather than letting the encoder "
                f"crop, which would silently change the field size mid-rollout")

    @nn.nowrap
    def _unused(self):                                          # pragma: no cover
        return None

    def __call__(self, x: jax.Array) -> jax.Array:
        """One field-space step: encode, evolve, decode, add. The emulator interface."""
        return x + self.decode(self.predict(self.encode(x)))


def latent_rollout(model, params, x0, steps):
    """Encode once, iterate the predictor `steps` times, decode once.

    This is the cheap mode and a *different map* from applying `__call__` `steps` times:
    it never returns to field space in between, so the encoder's reconstruction error is
    paid once instead of every step, and the trajectory is not constrained to match the
    teacher at intermediate steps. Returns only the final state, because intermediate
    states do not exist in field space here.
    """
    z = model.apply(params, x0, method=model.encode)

    def body(z, _):
        return model.apply(params, z, method=model.predict), None

    zK, _ = jax.lax.scan(body, z, xs=None, length=steps)
    return x0 + model.apply(params, zK, method=model.decode)


def latent_shape(grid: int, patch: int, latent_dim: int) -> tuple[int, int, int]:
    return (grid // patch, grid // patch, latent_dim)


def demo():
    import numpy as np
    key = jax.random.PRNGKey(0)
    for C in (1, 3):
        for grid in (16, 24, 48):
            x = jax.random.normal(key, (2, grid, grid, C))
            m = LatentFNOEmulator(out_channels=C, patch=4)
            p = m.init(key, x)
            y = m.apply(p, x)
            assert y.shape == x.shape, (grid, C, y.shape)
            # zero-init decoder => identity at initialisation, like every other emulator
            assert np.allclose(np.asarray(y), np.asarray(x), atol=1e-6)
            z = m.apply(p, x, method=m.encode)
            assert z.shape == (2, grid // 4, grid // 4, m.latent_dim), z.shape
            assert m.apply(p, z, method=m.predict).shape == z.shape
            assert m.apply(p, z, method=m.decode).shape == x.shape
            # latent rollout keeps the field shape and also starts as the identity
            r = latent_rollout(m, p, x, 5)
            assert r.shape == x.shape
            assert np.allclose(np.asarray(r), np.asarray(x), atol=1e-6)
    # an indivisible grid must raise, not crop
    m = LatentFNOEmulator(out_channels=1, patch=4)
    try:
        m.init(key, jax.random.normal(key, (1, 18, 18, 1)))
    except ValueError as e:
        assert "not divisible" in str(e)
    else:
        raise AssertionError("indivisible grid silently accepted")
    assert largest_patch(18, 4) == 3 and largest_patch(48, 4) == 4
    assert largest_patch(7, 4) == 1
    print("latent_fno.demo OK")


if __name__ == "__main__":
    demo()
