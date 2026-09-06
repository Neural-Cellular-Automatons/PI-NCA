"""Standard autoregressive surrogates: the baselines a reviewer will ask for first.

Neither of these carries a physics prior. That is the point. Without them the study
compares NCA variants against a spectral operator only, and any conclusion about
*locality* or *conservation* is confounded with "we did not try the obvious CNN".

* `ResNetEmulator` -- the workhorse autoregressive PDE surrogate: a stack of residual
  blocks with circular padding, fixed receptive field, no downsampling. This is the
  direct control for the NCA family: same locality class, same residual parameterisation,
  no cellular-automaton weight sharing across the update and no flux structure.
* `UNetEmulator` -- the standard multi-resolution surrogate (as used throughout PDE
  emulation and in PDEBench-style comparisons). It reaches a global receptive field by
  downsampling rather than by spectral mixing, so it separates "global information
  helps" from "Fourier mixing helps" -- which the FNO alone cannot.

Both predict a residual increment with a zero-initialised head, so they start as the
identity map exactly like the NCA baselines, and both take the same
`__call__(state) -> next_state` NHWC emulator interface as everything else in
`models/registry.py`.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

_HE = nn.initializers.he_normal()
_CIRC = dict(padding="CIRCULAR", kernel_init=_HE)


class ResNetEmulator(nn.Module):
    """Residual CNN, circular padding, no resolution change.

    Receptive field is 1 + 2*2*depth cells per step with 3x3 kernels, i.e. local like an
    NCA but with independent weights per block instead of one shared update rule.
    """
    out_channels: int = 1
    width: int = 32
    depth: int = 4
    kernel: int = 3

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        k = (self.kernel, self.kernel)
        h = nn.Conv(self.width, k, name="stem", **_CIRC)(x)
        for d in range(self.depth):
            r = nn.gelu(nn.Conv(self.width, k, name=f"b{d}c1", **_CIRC)(h))
            r = nn.Conv(self.width, k, name=f"b{d}c2", **_CIRC)(r)
            h = nn.gelu(h + r)
        delta = nn.Conv(self.out_channels, (1, 1), name="head", use_bias=False,
                        kernel_init=nn.initializers.zeros)(h)
        return x + delta


def _down(u):
    """2x2 mean pool on a periodic grid (H, W assumed even at this level)."""
    B, H, W, C = u.shape
    return u.reshape(B, H // 2, 2, W // 2, 2, C).mean(axis=(2, 4))


def _up(u):
    """Nearest-neighbour 2x upsample -- cheap, and exactly inverts `_down`'s shape."""
    return jnp.repeat(jnp.repeat(u, 2, axis=1), 2, axis=2)


class UNetEmulator(nn.Module):
    """Small U-Net with circular padding and mean-pool / nearest-neighbour resampling.

    `levels` halvings; a 32x32 grid at levels=2 sees an 8x8 bottleneck, which is a
    global receptive field for these domains. Grids not divisible by 2**levels fall back
    to fewer levels rather than silently cropping -- an emulator that quietly changed
    resolution would corrupt every rollout metric downstream.
    """
    out_channels: int = 1
    width: int = 24
    levels: int = 2
    kernel: int = 3

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        k = (self.kernel, self.kernel)
        H, W = x.shape[1], x.shape[2]
        levels = self.levels
        while levels > 0 and (H % (2 ** levels) or W % (2 ** levels)):
            levels -= 1

        def block(u, w, tag):
            u = nn.gelu(nn.Conv(w, k, name=f"{tag}c1", **_CIRC)(u))
            return nn.gelu(nn.Conv(w, k, name=f"{tag}c2", **_CIRC)(u))

        h = block(x, self.width, "in")
        skips = []
        for lv in range(levels):
            skips.append(h)
            h = block(_down(h), self.width * (2 ** (lv + 1)), f"d{lv}")
        for lv in reversed(range(levels)):
            h = _up(h)
            h = jnp.concatenate([h, skips[lv]], axis=-1)
            h = block(h, self.width * (2 ** lv), f"u{lv}")
        delta = nn.Conv(self.out_channels, (1, 1), name="head", use_bias=False,
                        kernel_init=nn.initializers.zeros)(h)
        return x + delta


class IdentityEmulator(nn.Module):
    """The do-nothing floor: g(x) = x.

    Any architecture whose rollout error exceeds this has learned something worse than
    nothing, and the paper reports that rather than only ranking the survivors. Carries
    one unused parameter so it fits the same init/apply plumbing as every other model.
    """
    out_channels: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        z = self.param("unused", nn.initializers.zeros, (1,))
        return x + z * 0.0


def demo():
    import numpy as np
    key = jax.random.PRNGKey(0)
    for C in (1, 3):
        x = jax.random.normal(key, (2, 16, 16, C))
        for m in (ResNetEmulator(out_channels=C), UNetEmulator(out_channels=C),
                  IdentityEmulator(out_channels=C)):
            p = m.init(key, x)
            y = m.apply(p, x)
            assert y.shape == x.shape, (type(m).__name__, y.shape)
            # zero-init head => every model starts as the identity
            assert np.allclose(np.asarray(y), np.asarray(x), atol=1e-6), type(m).__name__
    # odd grid must degrade levels, not crash or resize
    x = jax.random.normal(key, (1, 12, 12, 1))
    m = UNetEmulator(out_channels=1, levels=3)
    assert m.apply(m.init(key, x), x).shape == x.shape
    print("baselines.demo OK")


if __name__ == "__main__":
    demo()
