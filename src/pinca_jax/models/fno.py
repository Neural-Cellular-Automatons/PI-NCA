"""Fourier Neural Operator (2-D, Flax linen) — the global/spectral baseline.

Li et al. 2021 (arXiv:2010.08895). Each spectral layer mixes information globally
via a truncated Fourier multiplier:  v -> σ( W·v + F⁻¹( R ⊙ F[v] ) ).
Contrast with the NCA's one-cell-per-step locality: the FNO has a global receptive
field in O(1) layers (lit review §4).

Used here as a one-step emulator predicting a residual state increment, so it is
directly comparable to the NCA baselines under the same harness.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class SpectralConv2d(nn.Module):
    out_channels: int
    modes1: int
    modes2: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        scale = 1.0 / (C * self.out_channels)

        def cparam(name):
            r = self.param(name + "_r", nn.initializers.normal(scale), (m1, m2, C, self.out_channels))
            i = self.param(name + "_i", nn.initializers.normal(scale), (m1, m2, C, self.out_channels))
            return r + 1j * i

        w_top = cparam("w_top")   # low-freq corner (top rows)
        w_bot = cparam("w_bot")   # low-freq corner (bottom rows / negative freqs)

        x_ft = jnp.fft.rfft2(x, axes=(1, 2))  # (B,H,Wf,C) complex
        Wf = x_ft.shape[2]
        out_ft = jnp.zeros((B, H, Wf, self.out_channels), dtype=x_ft.dtype)

        def mix(slc, w):
            return jnp.einsum("bxyi,xyio->bxyo", slc, w)

        out_ft = out_ft.at[:, :m1, :m2, :].set(mix(x_ft[:, :m1, :m2, :], w_top))
        out_ft = out_ft.at[:, -m1:, :m2, :].set(mix(x_ft[:, -m1:, :m2, :], w_bot))
        return jnp.fft.irfft2(out_ft, s=(H, W), axes=(1, 2))


class FNO2d(nn.Module):
    out_channels: int = 1
    width: int = 24
    modes: int = 8
    depth: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        v = nn.Conv(self.width, (1, 1), name="lift")(x)
        for d in range(self.depth):
            spec = SpectralConv2d(self.width, self.modes, self.modes, name=f"spec{d}")(v)
            local = nn.Conv(self.width, (1, 1), name=f"w{d}")(v)
            v = nn.gelu(spec + local)
        v = nn.gelu(nn.Conv(self.width, (1, 1), name="proj1")(v))
        delta = nn.Conv(self.out_channels, (1, 1), name="proj2",
                        kernel_init=nn.initializers.zeros)(v)
        return x + delta  # residual one-step emulator (parity with NCA framing)
