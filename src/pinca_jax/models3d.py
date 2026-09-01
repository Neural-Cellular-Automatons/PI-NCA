"""3-D neural architectures (Flax linen, NDHWC). 3-D analogues of the 2-D models.

- NCA3D            : plain residual 3-D NCA (no conservation)
- FluxNCA3D        : conservative scalar 3-D flux-divergence NCA (PI-NCA)
- MultiChannelFluxNCA3D : per-field conservative 3-D flux NCA (gray_scott/FHN)
- MultiScaleFluxNCA3D   : dilated multi-scale 3-D perception + conservative flux
- FNO3D            : 3-D Fourier neural operator (global spectral baseline)
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .physics3d import (multichannel_divergence_update,
                        conserve_energy_per_channel, total_mass_per_channel)

_HE = nn.initializers.he_normal()
_Z, _Y, _X = 1, 2, 3


class NCA3D(nn.Module):
    out_channels: int = 1
    perceive_features: int = 32
    hidden_features: int = 64

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Conv(self.perceive_features, (3, 3, 3), padding="CIRCULAR", kernel_init=_HE, name="perceive")(x))
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1, 1), kernel_init=_HE, name="proc1")(h))
        d = nn.Conv(self.out_channels, (1, 1, 1), use_bias=False,
                    kernel_init=nn.initializers.zeros, name="upd")(h)
        return x + d


class FluxNCA3D(nn.Module):
    """Conservative 3-D flux NCA, generic in C. Identical numerics at C == 1.

    With `bounds` set it also clips and re-projects per-field mass, giving the 3-D
    counterpart of BoundedConsFluxNCA (bounded AND conserving) with no extra class.
    """
    out_channels: int = 1
    perceive_features: int = 32
    hidden_features: int = 64
    bounds: tuple | None = None

    @nn.compact
    def __call__(self, x):
        tgt = total_mass_per_channel(x)
        h = nn.relu(nn.Conv(self.perceive_features, (3, 3, 3), padding="CIRCULAR", kernel_init=_HE, name="perceive")(x))
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1, 1), kernel_init=_HE, name="proc1")(h))
        flux = nn.Conv(3 * self.out_channels, (1, 1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux")(h)
        out = multichannel_divergence_update(x, flux)
        if self.bounds is not None:
            out = jnp.clip(out, self.bounds[0], self.bounds[1])
            out = conserve_energy_per_channel(out, tgt)
        return out


class MultiChannelFluxNCA3D(nn.Module):
    out_channels: int = 2
    perceive_features: int = 48
    hidden_features: int = 96

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Conv(self.perceive_features, (3, 3, 3), padding="CIRCULAR", kernel_init=_HE, name="perceive")(x))
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1, 1), kernel_init=_HE, name="proc1")(h))
        flux = nn.Conv(3 * self.out_channels, (1, 1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux")(h)
        return multichannel_divergence_update(x, flux)


class MultiScaleFluxNCA3D(nn.Module):
    out_channels: int = 1
    features: int = 20
    hidden_features: int = 64
    dilations: tuple = (1, 2)
    conserve: bool = True
    bounds: tuple | None = None

    @nn.compact
    def __call__(self, x):
        tgt = total_mass_per_channel(x)
        percepts = [nn.Conv(self.features, (3, 3, 3), padding="CIRCULAR", kernel_dilation=(d, d, d),
                            kernel_init=_HE, name=f"perceive_d{d}")(x) for d in self.dilations]
        h = nn.relu(jnp.concatenate(percepts, axis=-1))
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1, 1), kernel_init=_HE, name="proc1")(h))
        flux = nn.Conv(3 * self.out_channels, (1, 1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux")(h)
        out = multichannel_divergence_update(x, flux)
        if self.bounds is not None:
            out = jnp.clip(out, self.bounds[0], self.bounds[1])
        if self.conserve:
            out = conserve_energy_per_channel(out, tgt)
        return out


class SpectralConv3d(nn.Module):
    out_channels: int
    modes: int

    @nn.compact
    def __call__(self, x):
        B, D, H, W, C = x.shape
        m = self.modes
        mD, mH, mW = min(m, D // 2), min(m, H // 2), min(m, W // 2 + 1)
        scale = 1.0 / (C * self.out_channels)

        def cparam(name):
            r = self.param(name + "_r", nn.initializers.normal(scale), (2, 2, mD, mH, mW, C, self.out_channels))
            i = self.param(name + "_i", nn.initializers.normal(scale), (2, 2, mD, mH, mW, C, self.out_channels))
            return r + 1j * i

        W4 = cparam("w")  # 4 corners (±D, ±H) × low W
        x_ft = jnp.fft.rfftn(x, axes=(1, 2, 3))
        out_ft = jnp.zeros((B, D, H, W // 2 + 1, self.out_channels), dtype=x_ft.dtype)

        def mix(blk, w):
            return jnp.einsum("bdhwi,dhwio->bdhwo", blk, w)

        sl = [(slice(None, mD), slice(None, mH)), (slice(None, mD), slice(-mH, None)),
              (slice(-mD, None), slice(None, mH)), (slice(-mD, None), slice(-mH, None))]
        for idx, (sd, sh) in enumerate(sl):
            block = x_ft[:, sd, sh, :mW, :]
            out_ft = out_ft.at[:, sd, sh, :mW, :].set(mix(block, W4[idx // 2, idx % 2]))
        return jnp.fft.irfftn(out_ft, s=(D, H, W), axes=(1, 2, 3))


class FNO3D(nn.Module):
    out_channels: int = 1
    width: int = 12
    modes: int = 6
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        v = nn.Conv(self.width, (1, 1, 1), name="lift")(x)
        for d in range(self.depth):
            s = SpectralConv3d(self.width, self.modes, name=f"spec{d}")(v)
            w = nn.Conv(self.width, (1, 1, 1), name=f"w{d}")(v)
            v = nn.gelu(s + w)
        v = nn.gelu(nn.Conv(self.width, (1, 1, 1), name="proj1")(v))
        out = nn.Conv(self.out_channels, (1, 1, 1), kernel_init=nn.initializers.zeros, name="proj2")(v)
        return x + out
