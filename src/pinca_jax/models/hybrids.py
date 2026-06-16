"""Hybrid architectures (Flax linen), motivated by the baseline findings.

Each hybrid targets a *specific* observed failure (see docs/experimental_report.md,
docs/ablation_report.md):

1. BoundedConsFluxNCA  — resolves the stability↔conservation tension: the naïve
   output clip that recovered stiff-PDE accuracy DESTROYED the flux NCA's exact
   mass conservation. Here we clip AND re-project total mass, so the update is both
   bounded (stable) and mass-conserving.

2. SpectralFluxNCA  — the central hybrid hypothesis: combine the FNO's global
   spectral mixing (fixes locality's slow information propagation) with the NCA's
   local conservative flux-divergence. A "global operator + local conservation" model.

3. MultiScaleFluxNCA — widen the receptive field per step via dilated perception
   (1,2,4) without a full FFT — a cheaper middle ground between pure-local NCA and
   global FNO.

All operate on NHWC, C = pde channels (C=1 conservative variants for scalar fields).
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..physics import divergence_flux_update, conserve_energy, total_mass
from .fno import SpectralConv2d


class BoundedConsFluxNCA(nn.Module):
    """Flux-divergence NCA + (clip → mass re-projection): bounded AND mass-conserving."""
    bounds: tuple = (-1.0, 1.0)
    perceive_features: int = 32
    hidden_features: int = 64

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tgt = total_mass(x)  # conserve this step's total mass
        p = nn.Conv(self.perceive_features, (3, 3), padding="CIRCULAR", name="perceive")(x)
        h = nn.relu(p)
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), name="proc1")(h))
        h = nn.relu(nn.Conv(self.perceive_features, (1, 1), name="proc2")(h))
        flux = nn.Conv(2, (1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux_head")(h)
        x = divergence_flux_update(x, flux)        # conserves mass
        x = jnp.clip(x, self.bounds[0], self.bounds[1])  # bounds (breaks conservation)
        x = conserve_energy(x, tgt)                # restores exact total mass
        return x


class SpectralFluxNCA(nn.Module):
    """Local conservative flux-divergence + global FNO spectral correction.

    update = divergence(flux_local)  +  spectral_global(x)
    The local term conserves mass; the spectral term supplies global reach. With
    `conserve=True` the global term is mass-projected so the whole update stays
    mass-conserving (test both)."""
    out_channels: int = 1  # scalar conservative field
    perceive_features: int = 32
    hidden_features: int = 64
    width: int = 16
    modes: int = 8
    spectral_depth: int = 2
    conserve: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tgt = total_mass(x)
        # local conservative stream → flux divergence (mass-conserving)
        h = nn.relu(nn.Conv(self.perceive_features, (3, 3), padding="CIRCULAR", name="perceive")(x))
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), name="proc1")(h))
        flux = nn.Conv(2, (1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux_head")(h)
        x_local = divergence_flux_update(x, flux)
        # global spectral stream
        v = nn.Conv(self.width, (1, 1), name="lift")(x)
        for d in range(self.spectral_depth):
            s = SpectralConv2d(self.width, self.modes, self.modes, name=f"spec{d}")(v)
            w = nn.Conv(self.width, (1, 1), name=f"w{d}")(v)
            v = nn.gelu(s + w)
        g = nn.Conv(self.out_channels, (1, 1), name="proj",
                    kernel_init=nn.initializers.zeros)(v)
        out = x_local + g
        if self.conserve:
            out = conserve_energy(out, tgt)
        return out


class MultiScaleFluxNCA(nn.Module):
    """Dilated multi-scale perception (1,2,4) + conservative flux-divergence."""
    out_channels: int = 1
    features: int = 24
    hidden_features: int = 64
    dilations: tuple = (1, 2, 4)
    conserve: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tgt = total_mass(x)
        percepts = [
            nn.Conv(self.features, (3, 3), padding="CIRCULAR",
                    kernel_dilation=(d, d), name=f"perceive_d{d}")(x)
            for d in self.dilations
        ]
        h = nn.relu(jnp.concatenate(percepts, axis=-1))
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), name="proc1")(h))
        flux = nn.Conv(2, (1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux_head")(h)
        out = divergence_flux_update(x, flux)
        if self.conserve:
            out = conserve_energy(out, tgt)
        return out
