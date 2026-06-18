"""Configurable NCA for controlled ablations A4 (conservation on/off) and A5
(neighbourhood / receptive-field size). One backbone, switchable head and perception,
so variants differ in exactly one factor."""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..physics import divergence_flux_update, conserve_energy, total_mass

_HE = nn.initializers.he_normal()


class AblationNCA(nn.Module):
    out_channels: int = 1
    perceive_features: int = 32
    hidden_features: int = 64
    kernel: int = 3
    dilations: tuple = (1,)          # A5: (1,) local; (1,2,4) multi-scale
    head: str = "flux"               # A4: "flux" (conserves mass) vs "residual" (free)
    conserve_proj: bool = False      # extra energy projection on top

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tgt = total_mass(x)
        percepts = [
            nn.Conv(self.perceive_features, (self.kernel, self.kernel), padding="CIRCULAR",
                    kernel_dilation=(d, d), kernel_init=_HE, name=f"perceive_d{d}")(x)
            for d in self.dilations
        ]
        h = nn.relu(jnp.concatenate(percepts, axis=-1) if len(percepts) > 1 else percepts[0])
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), kernel_init=_HE, name="proc1")(h))
        h = nn.relu(nn.Conv(self.perceive_features, (1, 1), kernel_init=_HE, name="proc2")(h))
        if self.head == "flux":
            flux = nn.Conv(2, (1, 1), use_bias=False,
                           kernel_init=nn.initializers.zeros, name="flux_head")(h)
            out = divergence_flux_update(x, flux)
        else:  # residual
            delta = nn.Conv(self.out_channels, (1, 1), use_bias=False,
                            kernel_init=nn.initializers.zeros, name="upd")(h)
            out = x + delta
        if self.conserve_proj:
            out = conserve_energy(out, tgt)
        return out
