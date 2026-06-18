"""Plain Neural Cellular Automaton (Flax linen) — the unconstrained baseline.

Contrast with models/flux_nca.py: the plain NCA predicts a residual *state*
increment directly (no flux/divergence conservation structure). Comparing the two
isolates the contribution of the conservation inductive bias (an ablation axis).

Emulator interface: __call__(state) -> next_state, NHWC, C = pde channels.
Optionally carries `hidden` extra channels appended to the state for richer local
memory (classic NCA); hidden=0 reproduces a stateless local update for parity
with the flux NCA.
"""
from __future__ import annotations

import flax.linen as nn
import jax

_HE = nn.initializers.he_normal()  # better start for ReLU convs (matches originals)


class NCA(nn.Module):
    out_channels: int = 1
    perceive_features: int = 48
    hidden_features: int = 64

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        p = nn.Conv(self.perceive_features, (3, 3), padding="CIRCULAR", kernel_init=_HE, name="perceive")(x)
        h = nn.relu(p)
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), kernel_init=_HE, name="proc1")(h))
        h = nn.relu(nn.Conv(self.perceive_features, (1, 1), kernel_init=_HE, name="proc2")(h))
        delta = nn.Conv(self.out_channels, (1, 1), use_bias=False,
                        kernel_init=nn.initializers.zeros, name="update")(h)
        return x + delta  # residual update; zero-init -> starts as identity
