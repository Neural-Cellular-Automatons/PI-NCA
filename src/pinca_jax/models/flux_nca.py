"""DeepFluxNCA — conservative Physics-Informed Neural Cellular Automaton (Flax linen).

Migrated from the PyTorch `DeepFluxNCA` in `PI NCA_v1.py`.

PyTorch reference (NCHW):
    perceive = Conv2d(1, 32, 3, padding=1, padding_mode="circular")   # +bias
    process  = ReLU -> Conv2d(32,64,1) -> ReLU -> Conv2d(64,32,1) -> ReLU
               -> Conv2d(32, 2, 1, bias=False)   # last layer zero-initialised
    forward(x):
        flux = process(perceive(x))
        fx, fy = flux[:,0:1], flux[:,1:2]
        dx = (roll(fx,1,W) - fx) + (roll(fy,1,H) - fy)
        return x + dx

JAX port notes:
- Channels-last (NHWC). Flax `Conv(..., padding="CIRCULAR")` reproduces the
  circular padding of the perceive conv.
- The zero-init of the final flux conv is preserved (kernel_init=zeros): the NCA
  starts as the identity map x->x, a standard NCA stabiliser.
- The divergence-flux update is factored into physics.divergence_flux_update so
  hybrids can reuse the conservation structure.
"""
from __future__ import annotations

import flax.linen as nn
import jax

from ..physics import divergence_flux_update


class DeepFluxNCA(nn.Module):
    """Conservative flux-form NCA. Input/output state: (B, H, W, 1)."""

    perceive_features: int = 32
    hidden_features: int = 64
    flux_features: int = 2  # (f_x, f_y)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Perception: learnable 3x3 circular conv (NCA "perceive" stencil).
        p = nn.Conv(self.perceive_features, (3, 3), padding="CIRCULAR", name="perceive")(x)
        # Pointwise processing MLP (1x1 convs == per-cell MLP).
        h = nn.relu(p)
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), name="proc1")(h))
        h = nn.relu(nn.Conv(self.perceive_features, (1, 1), name="proc2")(h))
        # Flux head, zero-initialised -> NCA starts as identity (no-op update).
        flux = nn.Conv(
            self.flux_features, (1, 1), use_bias=False,
            kernel_init=nn.initializers.zeros, name="flux_head",
        )(h)
        # Conservative discrete-divergence update.
        return divergence_flux_update(x, flux)
