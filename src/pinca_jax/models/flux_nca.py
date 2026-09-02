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

from ..physics import multichannel_divergence_update

# He/kaiming-normal init for ReLU layers (matches the originals' nn.init.kaiming_normal_,
# a better starting point than Flax's default lecun_normal for ReLU nets).
_HE = nn.initializers.he_normal()


class DeepFluxNCA(nn.Module):
    """Conservative flux-form NCA. State: (B, H, W, C), any C.

    Emits one (f_x, f_y) pair per channel and applies a per-channel discrete
    divergence, so EVERY field's total is conserved separately.

    At C == 1 this is numerically identical to the original scalar implementation:
    the head is still 2 channels with the same zero init, and
    `multichannel_divergence_update` reduces exactly to `divergence_flux_update`
    (same slices, same roll axes). The migration-correctness test against the
    PyTorch reference therefore still holds - see tests/test_uniform_matrix.py.
    """

    out_channels: int = 1
    perceive_features: int = 32
    hidden_features: int = 64

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Perception: learnable 3x3 circular conv (NCA "perceive" stencil).
        p = nn.Conv(self.perceive_features, (3, 3), padding="CIRCULAR", kernel_init=_HE, name="perceive")(x)
        # Pointwise processing MLP (1x1 convs == per-cell MLP).
        h = nn.relu(p)
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), kernel_init=_HE, name="proc1")(h))
        h = nn.relu(nn.Conv(self.perceive_features, (1, 1), kernel_init=_HE, name="proc2")(h))
        # Flux head, zero-initialised -> NCA starts as identity (no-op update).
        flux = nn.Conv(
            2 * self.out_channels, (1, 1), use_bias=False,
            kernel_init=nn.initializers.zeros, name="flux_head",
        )(h)
        # Conservative discrete-divergence update (per channel).
        return multichannel_divergence_update(x, flux)


class MultiChannelFluxNCA(nn.Module):
    """Conservative flux-divergence NCA for multi-field states (SWE C=3, FHN/GS C=2).

    Predicts a 2-component flux per channel and applies a per-channel discrete
    divergence, so each field's total sum is conserved on the periodic grid. Correct
    prior for periodic conservation laws (shallow-water); intentionally mismatched for
    source-term reaction systems (FHN) — a probe of when conservation helps vs hurts."""

    out_channels: int = 3
    perceive_features: int = 48
    hidden_features: int = 96

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        p = nn.Conv(self.perceive_features, (3, 3), padding="CIRCULAR", kernel_init=_HE, name="perceive")(x)
        h = nn.relu(p)
        h = nn.relu(nn.Conv(self.hidden_features, (1, 1), kernel_init=_HE, name="proc1")(h))
        h = nn.relu(nn.Conv(self.perceive_features, (1, 1), kernel_init=_HE, name="proc2")(h))
        flux = nn.Conv(2 * self.out_channels, (1, 1), use_bias=False,
                       kernel_init=nn.initializers.zeros, name="flux_head")(h)
        return multichannel_divergence_update(x, flux)
