"""FINN-style finite-volume neural network: the closest published baseline to PI-NCA.

FINN \\citep{praditia2021finn, karlbauer2022composing} learns the *interface* fluxes of a
finite-volume discretisation. The distinction from PI-NCA is where the learned function
lives and what it sees:

* PI-NCA learns a **cellwise** rule. One shared convolutional rule reads a cell's
  neighbourhood and emits that cell's outgoing flux per axis.
* FINN learns a **facewise** rule. For each face between two cells it applies a shared
  network to the pair of adjacent states, splitting the flux into a diffusive part
  proportional to the state difference across the face and an advective part carried by
  the upwind cell. The same face flux then enters both neighbours with opposite sign.

Both are conservative by construction -- the same telescoping argument applies -- so this
is a like-for-like conservative baseline rather than an unconstrained one, and it is the
model a reviewer will name when asking whether the flux idea needs a cellular automaton
at all.

What this is not: the published FINN assembles equation-specific modules (retardation
factors for solute transport, and so on) and is usually fitted to one system at a time.
This is a single re-usable FINN-style module applied unchanged to every phenomenon in the
benchmark, so it is labelled a FINN-*style* baseline throughout and is not presented as a
reproduction of the authors' numbers.

The optional `source` term is a learned reaction added to the divergence. It is off by
default, because with it the model no longer conserves anything; it exists so the same
architecture can be given to the reaction systems, where conservation is the wrong prior.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..physics import multichannel_divergence_update

_HE = nn.initializers.he_normal()


class FINN2d(nn.Module):
    """Facewise conservative flux network, 2-D periodic grid, NHWC.

    For axis `a`, the face between cell `i` and its neighbour `i+1` carries
    ``F = D([u_i, u_j]) * (u_j - u_i) + A(u_i)``, and the update is the periodic
    divergence of those face fluxes. `D` and `A` are pointwise MLPs shared by every face
    on that axis, so the model is translation-equivariant and independent of grid size.
    """
    out_channels: int = 1
    hidden: int = 32
    depth: int = 2
    source: bool = False

    def _mlp(self, z, n_out, tag):
        for i in range(self.depth):
            z = nn.gelu(nn.Conv(self.hidden, (1, 1), kernel_init=_HE,
                                name=f"{tag}_h{i}")(z))
        return nn.Conv(n_out, (1, 1), name=f"{tag}_out", use_bias=False,
                       kernel_init=nn.initializers.zeros)(z)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        C = self.out_channels
        faces = []
        for a, axis in enumerate((2, 1)):                  # x faces, then y faces
            nbr = jnp.roll(x, -1, axis=axis)               # state on the far side
            pair = jnp.concatenate([x, nbr], axis=-1)
            # Diffusive: a learned, state-dependent conductance times the jump across the
            # face. Writing it this way means zero jump gives zero flux, which is what
            # makes a constant state an exact fixed point of the diffusive term.
            diff = self._mlp(pair, C, f"diff{a}") * (nbr - x)
            # Advective: carried by the upwind cell only, so the face flux is not
            # symmetric in the two neighbours.
            adv = self._mlp(x, C, f"adv{a}")
            faces.append(diff + adv)
        # Interleave to the [fx_0, fy_0, fx_1, fy_1, ...] layout the divergence expects.
        flux = jnp.stack(faces, axis=-1).reshape(x.shape[:-1] + (2 * C,))
        out = multichannel_divergence_update(x, flux)
        if self.source:
            out = out + self._mlp(x, C, "src")
        return out


def demo():
    import numpy as np
    key = jax.random.PRNGKey(0)
    for C in (1, 3):
        x = jax.random.normal(key, (2, 16, 16, C))
        m = FINN2d(out_channels=C)
        p = m.init(key, x)
        y = m.apply(p, x)
        assert y.shape == x.shape, (C, y.shape)
        assert np.allclose(np.asarray(y), np.asarray(x), atol=1e-6)   # identity at init
        # conservation for arbitrary weights, which is the property under test
        p2 = jax.tree_util.tree_map(lambda v: v + 0.05, p)
        drift = np.abs(np.asarray(m.apply(p2, x).sum((1, 2)) - x.sum((1, 2)))).max()
        assert drift < 1e-3, (C, drift)
        # a constant field must stay constant under the diffusive term alone
        const = jnp.ones((1, 8, 8, C)) * 0.3
        pc = m.init(key, const)
        pc = jax.tree_util.tree_map(
            lambda v: v + 0.05, pc)
        y_const = m.apply(pc, const)
        assert np.allclose(np.asarray(y_const - const),
                           np.asarray(y_const - const).mean(), atol=1e-5)
        # the source term breaks conservation on purpose
        ms = FINN2d(out_channels=C, source=True)
        ps = jax.tree_util.tree_map(lambda v: v + 0.05, ms.init(key, x))
        assert np.abs(np.asarray(ms.apply(ps, x).sum((1, 2)) - x.sum((1, 2)))).max() > 1e-3
    print("finn.demo OK")


if __name__ == "__main__":
    demo()
