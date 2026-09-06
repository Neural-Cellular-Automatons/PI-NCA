"""Conservation operators shared across models (JAX).

Migrated from `PI NCA_v1.py`:
- `conserve_energy` (mass/energy projection onto a fixed total).
- the discrete-divergence flux update used by DeepFluxNCA, exposed standalone so
  hybrid models can reuse the conservation structure.

Channels-last (NHWC) to match Flax convs: arrays are (B, H, W, C).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def total_mass(u: jax.Array) -> jax.Array:
    """Sum over spatial+channel axes (H, W, C), keepdims → (B,1,1,1)."""
    return u.sum(axis=(1, 2, 3), keepdims=True)


def conserve_energy(u: jax.Array, target_sum: jax.Array) -> jax.Array:
    """Project `u` so its total mass equals `target_sum`.

    PyTorch reference distributed the deficit equally over cells, dividing by
    `u.shape[-1]**2` (valid only for square H==W in NCHW). Here, in NHWC, we
    divide by the true number of spatial cells H*W so it is correct for any shape
    (identical to the reference when H==W).
    """
    n_cells = u.shape[1] * u.shape[2]
    diff = (target_sum - total_mass(u)) / n_cells
    return u + diff


def divergence_flux_update(x: jax.Array, flux: jax.Array) -> jax.Array:
    """Apply a discrete divergence of a 2-channel flux field as a state increment.

    `flux[..., 0]` is the x-flux (width axis=2), `flux[..., 1]` the y-flux
    (height axis=1). Backward-difference divergence, matching the PyTorch:
        dx = (roll(fx,1,W) - fx) + (roll(fy,1,H) - fy)
    Returns `x + dx`. This makes the update a discrete conservation law: the
    net increment summed over a periodic grid is exactly zero (telescoping),
    so mass is conserved up to floating point.
    """
    fx = flux[..., 0:1]
    fy = flux[..., 1:2]
    dx = (jnp.roll(fx, 1, axis=2) - fx) + (jnp.roll(fy, 1, axis=1) - fy)
    return x + dx


def multichannel_divergence_update(x: jax.Array, flux: jax.Array) -> jax.Array:
    """Per-channel discrete-divergence update for multi-field states.

    x: (B,H,W,C); flux: (B,H,W,2C) as [fx_0,fy_0, fx_1,fy_1, ...]. Each channel gets
    its own conservative flux-divergence increment, so EACH channel's total sum is
    conserved on a periodic grid (telescoping). Physically correct on a periodic
    domain for conserved quantities (e.g. shallow-water mass+momentum); a deliberately
    *wrong* prior for reaction systems with source terms (FitzHugh-Nagumo) — which is
    itself a test of when the conservation bias helps vs hurts.
    """
    B, H, W, C = x.shape
    f = flux.reshape(B, H, W, C, 2)
    fx, fy = f[..., 0], f[..., 1]  # (B,H,W,C)
    dx = (jnp.roll(fx, 1, axis=2) - fx) + (jnp.roll(fy, 1, axis=1) - fy)
    return x + dx


def total_mass_per_channel(u: jax.Array) -> jax.Array:
    """Sum over spatial axes only, per channel, keepdims -> (B,1,1,C).

    `total_mass` lumps all channels into one number, which is correct for a scalar
    field but wrong for a multi-field state: projecting against it would let one
    field's deficit be paid out of another's. Multi-channel models must conserve
    each field separately.  Identical to `total_mass` when C == 1.
    """
    return u.sum(axis=(1, 2), keepdims=True)


def conserve_energy_per_channel(u: jax.Array, target_sum: jax.Array) -> jax.Array:
    """Project each channel of `u` so that channel's total equals `target_sum`.

    `target_sum` is (B,1,1,C) from `total_mass_per_channel`. Identical to
    `conserve_energy` when C == 1.
    """
    n_cells = u.shape[1] * u.shape[2]
    diff = (target_sum - total_mass_per_channel(u)) / n_cells
    return u + diff


def conserve_energy_bounded(u, target_sum, lo, hi, eps=1e-12):
    """Project `u` onto {sum == target_sum} WITHOUT leaving [lo, hi].

    The uniform projection in `conserve_energy_per_channel` adds the same offset to
    every cell, so applying it after a clip pushes cells straight back out of the
    range the clip just enforced -- the model is then neither bounded nor, in any
    useful sense, both. Measured on Cahn-Hilliard this costs a few percent of cells,
    which is exactly the property the bounded variants exist to provide.

    This projection distributes the mass deficit in proportion to each cell's remaining
    *headroom* (`hi - u` when adding, `u - lo` when removing). Mass is then restored
    exactly and no cell can cross the bound, because the total increment is capped by
    the total headroom by construction. It is one vectorised pass, differentiable, and
    reduces to the uniform projection when every cell has equal headroom.

    Feasibility: a target mass outside [n_cells*lo, n_cells*hi] cannot be reached inside
    the box. In that case the field saturates at the bound and mass is *not* restored --
    the correct behaviour, since the alternative is silently violating the bound. Use
    `projection_residual` to measure how often that happens rather than assuming it does
    not.
    """
    n_cells = u.shape[1] * u.shape[2]
    deficit = target_sum - total_mass_per_channel(u)          # (B,1,1,C)
    head_up = jnp.sum(hi - u, axis=(1, 2), keepdims=True)
    head_dn = jnp.sum(u - lo, axis=(1, 2), keepdims=True)
    cap = jnp.where(deficit >= 0, head_up, head_dn) + eps
    share = jnp.where(deficit >= 0, hi - u, u - lo) / cap      # sums to 1 per (B,C)
    # scale <= 1 keeps every cell inside the box; it only binds when the target is
    # infeasible for this box, which is reported rather than hidden.
    scale = jnp.minimum(jnp.abs(deficit) / cap, 1.0) * jnp.sign(deficit)
    out = u + scale * cap * share
    return jnp.clip(out, lo, hi)                               # float-error guard only


def projection_residual(u, target_sum):
    """Relative mass error left after a projection -> (B,1,1,C). 0 = exactly conserved."""
    scale = jnp.abs(target_sum) + 1e-12
    return jnp.abs(total_mass_per_channel(u) - target_sum) / scale


def demo():
    import numpy as np
    key = jax.random.PRNGKey(0)
    lo, hi = -1.0, 1.0
    u = jnp.clip(jax.random.normal(key, (4, 8, 8, 2)), lo, hi)
    tgt = total_mass_per_channel(u) + jnp.array([[[[3.0, -2.5]]]])   # feasible shift
    v = conserve_energy_bounded(u, tgt, lo, hi)
    assert float(jnp.max(v)) <= hi + 1e-6 and float(jnp.min(v)) >= lo - 1e-6
    assert float(jnp.max(projection_residual(v, tgt))) < 1e-4, "mass not restored"
    # the uniform projection is the thing this replaces: it leaves the box
    w = conserve_energy_per_channel(u, tgt)
    assert float(jnp.max(w)) > hi, "uniform projection unexpectedly stayed in bounds"
    # infeasible target saturates at the bound instead of violating it
    huge = jnp.full_like(tgt, 1e6)
    z = conserve_energy_bounded(u, huge, lo, hi)
    assert float(jnp.max(z)) <= hi + 1e-6 and float(jnp.min(z)) >= lo - 1e-6
    assert float(jnp.min(projection_residual(z, huge))) > 0.0    # honestly not conserved
    # flux-divergence update conserves mass on a periodic grid
    x = jax.random.normal(key, (2, 6, 6, 1))
    f = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 6, 2))
    assert np.allclose(np.asarray(total_mass(divergence_flux_update(x, f))),
                       np.asarray(total_mass(x)), atol=1e-4)
    print("physics.demo OK")


if __name__ == "__main__":
    demo()
