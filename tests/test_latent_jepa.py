"""Correctness gate for the latent-FNO architecture and the JEPA pretraining protocol.

These assert the properties the comparison rests on rather than that the code runs: the
architecture satisfies the emulator interface the rest of the benchmark uses, the
collapse diagnostic actually separates a degenerate representation from a healthy one,
the decoder-only probe really freezes the encoder, and the latent objective really does
not touch the decoder.

The frozen-probe test exists because the first implementation was wrong in a way that
would have invalidated every probe number: `optax.masked` applies its inner transform to
the selected leaves and passes the remaining updates through *untouched*, so the
"frozen" encoder received its raw gradient and moved by gradient ascent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax import ic, jepa, metrics
from pinca_jax.equations import pdes
from pinca_jax.models import registry
from pinca_jax.models.latent_fno import (LatentFNOEmulator, largest_patch,
                                         latent_rollout)


# ---------------------------------------------------------------- architecture ---
@pytest.mark.parametrize("grid", [16, 24, 48])
@pytest.mark.parametrize("C", [1, 3])
def test_latent_fno_is_shape_preserving_and_starts_as_identity(grid, C):
    """The emulator interface: (B,H,W,C) -> (B,H,W,C), identity at initialisation."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, grid, grid, C))
    m = LatentFNOEmulator(out_channels=C, patch=4)
    p = m.init(key, x)
    y = m.apply(p, x)
    assert y.shape == x.shape
    assert np.allclose(np.asarray(y), np.asarray(x), atol=1e-6)


def test_latent_stages_have_the_advertised_shapes():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 48, 48, 1))
    m = LatentFNOEmulator(out_channels=1, patch=4, latent_dim=16)
    p = m.init(key, x)
    z = m.apply(p, x, method=m.encode)
    assert z.shape == (2, 12, 12, 16)
    assert m.apply(p, z, method=m.predict).shape == z.shape
    assert m.apply(p, z, method=m.decode).shape == x.shape


def test_indivisible_grid_raises_rather_than_cropping():
    """A silently cropped field would corrupt every rollout metric downstream."""
    key = jax.random.PRNGKey(0)
    m = LatentFNOEmulator(out_channels=1, patch=4)
    with pytest.raises(ValueError, match="not divisible"):
        m.init(key, jax.random.normal(key, (1, 18, 18, 1)))


def test_largest_patch_degrades_instead_of_failing():
    assert largest_patch(48, 4) == 4
    assert largest_patch(18, 4) == 3
    assert largest_patch(7, 4) == 1
    for grid in (12, 16, 24, 32, 48):      # every grid the benchmark actually uses
        assert grid % largest_patch(grid, 4) == 0


def test_latent_rollout_is_a_different_map_from_per_step_decoding():
    """They must not be pooled into one accuracy column, so they must differ.

    With a trained (non-zero) decoder, encoding once and iterating the latent is not the
    same as decoding every step, because decode(encode(x)) is not the identity.
    """
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 16, 1))
    m = LatentFNOEmulator(out_channels=1, patch=4, latent_dim=8)
    p = m.init(key, x)
    # give the decoder non-zero weights; at init it is zero and both modes are identity
    p = jax.tree_util.tree_map_with_path(
        lambda path, v: (v + 0.05 if any("decoder" in str(k) for k in path) else v), p)
    per_step = x
    for _ in range(4):
        per_step = m.apply(p, per_step)
    lat = latent_rollout(m, p, x, 4)
    assert not np.allclose(np.asarray(per_step), np.asarray(lat), atol=1e-4)


def test_registry_entries_build_and_are_budget_classified():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 48, 48, 1))
    for name in ("latent_fno", "latent_fno_iso"):
        assert name in registry.BUDGET_CLASS
        m = registry.REGISTRY[name].make(1)()
        y = m.apply(m.init(key, x), x)
        assert y.shape == x.shape
    small = registry.REGISTRY["latent_fno_iso"].make(1)()
    big = registry.REGISTRY["latent_fno"].make(1)()
    n_small = metrics.param_count(small.init(key, x))
    n_big = metrics.param_count(big.init(key, x))
    assert n_small < n_big / 10, (n_small, n_big)


# ------------------------------------------------------------------ collapse ---
def test_collapse_metric_separates_healthy_from_degenerate():
    key = jax.random.PRNGKey(0)
    d = 16
    healthy = jax.random.normal(key, (4, 6, 6, d))
    constant = jnp.ones((4, 6, 6, d))
    u = jax.random.normal(key, (4 * 6 * 6, 1))
    v = jax.random.normal(jax.random.PRNGKey(1), (1, d))
    rank_one = (u @ v).reshape(4, 6, 6, d)

    h, c, r = (jepa.collapse_metrics(z) for z in (healthy, constant, rank_one))
    assert h["eff_rank"] > 0.8
    assert c["eff_rank"] == pytest.approx(1 / d, abs=1e-6) and c["latent_std"] == 0.0
    # the case a standard-deviation check alone would miss: spread, but one direction
    assert r["latent_std"] > 0.1
    assert r["eff_rank"] == pytest.approx(1 / d, abs=1e-6)
    assert r["top1_var_frac"] > 0.99


def test_normalise_removes_the_shrink_to_zero_escape():
    """Scaling a representation down must not reduce the loss; that is the collapse route."""
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (4, 6, 6, 8))
    t = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 6, 8))
    full = jepa.latent_loss(z, t)
    shrunk = jepa.latent_loss(z * 1e-3, t)
    # Scale invariance holds up to the eps floor in `_normalise`: at a scale of 1e-3 the
    # standard deviation is also ~1e-3, so eps/sd is ~1e-3 and leaks that much through.
    # A 1% tolerance is therefore the honest bound, not 0.1%.
    assert abs(float(full) - float(shrunk)) < 1e-2 * max(1.0, float(full))


def test_variance_hinge_penalises_a_collapsed_prediction():
    key = jax.random.PRNGKey(0)
    t = jax.random.normal(key, (4, 6, 6, 8))
    collapsed = jnp.zeros((4, 6, 6, 8))
    healthy = jax.random.normal(jax.random.PRNGKey(2), (4, 6, 6, 8))
    lo = jepa.latent_loss(healthy, t, var_weight=1.0)
    hi = jepa.latent_loss(collapsed, t, var_weight=1.0)
    assert float(hi) > float(lo)


# -------------------------------------------------------------------- protocol ---
def _tiny_pretrain(**kw):
    return jepa.jepa_pretrain("heat", 16, epochs=6, batch=2, k_steps=2, preseed=0,
                              latent_dim=8, patch=4, width=8, modes=3, depth=1, **kw)


def test_latent_objective_leaves_the_decoder_at_zero():
    """The probe is only a probe if pretraining never trained the readout."""
    pre = _tiny_pretrain()
    assert pre["decoder_untouched"] is True


def test_decoder_probe_freezes_the_encoder_and_predictor():
    """The bug this pins: optax.masked would move the unmasked leaves by raw gradient."""
    pre = _tiny_pretrain()
    before = jepa._encoder_fingerprint(pre["params"])
    fit = jepa.fit_decoder(pre, "heat", 16, epochs=6, batch=2, rollout_steps=2, preseed=0)
    assert jepa._encoder_fingerprint(fit["params"]) == before
    # and the decoder did move, otherwise the probe measured nothing
    assert any(float(jnp.max(jnp.abs(v))) > 0 for v in jepa._decoder_leaves(fit["params"]))
    assert fit["trainable_params"] > 0


def test_decoder_labels_route_everything_else_to_freeze():
    pre = _tiny_pretrain()
    labels = jepa._decoder_labels(pre["params"])
    flat = jax.tree_util.tree_leaves(labels)
    assert set(flat) == {"train", "freeze"}
    assert flat.count("train") >= 1 and flat.count("freeze") >= 1


def test_degenerate_encoder_is_the_collapse_floor():
    pre = _tiny_pretrain()
    deg = jepa._degenerate(pre)
    x = ic.make_state(jax.random.PRNGKey(0), "heat", 4, 16)
    z = deg["model"].apply(deg["params"], x, method=deg["model"].encode)
    cm = jepa.collapse_metrics(z)
    assert cm["latent_std"] == pytest.approx(0.0, abs=1e-6)
    assert cm["eff_rank"] < 0.2


def test_latent_rollout_cost_is_measured_not_assumed():
    """The efficiency claim must come from a timing, and both modes must be timed."""
    pre = _tiny_pretrain()
    x = ic.make_state(jax.random.PRNGKey(0), "heat", 4, 16)
    cost = jepa.rollout_cost(pre["model"], pre["params"], x, 8, n=2)
    assert cost["per_step"] > 0 and cost["latent"] > 0
    assert cost["speedup"] == pytest.approx(cost["per_step"] / cost["latent"], rel=1e-6)
