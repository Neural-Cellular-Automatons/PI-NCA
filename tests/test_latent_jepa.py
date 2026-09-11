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

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax import ic, jepa, metrics
from pinca_jax.equations import pdes
from pinca_jax.models import registry
from pinca_jax.models.latent_fno import (PREDICTORS, LatentFNOEmulator, largest_patch,
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


# ------------------------------------------------------------- latent predictors ---
@pytest.mark.parametrize("name", sorted(PREDICTORS))
def test_every_latent_predictor_satisfies_the_interface_and_starts_as_identity(name):
    """Swapping the latent predictor must not change the contract the harness relies on."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 24, 24, 1))
    m = LatentFNOEmulator(out_channels=1, patch=4, latent_dim=8, width=8, modes=3,
                          depth=2, predictor=name)
    p = m.init(key, x)
    assert m.apply(p, x).shape == x.shape
    assert np.allclose(np.asarray(m.apply(p, x)), np.asarray(x), atol=1e-6)
    z = m.apply(p, x, method=m.encode)
    assert m.apply(p, z, method=m.predict).shape == z.shape
    # the frozen-probe machinery selects by parameter path, so the name must not drift
    assert any("predictor" in str(k) for path, _ in
               jax.tree_util.tree_flatten_with_path(p)[0] for k in path)


def test_unknown_predictor_raises():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="unknown latent predictor"):
        LatentFNOEmulator(predictor="nope").init(
            key, jax.random.normal(key, (1, 16, 16, 1)))


def test_full_bandwidth_modes_resolve_against_the_latent_grid():
    """`modes="full"` must become a concrete count, not reach the module as a string."""
    m = jepa._make_model("heat", 48, 8, 4, 8, "full", 2)
    assert m.modes == 6                      # latent grid 12 -> 12//2
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 48, 48, 1))
    assert m.apply(m.init(key, x), x).shape == x.shape


# -------------------------------------------------------------------- objectives ---
def test_the_three_objectives_coincide_at_a_single_step():
    """At k_steps=1 there is nothing to compose, so all three must be the same objective.

    This is the invariant that pins the composition logic: if `multi` or `final` differed
    from `oneshot` here, the difference would be in the plumbing rather than in the
    horizon, and the variant comparison would be measuring a bug.
    """
    out = {}
    for obj in ("oneshot", "final", "multi"):
        pre = jepa.jepa_pretrain("heat", 16, epochs=2, batch=2, k_steps=1, preseed=0,
                                 latent_dim=8, patch=4, width=8, modes=3, depth=1,
                                 objective=obj, seed=3)
        out[obj] = pre["losses"]
    assert out["oneshot"] == pytest.approx(out["final"], rel=1e-5)
    assert out["oneshot"] == pytest.approx(out["multi"], rel=1e-5)


def test_multi_step_objective_differs_from_oneshot_at_a_real_horizon():
    """And at k_steps>1 they must genuinely differ -- one is a k-step map, one is not."""
    kw = dict(epochs=3, batch=2, k_steps=4, preseed=0, latent_dim=8, patch=4, width=8,
              modes=3, depth=1, seed=3)
    a = jepa.jepa_pretrain("heat", 16, objective="oneshot", **kw)["losses"]
    b = jepa.jepa_pretrain("heat", 16, objective="multi", **kw)["losses"]
    assert not np.allclose(a, b, rtol=1e-3)


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="unknown objective"):
        jepa.jepa_pretrain("heat", 16, epochs=1, batch=2, k_steps=1, preseed=0,
                           latent_dim=8, patch=4, width=8, modes=3, depth=1,
                           objective="nope")


def test_multi_objective_still_leaves_the_decoder_untouched():
    """The probe protocol must survive the objective change, or probe numbers are void."""
    pre = jepa.jepa_pretrain("heat", 16, epochs=3, batch=2, k_steps=3, preseed=0,
                             latent_dim=8, patch=4, width=8, modes=3, depth=1,
                             objective="multi")
    assert pre["decoder_untouched"] is True


# ----------------------------------------------------------------------- variants ---
def test_every_variant_spec_is_buildable_and_names_a_known_axis():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 24, 24, 1))
    for name in jepa.VARIANTS:
        arch, obj, note = jepa.variant_spec(name)
        assert note, f"{name} has no stated reason to exist"
        assert obj["objective"] in ("oneshot", "final", "multi")
        m = jepa._make_model("heat", 24, **arch)
        assert m.apply(m.init(key, x), x).shape == x.shape


def test_variants_sharing_an_architecture_share_its_key():
    """The saving that makes the sweep affordable: one control per architecture."""
    keys = {n: jepa.arch_key(jepa.variant_spec(n)[0]) for n in jepa.VARIANTS}
    fno_objective_variants = ["fno_oneshot", "fno_final", "fno_multi", "fno_simsiam",
                              "fno_vicreg", "fno_cosine"]
    assert len({keys[n] for n in fno_objective_variants}) == 1
    # and the architectural variants must NOT collide with it
    for n in ("nca_multi", "flux_multi", "fno_nopatch", "fno_fullband"):
        assert keys[n] != keys["fno_multi"], n


def test_collapse_floor_does_not_depend_on_the_predictor():
    """Zeroing the encoder feeds every predictor the same zeros, so the floor is shared.

    Asserted rather than assumed, because the sweep reports one floor per architecture
    and a predictor that broke this (a non-residual one, say) would make those rows
    silently incomparable.
    """
    x = ic.make_state(jax.random.PRNGKey(0), "heat", 2, 16)
    outs = []
    for name in sorted(PREDICTORS):
        arch = dict(latent_dim=8, patch=4, width=8, modes=3, depth=1, predictor=name)
        deg = jepa._degenerate(jepa._fresh("heat", 16, arch, 0))
        outs.append(np.asarray(deg["model"].apply(deg["params"], x)))
    for o in outs[1:]:
        assert np.allclose(outs[0], o, atol=1e-6)


def _sweep_kw(tmp_path, **over):
    kw = dict(grid=12, seeds=(0,), epochs=2, jepa_epochs=2, probe_epochs=2, batch=2,
              rollout=2, eval_steps=4, n_eval=2, with_ft=False,
              out_path=str(tmp_path / "sw.json"), verbose=False)
    kw.update(over)
    return kw


def test_sweep_shares_controls_and_renders(tmp_path):
    sw = jepa.sweep("heat", variants=["fno_multi", "fno_cosine", "nca_multi"],
                    **_sweep_kw(tmp_path))
    assert set(sw["variants"]) == {"fno_multi", "fno_cosine", "nca_multi"}
    # two variants share the fno architecture, the third does not: two controls, not three
    assert len(sw["controls"]) == 2
    for v in sw["variants"].values():
        assert v["arch_key"] in sw["controls"]
        assert v["ft"] is None                      # with_ft=False
        assert v["probe"]["rel_l2"]["mean"] > 0
    md = jepa.sweep_markdown("heat", sw)
    assert "Does pretraining pay?" not in md        # no ft regime was run
    assert "Did the representation learn anything?" in md
    assert "`fno_multi`" in md and "floor" in md


def test_sweep_resumes_finished_variants_and_refuses_a_changed_condition(tmp_path):
    kw = _sweep_kw(tmp_path)
    jepa.sweep("heat", variants=["fno_multi"], **kw)
    with open(kw["out_path"], encoding="utf-8") as f:
        first = json.load(f)
    # same conditions: the finished variant is reused verbatim
    again = jepa.sweep("heat", variants=["fno_multi"], **kw)
    assert (again["variants"]["fno_multi"]["probe"]["rel_l2"]["mean"]
            == first["variants"]["fno_multi"]["probe"]["rel_l2"]["mean"])
    # a different scale must NOT be mixed into the same file
    fresh = jepa.sweep("heat", variants=["fno_multi"], **_sweep_kw(tmp_path, grid=16))
    assert fresh["cond"]["grid"] == 16
    assert (fresh["variants"]["fno_multi"]["probe"]["rel_l2"]["mean"]
            != first["variants"]["fno_multi"]["probe"]["rel_l2"]["mean"])


def test_finetune_uses_the_same_recipe_as_its_own_control(tmp_path):
    """`jepa_ft` must differ from `distill` only in where the weights came from.

    The first implementation hand-rolled the fine-tuning loop and so differed from the
    control in three ways at once: no LR warmup, no divergence guard in the training
    rollout, and a Python epoch loop rather than one jitted scan. That made the headline
    comparison partly a comparison of two training recipes. It now delegates to
    `train_emulator`, and this pins both halves of that: the shared function is used, and
    it really does start from the supplied weights.
    """
    from pinca_jax.harness import train_emulator

    pre = _tiny_pretrain()
    cfg = jepa._cfg("heat", 16, 0, 2, 4, 2, 2, 0)          # zero epochs: plumbing only
    # `train_emulator` donates its parameter buffers, so the caller must hand it a copy;
    # this is the assertion that caught it deleting the pretrained weights in place.
    copy = jax.tree_util.tree_map(jnp.copy, pre["params"])
    out = train_emulator(lambda: pre["model"], cfg, init_params=copy)
    for a, b in zip(jax.tree_util.tree_leaves(out["params"]),
                    jax.tree_util.tree_leaves(pre["params"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    ft = jepa._finetune(pre, cfg)
    assert ft["model"] is pre["model"]
    for a, b in zip(jax.tree_util.tree_leaves(ft["params"]),
                    jax.tree_util.tree_leaves(pre["params"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # and the pretrained weights survive fine-tuning, because the regimes that run after
    # it in `study` (the probe's floor) read them
    assert jepa._encoder_fingerprint(pre["params"])
