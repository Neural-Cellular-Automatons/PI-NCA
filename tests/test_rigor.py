"""Correctness gate for the statistical / generalisation / stability machinery.

These assert the properties the paper's claims rest on, not just that code runs:
the bootstrap covers, the paired test is paired, Holm actually corrects, the OOD
shifts really are different distributions, the closed-form solvers are the closed
forms, and the bounded projection is bounded.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax import ic, metrics, ood, physics, stats, teacher_error
from pinca_jax.equations import pdes
from pinca_jax.models import registry


# ------------------------------------------------------------------ stats ---
def test_bootstrap_ci_covers_true_mean():
    """A 95% interval must cover the truth ~95% of the time, not by construction."""
    rng = np.random.default_rng(0)
    covered = 0
    trials = 200
    for i in range(trials):
        s = stats.summarize(rng.normal(0.0, 1.0, 40), seed=i)
        covered += (s.ci_lo <= 0.0 <= s.ci_hi)
    assert 0.88 <= covered / trials <= 1.0, covered / trials


def test_summarize_single_value_is_degenerate_not_wrong():
    s = stats.summarize([3.0])
    assert s.n == 1 and s.std == 0.0 and s.ci_lo == s.ci_hi == 3.0


def test_paired_test_detects_a_real_difference_and_ignores_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.3, 60)
    worse = stats.paired_test(a, a + 0.5, lower_is_better=True)
    assert worse["better"] == "a" and worse["significant"] and worse["win_rate"] == 1.0
    noise = stats.paired_test(a, a + rng.normal(0, 1e-9, 60))
    assert noise["better"] == "tie"


def test_paired_test_rejects_unpaired_input():
    with pytest.raises(ValueError):
        stats.paired_test([1.0, 2.0], [1.0, 2.0, 3.0])


def test_holm_is_more_conservative_than_uncorrected():
    p = [0.02, 0.03, 0.04, 0.045]
    assert sum(stats.holm_bonferroni(p)) < sum(x < 0.05 for x in p)


def test_holm_step_down_stops_at_first_failure():
    """Ordering is by p-value, not by position: the 0.9 blocks nothing below it."""
    assert stats.holm_bonferroni([0.001, 0.9, 0.001]) == [True, False, True]
    # ...but a mid-sized p that fails its own threshold does block everything after it
    assert stats.holm_bonferroni([0.001, 0.03, 0.04]) == [True, False, False]


def test_compare_archs_picks_the_best_reference():
    rng = np.random.default_rng(2)
    base = rng.normal(1.0, 0.1, 40)
    out = stats.compare_archs({"a": list(base), "b": list(base + 0.4),
                               "c": list(base + 0.8)}, lower_is_better=True)
    assert out["reference"] == "a"
    assert out["comparisons"]["c"]["holm_significant"]


# ---------------------------------------------------------------- metrics ---
def test_rel_l2_per_sample_matches_the_batch_metric_on_a_uniform_batch():
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 8, 8, 1))
    y = x * 1.1
    per = metrics.rel_l2_per_sample(y, x)
    assert len(per) == 5
    assert np.allclose(per, metrics.rel_l2(y, x), rtol=1e-4)


def test_rel_l2_per_sample_separates_samples():
    x = jnp.ones((3, 4, 4, 1))
    y = x.at[1].add(1.0)
    per = metrics.rel_l2_per_sample(y, x)
    assert per[0] < 1e-6 and per[2] < 1e-6 and per[1] > 0.9


def test_conservation_drift_is_flat_for_a_conserving_solver():
    x0 = ic.make_state(jax.random.PRNGKey(0), "heat", 3, 16)
    traj = pdes.rollout_trajectory(pdes.REGISTRY["heat"], x0, 20)
    assert max(metrics.conservation_drift(traj, x0)) < 1e-4


def test_conservation_error_per_channel_is_not_lumped():
    """A state that gains mass in one channel and loses it in another is NOT conserved."""
    x0 = jnp.ones((2, 4, 4, 2))
    pred = x0.at[..., 0].add(1.0).at[..., 1].add(-1.0)
    assert metrics.conservation_error(pred, x0) < 1e-6           # lumped total: looks fine
    per = metrics.conservation_error_per_channel(pred, x0)
    assert min(per) > 10.0                                       # per channel: caught


# ------------------------------------------------------------------- OOD ---
@pytest.mark.parametrize("pde", list(pdes.REGISTRY))
def test_every_ood_shift_is_finite_and_correctly_shaped(pde):
    k = jax.random.PRNGKey(0)
    C = pdes.REGISTRY[pde].channels
    for name in ood.IC_SHIFTS:
        x = ood.shifted_state(k, pde, 2, 16, name)
        if x is None:
            continue
        assert x.shape == (2, 16, 16, C), (pde, name, x.shape)
        assert bool(jnp.all(jnp.isfinite(x))), (pde, name)


def test_ood_shifts_actually_shift_the_distribution():
    """A 'held-out' axis that produces the training distribution is not a test set."""
    k = jax.random.PRNGKey(0)
    base = ood.shifted_state(k, "heat", 8, 24, "in_dist")
    for name in ("amp_2x", "amp_half", "blobs_many", "scale_wide", "spectrum_rough"):
        x = ood.shifted_state(k, "heat", 8, 24, name)
        assert not np.allclose(np.asarray(x), np.asarray(base), atol=1e-3), name


def test_phase_shift_preserves_the_field_up_to_translation():
    """The phase axis is a negative control: it must be a pure torus translation."""
    k = jax.random.PRNGKey(0)
    base = ood.shifted_state(k, "heat", 4, 16, "in_dist")
    rolled = ood.shifted_state(k, "heat", 4, 16, "phase")
    assert np.isclose(float(jnp.sum(base)), float(jnp.sum(rolled)), rtol=1e-5)
    assert np.isclose(float(jnp.std(base)), float(jnp.std(rolled)), rtol=1e-5)


def test_coefficient_shift_changes_the_dynamics():
    x0 = ic.make_state(jax.random.PRNGKey(0), "heat", 2, 16)
    a = pdes.rollout(ood._spec_for("heat"), x0, 10)
    b = pdes.rollout(ood._spec_for("heat", 2.0), x0, 10)
    assert metrics.rel_l2(b, a) > 1e-2


def test_shallow_water_amplitude_shift_keeps_depth_positive():
    x = ood.shifted_state(jax.random.PRNGKey(0), "shallow_water", 4, 16, "amp_4x")
    assert float(jnp.min(x[..., 0])) > 0.0


# -------------------------------------------------------- teacher error ---
def test_analytic_solution_is_the_identity_at_t_zero():
    for pde in teacher_error.ANALYTIC:
        x = ic.make_state(jax.random.PRNGKey(0), pde, 2, 16)
        assert metrics.rel_l2(teacher_error.analytic_solution(pde, x, 0.0), x) < 1e-5


def test_refining_dt_converges_to_the_semidiscrete_solution():
    """Forward Euler must converge to the exactly-integrated discretised system."""
    u0 = ic.make_state(jax.random.PRNGKey(0), "heat", 2, 16)
    spec = pdes.REGISTRY["heat"]
    t = spec.params["dt"] * 8
    semi = teacher_error.analytic_solution("heat", u0, t, discrete=True)
    coarse = metrics.rel_l2(pdes.rollout(spec, u0, 8), semi)
    fine_spec = pdes.PDESpec("heat", 1, spec.step, {**spec.params, "dt": spec.params["dt"] / 16}, True)
    fine = metrics.rel_l2(pdes.rollout(fine_spec, u0, 128), semi)
    assert fine < coarse / 5


def test_spatial_error_is_a_floor_that_dt_cannot_cross():
    """The claim the paper makes about the teacher: the stencil error does not vanish."""
    u0 = ic.make_state(jax.random.PRNGKey(0), "heat", 2, 16)
    spec = pdes.REGISTRY["heat"]
    t = spec.params["dt"] * 8
    semi = teacher_error.analytic_solution("heat", u0, t, discrete=True)
    exact = teacher_error.analytic_solution("heat", u0, t, discrete=False)
    fine_spec = pdes.PDESpec("heat", 1, spec.step, {**spec.params, "dt": spec.params["dt"] / 64}, True)
    temporal = metrics.rel_l2(pdes.rollout(fine_spec, u0, 512), semi)
    assert metrics.rel_l2(semi, exact) > 10 * temporal


def test_self_convergence_reports_first_order_for_forward_euler():
    r = teacher_error.self_convergence("nagumo", 16, 16, batch=2)
    assert 0.6 < r["observed_order"] < 1.6, r["observed_order"]


# ------------------------------------------------ bounded mass projection ---
def test_headroom_projection_conserves_mass_without_leaving_the_box():
    key = jax.random.PRNGKey(0)
    lo, hi = -1.0, 1.0
    u = jnp.clip(jax.random.normal(key, (4, 8, 8, 2)), lo, hi)
    tgt = physics.total_mass_per_channel(u) + 3.0
    v = physics.conserve_energy_bounded(u, tgt, lo, hi)
    assert float(jnp.max(v)) <= hi + 1e-6 and float(jnp.min(v)) >= lo - 1e-6
    assert float(jnp.max(physics.projection_residual(v, tgt))) < 1e-4


def test_uniform_projection_leaves_the_box_which_is_why_headroom_exists():
    key = jax.random.PRNGKey(0)
    u = jnp.clip(jax.random.normal(key, (4, 8, 8, 2)), -1.0, 1.0)
    tgt = physics.total_mass_per_channel(u) + 3.0
    w = physics.conserve_energy_per_channel(u, tgt)
    assert float(jnp.max(w)) > 1.0


def test_infeasible_target_saturates_rather_than_violating_the_bound():
    key = jax.random.PRNGKey(0)
    u = jnp.clip(jax.random.normal(key, (2, 6, 6, 1)), -1.0, 1.0)
    huge = jnp.full((2, 1, 1, 1), 1e6)
    z = physics.conserve_energy_bounded(u, huge, -1.0, 1.0)
    assert float(jnp.max(z)) <= 1.0 + 1e-6
    assert float(jnp.min(physics.projection_residual(z, huge))) > 0.0


def test_headroom_projection_is_differentiable():
    lo, hi = -1.0, 1.0
    u = jnp.clip(jax.random.normal(jax.random.PRNGKey(0), (2, 6, 6, 1)), lo, hi)
    tgt = physics.total_mass_per_channel(u) + 1.0
    g = jax.grad(lambda x: jnp.sum(physics.conserve_energy_bounded(x, tgt, lo, hi) ** 2))(u)
    assert bool(jnp.all(jnp.isfinite(g)))


# ------------------------------------------------------- new baselines ---
@pytest.mark.parametrize("arch", ["resnet", "resnet_iso", "unet", "unet_iso", "identity"])
@pytest.mark.parametrize("C", [1, 3])
def test_baselines_start_as_the_identity_map(arch, C):
    """Zero-init heads: every emulator begins as g(x)=x, like the NCA baselines."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 16, C))
    model = registry.REGISTRY[arch].make(C)()
    y = model.apply(model.init(key, x), x)
    assert y.shape == x.shape
    assert np.allclose(np.asarray(y), np.asarray(x), atol=1e-6)


def test_identity_floor_is_exactly_the_identity_after_a_rollout():
    from pinca_jax.harness import _emu_traj
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 16, 16, 1))
    m = registry.REGISTRY["identity"].make(1)()
    p = m.init(key, x)
    assert np.allclose(np.asarray(_emu_traj(m, p, x, 20)[-1]), np.asarray(x), atol=1e-6)


def test_unet_degrades_levels_on_an_indivisible_grid_instead_of_resizing():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 12, 12, 1))
    m = registry.REGISTRY["unet"].make(1)()
    assert m.apply(m.init(key, x), x).shape == x.shape


def test_every_registry_arch_has_a_budget_class():
    """An efficiency claim needs a budget class; a new arch must not silently escape one."""
    missing = [k for k in registry.REGISTRY if k not in registry.BUDGET_CLASS]
    assert missing == []
