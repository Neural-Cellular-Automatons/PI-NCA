"""Correctness gate for the experiments that ask which part of PI-NCA does the work.

Three questions, three groups of controls, and each only means something if the control
really has the property it is supposed to isolate:

* **Is it the cellular automaton or the flux head?** The flux head is bolted onto a
  spectral, a multi-resolution and a local non-shared-weight backbone. Each must conserve
  every field exactly for arbitrary weights, and its non-flux sibling must not, or the
  comparison is not about the head.
* **Does the flux idea need a cellular automaton at all?** The FINN-style baseline
  parameterises the flux facewise instead of cellwise. It must be conservative, and its
  source-term variant must not be, since that variant exists to be given to the reaction
  systems.
* **Does per-field conservation matter, or would one lumped total do?** The lumped
  control must conserve the sum over all fields while letting individual fields drift.

Plus the numerical-dispersion measurement, which has a closed-form answer to check
against, and the tuning sweep's plumbing.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax import dispersion, metrics, tuning
from pinca_jax.models import registry
from pinca_jax.models.finn import FINN2d
from pinca_jax.models.flux_nca import DeepFluxNCA, LumpedConsNCA

FLUX_PROBES = ["fno_flux", "unet_iso_flux", "resnet_iso_flux", "finn", "bounded_mc_nca"]
SIBLINGS = [("fno", "fno_flux"), ("unet_iso", "unet_iso_flux"),
            ("resnet_iso", "resnet_iso_flux")]


def _state(C, grid=16, seed=0):
    return jax.random.normal(jax.random.PRNGKey(seed), (2, grid, grid, C))


def _trained_like(model, x, shift=0.05, seed=0):
    """Params that are not the zero-initialised ones: conservation must hold there too."""
    p = model.init(jax.random.PRNGKey(seed), x)
    return jax.tree_util.tree_map(lambda v: v + shift, p)


def _per_field_drift(model, params, x):
    y = model.apply(params, x)
    return float(jnp.max(jnp.abs(y.sum(axis=(1, 2)) - x.sum(axis=(1, 2)))))


def _scale(x):
    """Magnitude the drift should be judged against: the largest per-field total."""
    return float(jnp.max(jnp.abs(x.sum(axis=(1, 2))))) + 1.0


# ------------------------------------------------------------------ flux heads ---
@pytest.mark.parametrize("name", FLUX_PROBES)
@pytest.mark.parametrize("C", [1, 3])
def test_flux_probe_conserves_every_field_for_arbitrary_weights(name, C):
    x = _state(C)
    model = registry.REGISTRY[name].make(C, bounds=(-1.0, 1.0))()
    drift = _per_field_drift(model, _trained_like(model, x), x)
    assert drift < 1e-3 * _scale(x), (name, C, drift)


@pytest.mark.parametrize("name", FLUX_PROBES)
def test_flux_probe_starts_as_the_identity(name):
    # Inside the bound, because a bounded configuration is deliberately *not* the identity
    # on a state that already violates the range it enforces.
    x = jnp.clip(_state(1), -1.0, 1.0)
    model = registry.REGISTRY[name].make(1, bounds=(-1.0, 1.0))()
    y = model.apply(model.init(jax.random.PRNGKey(0), x), x)
    assert np.allclose(np.asarray(y), np.asarray(x), atol=1e-6), name


@pytest.mark.parametrize("plain,flux", SIBLINGS)
def test_the_sibling_without_the_flux_head_does_not_conserve(plain, flux):
    """Otherwise the pair would not isolate the head."""
    x = _state(1)
    mp = registry.REGISTRY[plain].make(1)()
    mf = registry.REGISTRY[flux].make(1)()
    d_plain = _per_field_drift(mp, _trained_like(mp, x), x)
    d_flux = _per_field_drift(mf, _trained_like(mf, x), x)
    assert d_plain > 1e-2 * _scale(x), (plain, d_plain)
    assert d_flux < 1e-3 * _scale(x), (flux, d_flux)


@pytest.mark.parametrize("plain,flux", SIBLINGS)
def test_the_flux_head_is_nearly_free_in_parameters(plain, flux):
    """The comparison is only fair if the head does not change the budget."""
    x = _state(1)
    n_plain = metrics.param_count(registry.REGISTRY[plain].make(1)().init(
        jax.random.PRNGKey(0), x))
    n_flux = metrics.param_count(registry.REGISTRY[flux].make(1)().init(
        jax.random.PRNGKey(0), x))
    assert abs(n_flux - n_plain) / n_plain < 0.01, (plain, n_plain, n_flux)


# ----------------------------------------------------------------------- FINN ---
def test_finn_is_facewise_not_cellwise():
    """A cell's update must depend on the pair of states at each of its faces.

    The check that distinguishes the two parameterisations: perturbing a single cell
    changes the flux on the faces it touches, so exactly its own and its four neighbours'
    updates move, and nothing further away does.
    """
    x = jnp.zeros((1, 9, 9, 1)).at[0, 4, 4, 0].set(1.0)
    m = FINN2d(out_channels=1)
    p = _trained_like(m, x)
    d = np.abs(np.asarray(m.apply(p, x) - x))[0, :, :, 0]
    base = np.abs(np.asarray(m.apply(p, jnp.zeros_like(x)) - jnp.zeros_like(x)))[0, :, :, 0]
    moved = (np.abs(d - base) > 1e-6)
    assert moved[4, 4] and moved[3, 4] and moved[5, 4] and moved[4, 3] and moved[4, 5]
    assert not moved[0, 0] and not moved[7, 7]


def test_finn_source_variant_breaks_conservation_on_purpose():
    x = _state(2)
    m = FINN2d(out_channels=2, source=True)
    assert _per_field_drift(m, _trained_like(m, x), x) > 1e-3 * _scale(x)


# ------------------------------------------------- lumped vs per-field ---
def _random_head(model, x, scale=0.5, seed=1):
    """Init, then give the output head random weights of a real size.

    Shifting every leaf by a constant is not enough here: the question is what happens
    when the update is *asymmetric across fields*, and a uniform shift produces a nearly
    symmetric one. Randomising the head alone drives channel-asymmetric increments while
    leaving the rest of the network at its initialisation.
    """
    p = model.init(jax.random.PRNGKey(0), x)
    key = jax.random.PRNGKey(seed)
    def fill(path, v):
        if any("update" in str(k) or "head" in str(k) for k in path):
            return jax.random.normal(key, v.shape) * scale
        return v
    return jax.tree_util.tree_map_with_path(fill, p)


def test_lumped_control_conserves_the_total_but_not_each_field():
    """The empirical content of the per-field argument, as a property of the control."""
    x = _state(3)
    m = LumpedConsNCA(out_channels=3)
    p = _random_head(m, x)
    y = m.apply(p, x)
    lumped = float(jnp.max(jnp.abs(y.sum(axis=(1, 2, 3)) - x.sum(axis=(1, 2, 3)))))
    per_field = float(jnp.max(jnp.abs(y.sum(axis=(1, 2)) - x.sum(axis=(1, 2)))))
    assert lumped < 1e-3 * _scale(x), lumped
    assert per_field > 10 * max(lumped, 1e-12), (lumped, per_field)
    # and the per-field model it is compared against conserves both, under the same
    # randomised head
    mp = DeepFluxNCA(out_channels=3)
    assert _per_field_drift(mp, _random_head(mp, x), x) < 1e-3 * _scale(x)


def test_lumped_control_is_parameter_matched_to_pi_nca():
    x = _state(3)
    k = jax.random.PRNGKey(0)
    n_l = metrics.param_count(LumpedConsNCA(out_channels=3).init(k, x))
    n_p = metrics.param_count(DeepFluxNCA(out_channels=3).init(k, x))
    assert abs(n_l - n_p) / n_p < 0.05, (n_l, n_p)


def test_bounded_wide_setting_respects_the_bound_and_the_mass():
    """The cell a pre-declared 'wide width, projection iff bounded' rule needs."""
    x = jnp.clip(_state(1), -1.0, 1.0)
    m = registry.REGISTRY["bounded_mc_nca"].make(1, bounds=(-1.0, 1.0))()
    y = m.apply(_trained_like(m, x, shift=0.3), x)
    assert float(jnp.max(y)) <= 1.0 + 1e-5 and float(jnp.min(y)) >= -1.0 - 1e-5
    assert float(jnp.max(jnp.abs(y.sum((1, 2)) - x.sum((1, 2))))) < 1e-3 * _scale(x)


# ------------------------------------------------------------ probes stay out ---
def test_probe_entries_are_excluded_from_the_uniform_matrix():
    """They answer one question each; putting them in the matrix would add unpaired rows."""
    for name in FLUX_PROBES + ["finn_src", "pi_nca_lumped"]:
        assert registry.REGISTRY[name].probe is True, name
        assert name not in registry.BENCH_ARCHS, name
        assert name in registry.BUDGET_CLASS, name
    assert len(registry.BENCH_ARCHS) == 16


# ------------------------------------------------------------------ dispersion ---
def test_measured_dispersion_matches_the_closed_form():
    """The wave scheme's realised frequency must match the analytic discrete relation.

    This is what licenses the statement that the continuum equation is non-dispersive
    while its discretisation is: the measurement and the closed form agree, so the phase
    error is the scheme's and not the fit's.
    """
    out = dispersion.study(grid=16, steps=64, modes=[(1, 0), (4, 0), (8, 0), (4, 4)])
    for row in out["modes"]:
        assert abs(row["omega_measured"] - row["omega_full_discrete"]) < 0.02 * \
            row["omega_full_discrete"], row
    lowest = out["modes"][0]
    highest = out["modes"][-2]                 # (8,0): the grid-scale axis mode here
    assert abs(lowest["phase_error"]) < 0.01
    assert highest["phase_error"] < -0.1       # short waves travel too slowly


def test_dispersion_reports_the_stability_ratio():
    a = dispersion.analytic(grid=16, mx=8, my=8, c=0.5, dt=0.05)
    assert a["stability_ratio"] < 1.0          # the benchmark's wave setting is stable
    unstable = dispersion.analytic(grid=16, mx=8, my=8, c=0.5, dt=5.0)
    assert unstable["stability_ratio"] > 1.0 and not np.isfinite(unstable["omega_full_discrete"])


# ---------------------------------------------------------------------- tuning ---
def test_tuning_sweep_reports_a_best_setting_and_a_loss_curve():
    out = tuning.study("heat", ["pi_nca"], grid=12, base_epochs=4, mult=2, batch=2,
                       rollout=2, eval_steps=4, n_eval=2, lrs=(1e-3, 3e-3))
    rec = out["pi_nca"]
    assert rec["best_setting"] in rec["settings"]
    assert rec["shared_recipe"] == "lr0.001_e4"
    for r in rec["settings"].values():
        assert len(r["loss_curve"]) >= 2 and r["rel_l2"] > 0
        assert r["second_half_improvement"] > 0
    # the long run must actually be the longer budget
    assert any(r["epochs"] == 8 for r in rec["settings"].values())
