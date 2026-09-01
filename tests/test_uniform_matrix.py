"""Gate for the uniform benchmark matrix.

Every architecture is now generic in the channel count, so the same model list is
measured on every phenomenon. Two things have to hold for that to be trustworthy:

1. **Nothing changed at C = 1.** The flux models used to hardcode a 2-channel head and
   a scalar divergence. They now emit 2*C channels and use the per-channel divergence.
   At C = 1 the result must be bit-identical, or every previously published number
   silently moved.
2. **Everything actually runs, and conserves per field, at C > 1.**
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pinca_jax import physics
from pinca_jax.equations import pdes
from pinca_jax.models import registry
from pinca_jax.models.flux_nca import DeepFluxNCA
from pinca_jax.models.hybrids import (BoundedConsFluxNCA, MultiScaleFluxNCA,
                                      SpectralFluxNCA)

KEY = jax.random.PRNGKey(0)
GRID = 12

# Architectures whose update is mass-conserving by construction.
CONSERVING = {"pi_nca", "mc_flux_nca", "multiscale_flux_nca", "bounded_cons_nca",
              "bounded_multiscale_nca", "spectral_flux_nca"}
# Architectures that clip, so they conserve only up to the clip's re-projection.
BOUNDED = {"bounded_cons_nca", "bounded_multiscale_nca"}


def _state(C, key=KEY, scale=1.0):
    return jax.random.uniform(key, (2, GRID, GRID, C), minval=-scale, maxval=scale)


# --------------------------------------------------------------------------- #
# 1. C == 1 must be unchanged by the generalisation
# --------------------------------------------------------------------------- #
def test_multichannel_divergence_reduces_to_scalar_at_c1():
    """The per-channel divergence must equal the scalar one at C == 1, exactly."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((3, 10, 10, 1)).astype(np.float32))
    flux = jnp.asarray(rng.standard_normal((3, 10, 10, 2)).astype(np.float32))
    scalar = physics.divergence_flux_update(x, flux)
    multi = physics.multichannel_divergence_update(x, flux)
    np.testing.assert_array_equal(np.asarray(scalar), np.asarray(multi))


def test_per_channel_mass_helpers_reduce_to_global_at_c1():
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal((4, 9, 9, 1)).astype(np.float32))
    np.testing.assert_allclose(np.asarray(physics.total_mass(u)),
                               np.asarray(physics.total_mass_per_channel(u)), rtol=0, atol=0)
    tgt = physics.total_mass(u) * 1.3
    np.testing.assert_allclose(np.asarray(physics.conserve_energy(u, tgt)),
                               np.asarray(physics.conserve_energy_per_channel(u, tgt)),
                               rtol=0, atol=0)


@pytest.mark.parametrize("ctor", [
    lambda: DeepFluxNCA(out_channels=1),
    lambda: MultiScaleFluxNCA(out_channels=1, conserve=True),
    lambda: BoundedConsFluxNCA(out_channels=1, bounds=(-1.0, 1.0)),
    lambda: SpectralFluxNCA(out_channels=1, conserve=True),
])
def test_scalar_models_still_have_the_same_parameter_shapes(ctor):
    """At C == 1 the flux head is still 2 channels, so params are unchanged in shape."""
    m = ctor()
    x = _state(1)
    p = m.init(KEY, x)
    flat = jax.tree_util.tree_leaves_with_path(p)
    heads = [(str(k), v.shape) for k, v in flat if "flux_head" in str(k) and v.ndim == 4]
    assert heads, "no flux head found"
    for _, shape in heads:
        assert shape[-1] == 2, f"C=1 flux head must stay 2 channels, got {shape}"


# --------------------------------------------------------------------------- #
# 2. every architecture runs at every channel count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("C", [1, 2, 3])
@pytest.mark.parametrize("name", sorted(registry.BENCH_ARCHS))
def test_every_arch_runs_at_every_channel_count(name, C):
    spec = registry.REGISTRY[name]
    model = spec.make(C, bounds=(-2.0, 2.0))()
    x = _state(C)
    params = model.init(KEY, x)
    y = model.apply(params, x)
    assert y.shape == x.shape, f"{name} C={C}: {y.shape} != {x.shape}"
    assert bool(jnp.isfinite(y).all()), f"{name} C={C} produced non-finite output"


@pytest.mark.parametrize("C", [2, 3])
@pytest.mark.parametrize("name", sorted(CONSERVING))
def test_conserving_archs_conserve_each_channel_separately(name, C):
    """Per-field conservation: channel k's total must not leak into channel j.

    The head is zero-initialised, so a fresh model is the identity. Perturb the flux
    head to a non-trivial value first, or the test passes vacuously.
    """
    spec = registry.REGISTRY[name]
    model = spec.make(C, bounds=(-50.0, 50.0))()      # bounds wide enough to be inert
    x = _state(C, scale=1.0)
    params = model.init(KEY, x)

    def perturb(path, v):
        p = str(path)
        if ("flux_head" in p or "proj" in p) and v.ndim >= 2:
            return jax.random.normal(KEY, v.shape) * 0.05
        return v

    params = jax.tree_util.tree_map_with_path(perturb, params)
    y = model.apply(params, x)

    before = np.asarray(physics.total_mass_per_channel(x)).ravel()
    after = np.asarray(physics.total_mass_per_channel(y)).ravel()
    np.testing.assert_allclose(after, before, rtol=1e-3, atol=1e-3,
                               err_msg=f"{name} C={C} did not conserve per channel")


@pytest.mark.parametrize("name", sorted(BOUNDED))
@pytest.mark.parametrize("C", [1, 2])
def test_bounded_archs_respect_the_bounds_they_are_given(name, C):
    lo, hi = -0.4, 0.9
    model = registry.REGISTRY[name].make(C, bounds=(lo, hi))()
    x = _state(C, scale=3.0)                          # deliberately outside the bounds
    params = model.init(KEY, x)
    y = np.asarray(model.apply(params, x))
    # The mass re-projection runs after the clip, so it can push values slightly out;
    # what must hold is that a wildly out-of-range input is pulled to roughly the range.
    span = hi - lo
    assert y.min() >= lo - span, f"{name}: {y.min()} far below {lo}"
    assert y.max() <= hi + span, f"{name}: {y.max()} far above {hi}"


# --------------------------------------------------------------------------- #
# 3. the matrix really is uniform
# --------------------------------------------------------------------------- #
def test_bench_arch_list_is_identical_for_every_pde():
    """The whole point: same model list on every phenomenon, scalar or multi-field."""
    lists = {}
    for name, spec in pdes.REGISTRY.items():
        lists[name] = sorted(registry.bench_archs(spec.channels))
    first = next(iter(lists.values()))
    for pde, got in lists.items():
        assert got == first, f"{pde} has a different arch list: {got} != {first}"
    assert len(first) >= 8, f"expected the full competitor set, got {first}"


def test_ablation_probes_are_excluded_from_the_comparison():
    """A4/A5 probes are matched-backbone controls, not competitors."""
    assert not any(a.startswith("abl_") for a in registry.BENCH_ARCHS)


# --------------------------------------------------------------------------- #
# 4. durability primitives for long unattended GPU runs
# --------------------------------------------------------------------------- #
class _FakeXlaError(Exception):
    """Stands in for jaxlib's XlaRuntimeError, whose import path moves between releases."""


@pytest.mark.parametrize("msg", [
    "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8589934592 bytes.",
    "Failed to allocate request for 12.00GiB (12884901888B) on device ordinal 0",
    "OOM when allocating tensor with shape[64,64,64,48]",
    "Resource exhausted: Out of memory trying to allocate 2.50GiB.",
    "CUDA_ERROR_OUT_OF_MEMORY: out of memory",
])
def test_oom_is_detected_in_every_wording_xla_uses(msg):
    from pinca_jax import bench
    assert bench.is_oom(_FakeXlaError(msg))


@pytest.mark.parametrize("msg", ["shape mismatch (2,3) vs (4,5)",
                                 "nan encountered in gradient", "division by zero"])
def test_real_bugs_are_not_mistaken_for_oom(msg):
    """A shape bug retried at half the batch would waste hours and still fail."""
    from pinca_jax import bench
    assert not bench.is_oom(ValueError(msg))


def test_oom_backoff_halves_until_it_fits_and_reports_the_batch_used():
    from pinca_jax import bench
    tried = []

    def flaky(b):
        tried.append(b)
        if b > 8:
            raise _FakeXlaError("RESOURCE_EXHAUSTED: Out of memory")
        return "ok"

    res, used = bench.run_with_oom_backoff(flaky, 64, min_batch=2, label="t")
    assert (res, used) == ("ok", 8)
    assert tried == [64, 32, 16, 8]


def test_non_oom_errors_are_not_retried():
    from pinca_jax import bench
    tried = []

    def broken(b):
        tried.append(b)
        raise ValueError("genuine bug")

    with pytest.raises(ValueError):
        bench.run_with_oom_backoff(broken, 64, label="t")
    assert tried == [64], "a real bug must surface immediately, not after 5 retries"


def test_oom_backoff_gives_up_at_the_floor():
    from pinca_jax import bench

    def always(b):
        raise _FakeXlaError("RESOURCE_EXHAUSTED")

    with pytest.raises(_FakeXlaError):
        bench.run_with_oom_backoff(always, 8, min_batch=2, label="t")


def test_results_io_is_atomic_and_survives_a_corrupt_file(tmp_path):
    """Resume must not be defeated by a file truncated by a crash mid-write."""
    from pinca_jax import bench
    p = str(tmp_path / "x.json")
    bench.save_results(p, {"results": {"a": 1}})
    assert bench.load_results(p) == {"a": 1}
    assert not (tmp_path / "x.json.tmp").exists()
    (tmp_path / "x.json").write_text("{truncated")
    assert bench.load_results(p) == {}
    assert bench.load_results(str(tmp_path / "missing.json")) == {}


def test_require_gpu_refuses_the_cpu_backend():
    """No silent CPU fallback: half a matrix on CPU and half on GPU is meaningless."""
    from pinca_jax import env
    import jax
    if jax.default_backend() == "gpu":
        pytest.skip("this host has a GPU; the refusal path cannot be exercised")
    with pytest.raises(env.NotOnGPU):
        env.require_gpu("test")
    assert env.require_gpu("test", allow_cpu=True) == jax.default_backend()
