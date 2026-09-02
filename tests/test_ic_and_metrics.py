"""IC generators produce correct shapes and stable short rollouts; metrics work."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax import ic, metrics
from pinca_jax.equations import pdes


@pytest.mark.parametrize("name", list(pdes.REGISTRY))
def test_ic_shape_and_rollout_stable(name):
    # Use the stable teacher config where the verbatim params are stiff (gray_scott).
    spec = pdes.STABLE.get(name, pdes.REGISTRY[name])
    key = jax.random.PRNGKey(0)
    s0 = ic.make_state(key, name, batch=2, size=16)
    assert s0.shape == (2, 16, 16, spec.channels)
    traj = pdes.rollout(spec, s0, 20)
    assert jnp.all(jnp.isfinite(traj)), f"{name} produced non-finite states"


def test_gray_scott_dt2_is_unstable_finding():
    """Documented finding: the verbatim Gray-Scott dt=2.0 exceeds the explicit-Euler
    diffusion stability limit (dt <= 1/(4*Du) = 1.25) and diverges; dt=1.0 is stable.
    Locked in so the deviation in pdes.STABLE stays justified."""
    s0 = ic.make_state(jax.random.PRNGKey(0), "gray_scott", 2, 16)
    faithful = pdes.rollout(pdes.REGISTRY["gray_scott"], s0, 16)
    stable = pdes.rollout(pdes.STABLE["gray_scott"], s0, 16)
    assert not bool(jnp.all(jnp.isfinite(faithful)))   # dt=2.0 blows up
    assert bool(jnp.all(jnp.isfinite(stable)))         # dt=1.0 is fine


def test_metrics_basic():
    a = jnp.asarray(np.zeros((2, 8, 8, 1), np.float32))
    b = a + 1.0
    assert metrics.mse(b, a) == pytest.approx(1.0)
    assert metrics.rel_l2(a, a) == pytest.approx(0.0, abs=1e-6)
    assert metrics.psnr(a, a) > 100  # identical → very high PSNR
    agg = metrics.aggregate([1.0, 2.0, 3.0])
    assert agg.mean == pytest.approx(2.0) and agg.n == 3


def test_conservation_diagnostic_on_heat():
    # Heat solver conserves total mass exactly (periodic) → conservation_error ≈ 0.
    spec = pdes.REGISTRY["heat"]
    s0 = ic.make_state(jax.random.PRNGKey(1), "heat", 2, 16)
    sT = pdes.rollout(spec, s0, 30)
    assert metrics.conservation_error(sT, s0) < 1e-2
