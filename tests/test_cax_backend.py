"""CAX is used on GPU only, and when used it must change nothing but speed."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax import cax_backend


def _force(monkeypatch, on):
    monkeypatch.setenv("PINCA_CAX", "1" if on else "0")


def test_policy_is_gpu_only_by_default(monkeypatch):
    monkeypatch.delenv("PINCA_CAX", raising=False)
    # No override: CAX iff the backend is a GPU. This suite runs on CPU in CI.
    assert cax_backend.use_cax() == (jax.default_backend() == "gpu")


def test_backends_agree(monkeypatch):
    f = lambda x: 0.99 * x + 0.01 * jnp.roll(x, 1, axis=1)
    x0 = jnp.asarray(np.random.default_rng(0).standard_normal((2, 8, 8, 1)), jnp.float32)
    _force(monkeypatch, False); ref = cax_backend.rollout(f, x0, 12)
    _force(monkeypatch, True);  cax = cax_backend.rollout(f, x0, 12)
    np.testing.assert_allclose(np.asarray(cax), np.asarray(ref), atol=1e-6)


def test_cax_rollout_is_differentiable(monkeypatch):
    _force(monkeypatch, True)
    x0 = jnp.ones((2, 4, 4, 1))
    g = jax.grad(lambda s: cax_backend.rollout(lambda x: x * s, x0, 8).sum())(1.01)
    # d/ds sum(x0 * s^8) = 8 s^7 sum(x0)
    assert float(g) == pytest.approx(8 * 1.01 ** 7 * x0.size, rel=1e-4)


def test_collect_always_uses_scan(monkeypatch):
    """Trajectories must come back stacked even with CAX forced on."""
    _force(monkeypatch, True)
    x0 = jnp.zeros((1, 4, 4, 1))
    traj = cax_backend.rollout(lambda x: x + 1.0, x0, 5, collect=True)
    assert traj.shape == (5, 1, 4, 4, 1)
    np.testing.assert_allclose(np.asarray(traj[-1]).ravel()[0], 5.0)
