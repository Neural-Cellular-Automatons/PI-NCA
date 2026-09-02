"""End-to-end smoke: the JAX heat PI-NCA trainer must actually reduce loss."""
import pytest

pytest.importorskip("optax")

from pinca_jax.configs import HeatNCAConfig
from pinca_jax.train_nca import train


def test_training_reduces_loss():
    cfg = HeatNCAConfig(grid_size=16, batch_size=8, rollout_steps=6, epochs=30, seed=0)
    out = train(cfg, verbose=False)
    losses = out["losses"]
    # Loss should drop meaningfully (conservative threshold to avoid flakiness).
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
