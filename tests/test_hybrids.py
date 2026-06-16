"""Hybrid architectures: shape, mass-conservation, and bounding properties."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pinca_jax.models.hybrids import (BoundedConsFluxNCA, SpectralFluxNCA, MultiScaleFluxNCA)
from pinca_jax import metrics


def _init_apply(model, C=1, n=12):
    x = jnp.asarray(np.random.default_rng(0).standard_normal((2, n, n, C)).astype(np.float32))
    x = jnp.tanh(x)  # in (-1,1) for the bounded model
    params = model.init(jax.random.PRNGKey(0), x)
    return x, model.apply(params, x)


def test_bounded_cons_conserves_mass_and_bounds():
    model = BoundedConsFluxNCA(bounds=(-1.0, 1.0))
    x, y = _init_apply(model)
    # exact total-mass conservation (the resolved tension)
    assert metrics.conservation_error(y, x) < 1e-3
    # bounded up to the tiny uniform re-projection (mass restore) — stays well-controlled
    assert float(jnp.max(jnp.abs(y))) < 1.5
    assert y.shape == x.shape


@pytest.mark.parametrize("ctor", [
    lambda: SpectralFluxNCA(conserve=True),
    lambda: MultiScaleFluxNCA(conserve=True),
])
def test_conserving_hybrids_conserve_mass(ctor):
    model = ctor()
    x, y = _init_apply(model)
    assert y.shape == x.shape
    assert metrics.conservation_error(y, x) < 1e-2


def test_hybrids_start_near_identity():
    # zero-init heads → first step is ~identity (NCA stabiliser); check small change.
    model = MultiScaleFluxNCA(conserve=True)
    x, y = _init_apply(model)
    assert float(jnp.mean(jnp.abs(y - x))) < 0.2
