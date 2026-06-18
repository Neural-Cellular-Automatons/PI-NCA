"""Correctness gate for the 3-D PDE suite: JAX steppers == verbatim PyTorch 3-D
references; 3-D conservation; 3-D model forward shapes."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pinca_jax.equations import pdes3d  # noqa: E402
from pinca_jax import physics3d, ic3d  # noqa: E402
import pinca_jax.models3d as M  # noqa: E402


# ---- verbatim torch 3-D operators (NCDHW; axes -1,-2,-3 = x,y,z) ----
def t_lap(f):
    s = -6.0 * f
    for a in (-1, -2, -3):
        s = s + torch.roll(f, 1, a) + torch.roll(f, -1, a)
    return s


def ref_heat(s, p):
    return s + p["dt"] * p["alpha"] * t_lap(s)


def ref_allen_cahn(s, p):
    return s + p["dt"] * (p["eps2"] * t_lap(s) + s - s ** 3)


def ref_nagumo(s, p):
    return s + p["dt"] * (p["D"] * t_lap(s) + s * (1.0 - s) * (s - p["a"]))


REFS = {"heat": ref_heat, "allen_cahn": ref_allen_cahn, "nagumo": ref_nagumo}


@pytest.mark.parametrize("name", list(REFS))
def test_pde3d_step_matches_torch(name):
    spec = pdes3d.REGISTRY[name]
    rng = np.random.default_rng(abs(hash(name)) % 7 + 1)
    s = rng.standard_normal((2, 1, 10, 10, 10)).astype(np.float32)  # NCDHW
    if name in ("nagumo",):
        s = (np.tanh(s) * 0.5 + 0.5).astype(np.float32)
    t_out = REFS[name](torch.from_numpy(s), spec.params).numpy()
    s_ndhwc = jnp.asarray(np.transpose(s, (0, 2, 3, 4, 1)))   # -> NDHWC
    j_out = np.transpose(np.asarray(spec.step(s_ndhwc, spec.params)), (0, 4, 1, 2, 3))
    np.testing.assert_allclose(j_out, t_out, atol=1e-5, rtol=1e-4)


def test_pde3d_rollout_matches_torch_heat():
    spec = pdes3d.REGISTRY["heat"]
    rng = np.random.default_rng(3)
    s = rng.standard_normal((2, 1, 10, 10, 10)).astype(np.float32)
    s_t = torch.from_numpy(s)
    for _ in range(10):
        s_t = ref_heat(s_t, spec.params)
    s_ndhwc = jnp.asarray(np.transpose(s, (0, 2, 3, 4, 1)))
    j = np.transpose(np.asarray(pdes3d.rollout(spec, s_ndhwc, 10)), (0, 4, 1, 2, 3))
    np.testing.assert_allclose(j, s_t.numpy(), atol=1e-4, rtol=1e-3)


def test_3d_flux_update_conserves_mass():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((2, 8, 8, 8, 1)).astype(np.float32))
    flux = jnp.asarray(rng.standard_normal((2, 8, 8, 8, 3)).astype(np.float32))
    out = physics3d.divergence_flux_update(x, flux)
    dmass = (out - x).sum(axis=(1, 2, 3, 4))
    np.testing.assert_allclose(np.asarray(dmass), 0.0, atol=1e-3)


def test_3d_solver_rollouts_stable():
    for name, spec in pdes3d.REGISTRY.items():
        s0 = ic3d.make_state(jax.random.PRNGKey(1), name, 2, 12)
        traj = pdes3d.rollout(spec, s0, 15)
        assert jnp.all(jnp.isfinite(traj)), name


@pytest.mark.parametrize("ctor,C", [
    (lambda: M.NCA3D(), 1), (lambda: M.FluxNCA3D(), 1),
    (lambda: M.MultiScaleFluxNCA3D(), 1), (lambda: M.FNO3D(), 1),
    (lambda: M.MultiChannelFluxNCA3D(out_channels=2), 2),
])
def test_3d_models_forward(ctor, C):
    x = jnp.zeros((1, 12, 12, 12, C))
    model = ctor()
    p = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(p, jnp.ones_like(x))
    assert y.shape == x.shape and bool(jnp.all(jnp.isfinite(y)))
