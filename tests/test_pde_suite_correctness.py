"""Correctness gate for the 8-PDE suite: JAX registry steppers == verbatim
PyTorch references from the notebook (docs/migration/pde_inventory.md).

References are NCHW (as in the notebook); JAX steppers are NHWC. We compare by
transposing. Tolerances are float32 (atol 1e-5) for a single step and looser for
a short rollout (accumulated reduction-order differences)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinca_jax.equations import pdes  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# ---- verbatim PyTorch operators (NCHW) ---- #
def t_lap(f):
    return (torch.roll(f, 1, -1) + torch.roll(f, -1, -1)
            + torch.roll(f, 1, -2) + torch.roll(f, -1, -2) - 4.0 * f)


def t_gx(f):
    return (torch.roll(f, -1, -1) - torch.roll(f, 1, -1)) * 0.5


def t_gy(f):
    return (torch.roll(f, -1, -2) - torch.roll(f, 1, -2)) * 0.5


# ---- verbatim PyTorch steppers (NCHW), one step each ---- #
def ref_heat(s, p):
    return s + p["dt"] * p["alpha"] * t_lap(s)


def ref_wave(s, p):
    u, v = s[:, 0:1], s[:, 1:2]
    v_new = v + p["dt"] * p["c"] ** 2 * t_lap(u)
    u_new = u + p["dt"] * v_new
    return torch.cat([u_new, v_new], 1)


def ref_adv_diff(s, p):
    return s + p["dt"] * (p["D"] * t_lap(s) - p["vx"] * t_gx(s) - p["vy"] * t_gy(s))


def ref_allen_cahn(s, p):
    return s + p["dt"] * (p["eps2"] * t_lap(s) + s - s ** 3)


def ref_gray_scott(s, p):
    u, v = s[:, 0:1], s[:, 1:2]
    uvv = u * v * v
    du = p["Du"] * t_lap(u) - uvv + p["F"] * (1.0 - u)
    dv = p["Dv"] * t_lap(v) + uvv - (p["F"] + p["k"]) * v
    return torch.cat([u + p["dt"] * du, v + p["dt"] * dv], 1)


def _ref_swe_rhs(s, g):
    h = torch.clamp(s[:, 0:1], min=1e-3)
    hu, hv = s[:, 1:2], s[:, 2:3]
    u, v = hu / h, hv / h
    dh = -(t_gx(hu) + t_gy(hv))
    dhu = -(t_gx(hu * u + 0.5 * g * h * h) + t_gy(hu * v))
    dhv = -(t_gx(hu * v) + t_gy(hv * v + 0.5 * g * h * h))
    return torch.cat([dh, dhu, dhv], 1)


def ref_shallow_water(s, p):
    dt, g = p["dt"], p["g"]
    k1 = _ref_swe_rhs(s, g)
    k2 = _ref_swe_rhs(s + 0.5 * dt * k1, g)
    k3 = _ref_swe_rhs(s + 0.5 * dt * k2, g)
    k4 = _ref_swe_rhs(s + dt * k3, g)
    ns = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    ns = ns.clone()
    ns[:, 0:1] = torch.clamp(ns[:, 0:1], min=1e-4)
    return ns


def ref_cahn_hilliard(s, p):
    mu = s ** 3 - s - p["eps2"] * t_lap(s)
    return torch.clamp(s + p["dt"] * t_lap(mu), -1.0, 1.0)


def ref_fhn(s, p):
    u, v = s[:, 0:1], s[:, 1:2]
    du = p["Du"] * t_lap(u) + (u - u ** 3 / 3.0 - v) / p["tau"]
    dv = p["Dv"] * t_lap(v) + p["eps"] * (u + p["a"] - p["b"] * v)
    return torch.cat([u + p["dt"] * du, v + p["dt"] * dv], 1)


def ref_nagumo(s, p):
    return s + p["dt"] * (p["D"] * t_lap(s) + s * (1.0 - s) * (s - p["a"]))


# Verbatim references for the finite-difference PDEs. (navier_stokes is pseudo-spectral
# — no roll-based torch ref; it is covered by the IC+rollout stability test instead.)
REFS = {
    "heat": ref_heat, "wave": ref_wave, "adv_diff": ref_adv_diff,
    "allen_cahn": ref_allen_cahn, "gray_scott": ref_gray_scott,
    "shallow_water": ref_shallow_water, "cahn_hilliard": ref_cahn_hilliard,
    "fitzhugh_nagumo": ref_fhn, "nagumo": ref_nagumo,
}


def _seed(name: str) -> int:
    """Deterministic per-PDE seed (zlib.crc32 — NOT Python's per-process-salted hash())."""
    import zlib
    return zlib.crc32(name.encode()) % 2**31


def _make_state(name, C, rng):
    s = rng.standard_normal((2, C, 12, 12)).astype(np.float32)
    if name == "shallow_water":      # need positive water height
        s[:, 0] = 1.0 + 0.1 * rng.standard_normal((2, 12, 12))
    if name in ("cahn_hilliard",):   # bounded order parameter
        s = np.tanh(s).astype(np.float32)
    return s


@pytest.mark.parametrize("name", list(REFS))
def test_pde_step_matches_torch(name):
    spec = pdes.REGISTRY[name]
    rng = np.random.default_rng(_seed(name))
    s_nchw = _make_state(name, spec.channels, rng)
    # torch reference
    t_out = REFS[name](torch.from_numpy(s_nchw), spec.params).detach().numpy()
    # jax stepper (NHWC)
    s_nhwc = jnp.asarray(np.transpose(s_nchw, (0, 2, 3, 1)))
    j_out = np.transpose(np.asarray(spec.step(s_nhwc, spec.params)), (0, 3, 1, 2))
    np.testing.assert_allclose(j_out, t_out, atol=1e-5, rtol=1e-4)


# Shallow-water is a nonlinear hyperbolic system integrated with RK4; over a
# multi-step rollout the (correct) single-step difference in float reduction order
# between JAX and PyTorch amplifies, so its rollout tolerance is looser than the
# parabolic/RD PDEs. The single-step test above still pins it at atol 1e-5.
_ROLLOUT_TOL = {"shallow_water": (1e-2, 1e-2)}


@pytest.mark.parametrize("name", list(REFS))
def test_pde_rollout_matches_torch(name):
    spec = pdes.REGISTRY[name]
    rng = np.random.default_rng(_seed(name) + 1)
    s_nchw = _make_state(name, spec.channels, rng)
    s_t = torch.from_numpy(s_nchw)
    for _ in range(10):
        s_t = REFS[name](s_t, spec.params)
    t_out = s_t.detach().numpy()
    s_nhwc = jnp.asarray(np.transpose(s_nchw, (0, 2, 3, 1)))
    j_out = np.transpose(np.asarray(pdes.rollout(spec, s_nhwc, 10)), (0, 3, 1, 2))
    atol, rtol = _ROLLOUT_TOL.get(name, (1e-4, 1e-3))
    np.testing.assert_allclose(j_out, t_out, atol=atol, rtol=rtol)


def test_registry_channels_consistent():
    assert pdes.REGISTRY["shallow_water"].channels == 3
    assert pdes.REGISTRY["gray_scott"].channels == 2
    assert pdes.REGISTRY["heat"].channels == 1
