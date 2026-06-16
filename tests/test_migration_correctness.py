"""Migration correctness gate: JAX port == PyTorch reference (to tolerance).

The original `PI NCA_v1.py` runs a full training loop at import time, so we
re-declare the reference `HeatEquationSolver` and `DeepFluxNCA` here *verbatim*
(byte-for-byte logic) and compare the JAX implementations against them.

This is the gate required before ANY architecture change (per the mandate:
"Preserve numerical correctness during migration ... Do not begin architecture
modifications until migration correctness is verified").
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pinca_jax.equations import heat  # noqa: E402
from pinca_jax.models.flux_nca import DeepFluxNCA  # noqa: E402
from pinca_jax import physics  # noqa: E402

ATOL, RTOL = 1e-5, 1e-4


# --------------------------------------------------------------------------- #
# Verbatim PyTorch reference (copied from PI NCA_v1.py)
# --------------------------------------------------------------------------- #
class RefHeatSolver(nn.Module):
    def __init__(self, alpha, dt):
        super().__init__()
        self.alpha_dt = alpha * dt
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.laplace = nn.Conv2d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
        self.laplace.weight.data = kernel
        self.laplace.requires_grad_(False)

    def step(self, u):
        return u + self.alpha_dt * self.laplace(u)

    def k_steps(self, u, k):
        for _ in range(k):
            u = self.step(u)
        return u


class RefDeepFluxNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceive = nn.Conv2d(1, 32, 3, padding=1, padding_mode="circular")
        self.process = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1, bias=False),
        )
        with torch.no_grad():
            self.process[-1].weight.zero_()

    def forward(self, x):
        flux = self.process(self.perceive(x))
        fx, fy = flux[:, 0:1], flux[:, 1:2]
        dx = (torch.roll(fx, 1, 3) - fx) + (torch.roll(fy, 1, 2) - fy)
        return x + dx


# --------------------------------------------------------------------------- #
# Solver equivalence
# --------------------------------------------------------------------------- #
def _nchw_to_nhwc(a):
    return np.transpose(a, (0, 2, 3, 1))


@pytest.mark.parametrize("H,W", [(16, 16), (16, 24)])
def test_laplacian_matches_torch_circular_conv(H, W):
    # NHWC convention: JAX laplacian acts on spatial axes (1,2). Compare to the
    # PyTorch circular-conv reference (NCHW) by transposing.
    rng = np.random.default_rng(0)
    u_nchw = rng.standard_normal((2, 1, H, W)).astype(np.float32)
    ref = RefHeatSolver(0.5, 0.1)
    t_lap = _nchw_to_nhwc(ref.laplace(torch.from_numpy(u_nchw)).detach().numpy())
    j_lap = np.asarray(heat.laplacian_periodic(jnp.asarray(_nchw_to_nhwc(u_nchw))))
    np.testing.assert_allclose(j_lap, t_lap, atol=ATOL, rtol=RTOL)


def test_laplacian_is_2d_isotropic():
    """Regression for the NHWC-axis bug: a radially symmetric bump must have a
    Laplacian symmetric under x<->y transpose. A degenerate 1-D Laplacian
    (rolling the channel axis) would fail this."""
    n = 21
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cx = cy = n // 2
    bump = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / 8.0)).astype(np.float32)
    u = jnp.asarray(bump[None, :, :, None])  # NHWC
    lap = np.asarray(heat.laplacian_periodic(u))[0, :, :, 0]
    np.testing.assert_allclose(lap, lap.T, atol=1e-5)         # x<->y symmetry
    assert lap[cy, cx] < 0                                    # concave at the peak


def test_heat_step_and_rollout_match_torch():
    rng = np.random.default_rng(1)
    u_nchw = rng.standard_normal((3, 1, 16, 16)).astype(np.float32)
    u = jnp.asarray(_nchw_to_nhwc(u_nchw))
    ref = RefHeatSolver(0.5, 0.1)
    alpha_dt = 0.5 * 0.1
    # single step
    t1 = _nchw_to_nhwc(ref.step(torch.from_numpy(u_nchw)).detach().numpy())
    j1 = np.asarray(heat.heat_step(u, alpha_dt))
    np.testing.assert_allclose(j1, t1, atol=ATOL, rtol=RTOL)
    # 25-step rollout (lax.scan vs python loop)
    tK = _nchw_to_nhwc(ref.k_steps(torch.from_numpy(u_nchw), 25).detach().numpy())
    jK = np.asarray(heat.rollout(u, alpha_dt, 25))
    np.testing.assert_allclose(jK, tK, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# NCA equivalence (weight-ported, NON-zero flux head so divergence is exercised)
# --------------------------------------------------------------------------- #
def _port_torch_nca_to_flax(ref: RefDeepFluxNCA, flax_params):
    """Copy PyTorch conv weights into the Flax param pytree.

    torch Conv2d kernel (out,in,kH,kW) -> flax kernel (kH,kW,in,out)  [transpose 2,3,1,0]
    """
    p = jax.tree_util.tree_map(lambda x: x, flax_params)  # mutable copy
    sd = dict(ref.named_parameters())

    def k(w):  # torch (O,I,kH,kW) -> flax (kH,kW,I,O)
        return jnp.asarray(w.detach().numpy().transpose(2, 3, 1, 0))

    def b(w):
        return jnp.asarray(w.detach().numpy())

    p["params"]["perceive"]["kernel"] = k(sd["perceive.weight"])
    p["params"]["perceive"]["bias"] = b(sd["perceive.bias"])
    p["params"]["proc1"]["kernel"] = k(sd["process.1.weight"])
    p["params"]["proc1"]["bias"] = b(sd["process.1.bias"])
    p["params"]["proc2"]["kernel"] = k(sd["process.3.weight"])
    p["params"]["proc2"]["bias"] = b(sd["process.3.bias"])
    p["params"]["flux_head"]["kernel"] = k(sd["process.5.weight"])
    return p


def test_flux_nca_matches_torch_with_nonzero_head():
    rng = np.random.default_rng(2)
    ref = RefDeepFluxNCA()
    # Replace the zero-initialised flux head with random weights so the divergence
    # update is non-trivial (otherwise both sides trivially return x).
    with torch.no_grad():
        ref.process[-1].weight.copy_(torch.randn_like(ref.process[-1].weight))

    x_nchw = rng.standard_normal((2, 1, 16, 16)).astype(np.float32)
    t_out = ref(torch.from_numpy(x_nchw)).detach().numpy()

    # Flax model is NHWC.
    x_nhwc = jnp.asarray(np.transpose(x_nchw, (0, 2, 3, 1)))
    model = DeepFluxNCA()
    params = model.init(jax.random.PRNGKey(0), x_nhwc)
    params = _port_torch_nca_to_flax(ref, params)
    j_out_nhwc = np.asarray(model.apply(params, x_nhwc))
    j_out = np.transpose(j_out_nhwc, (0, 3, 1, 2))  # back to NCHW for comparison

    np.testing.assert_allclose(j_out, t_out, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- #
# Conservation structure
# --------------------------------------------------------------------------- #
def test_divergence_update_conserves_mass():
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((4, 12, 12, 1)).astype(np.float32))
    flux = jnp.asarray(rng.standard_normal((4, 12, 12, 2)).astype(np.float32))
    out = physics.divergence_flux_update(x, flux)
    # Net mass change from the discrete divergence is ~0 on a periodic grid.
    dmass = (out - x).sum(axis=(1, 2, 3))
    np.testing.assert_allclose(np.asarray(dmass), 0.0, atol=1e-3)


def test_conserve_energy_hits_target():
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal((3, 10, 10, 1)).astype(np.float32))
    target = jnp.asarray(rng.standard_normal((3, 1, 1, 1)).astype(np.float32)) * 5.0
    proj = physics.conserve_energy(u, target)
    got = proj.sum(axis=(1, 2, 3), keepdims=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(target), atol=1e-3)
