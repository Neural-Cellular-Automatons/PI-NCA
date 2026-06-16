"""The 8-PDE suite — differentiable JAX reference solvers (numerical teachers).

Faithfully migrated from the PyTorch notebook
`PINCA_v3plus_SWE_FHN_CH_PSNR.ipynb` (see docs/migration/pde_inventory.md for the
verified source formulas). Every solver is a pure `step(state, params) -> state`
on an NHWC array (channel = state dimension); rollouts use `jax.lax.scan`.

Correctness is asserted against verbatim PyTorch references in
tests/test_pde_suite_correctness.py before any architecture work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp

from .operators import laplacian, grad_x, grad_y


# --------------------------------------------------------------------------- #
# Steppers. state is NHWC; channel axis holds the PDE's field components.
# --------------------------------------------------------------------------- #
def heat_step(s, p):
    return s + p["dt"] * p["alpha"] * laplacian(s)


def wave_step(s, p):  # (u, v)
    u, v = s[..., 0:1], s[..., 1:2]
    v_new = v + p["dt"] * p["c"] ** 2 * laplacian(u)
    u_new = u + p["dt"] * v_new
    return jnp.concatenate([u_new, v_new], axis=-1)


def adv_diff_step(s, p):
    return s + p["dt"] * (p["D"] * laplacian(s) - p["vx"] * grad_x(s) - p["vy"] * grad_y(s))


def allen_cahn_step(s, p):
    return s + p["dt"] * (p["eps2"] * laplacian(s) + s - s ** 3)


def gray_scott_step(s, p):  # (u, v)
    u, v = s[..., 0:1], s[..., 1:2]
    uvv = u * v * v
    du = p["Du"] * laplacian(u) - uvv + p["F"] * (1.0 - u)
    dv = p["Dv"] * laplacian(v) + uvv - (p["F"] + p["k"]) * v
    return jnp.concatenate([u + p["dt"] * du, v + p["dt"] * dv], axis=-1)


def _swe_rhs(s, g):  # (h, hu, hv)
    h = jnp.clip(s[..., 0:1], min=1e-3)
    hu, hv = s[..., 1:2], s[..., 2:3]
    u, v = hu / h, hv / h
    dh = -(grad_x(hu) + grad_y(hv))
    dhu = -(grad_x(hu * u + 0.5 * g * h * h) + grad_y(hu * v))
    dhv = -(grad_x(hu * v) + grad_y(hv * v + 0.5 * g * h * h))
    return jnp.concatenate([dh, dhu, dhv], axis=-1)


def shallow_water_step(s, p):  # RK4
    dt, g = p["dt"], p["g"]
    k1 = _swe_rhs(s, g)
    k2 = _swe_rhs(s + 0.5 * dt * k1, g)
    k3 = _swe_rhs(s + 0.5 * dt * k2, g)
    k4 = _swe_rhs(s + dt * k3, g)
    ns = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    h = jnp.clip(ns[..., 0:1], min=1e-4)
    return jnp.concatenate([h, ns[..., 1:2], ns[..., 2:3]], axis=-1)


def cahn_hilliard_step(s, p):
    mu = s ** 3 - s - p["eps2"] * laplacian(s)
    dudt = laplacian(mu)
    return jnp.clip(s + p["dt"] * dudt, -1.0, 1.0)


def fitzhugh_nagumo_step(s, p):  # (u, v)
    u, v = s[..., 0:1], s[..., 1:2]
    du = p["Du"] * laplacian(u) + (u - u ** 3 / 3.0 - v) / p["tau"]
    dv = p["Dv"] * laplacian(v) + p["eps"] * (u + p["a"] - p["b"] * v)
    return jnp.concatenate([u + p["dt"] * du, v + p["dt"] * dv], axis=-1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PDESpec:
    name: str
    channels: int
    step: Callable
    params: dict
    conserves_mass: bool = False  # exact discrete mass conservation of the solver


REGISTRY: dict[str, PDESpec] = {
    "heat": PDESpec("heat", 1, heat_step, dict(alpha=0.5, dt=0.1), conserves_mass=True),
    "wave": PDESpec("wave", 2, wave_step, dict(c=0.5, dt=0.05)),
    "adv_diff": PDESpec("adv_diff", 1, adv_diff_step, dict(D=0.1, vx=0.3, vy=0.2, dt=0.08), conserves_mass=True),
    "allen_cahn": PDESpec("allen_cahn", 1, allen_cahn_step, dict(eps2=0.01, dt=0.04)),
    "gray_scott": PDESpec("gray_scott", 2, gray_scott_step, dict(Du=0.2, Dv=0.05, F=0.035, k=0.065, dt=2.0)),
    "shallow_water": PDESpec("shallow_water", 3, shallow_water_step, dict(g=1.0, dt=0.05), conserves_mass=True),
    "cahn_hilliard": PDESpec("cahn_hilliard", 1, cahn_hilliard_step, dict(eps2=0.01, dt=0.5), conserves_mass=True),
    "fitzhugh_nagumo": PDESpec("fitzhugh_nagumo", 2, fitzhugh_nagumo_step, dict(Du=0.5, Dv=0.1, a=0.7, b=0.8, tau=12.5, eps=0.08, dt=0.1)),
}


def override_params(name: str, **overrides) -> PDESpec:
    """Return a copy of a registry spec with some params replaced.

    Used to obtain numerically stable variants where the verbatim notebook
    parameters violate an explicit-scheme stability limit. Example:
    Gray-Scott ships with dt=2.0 (faithful to the notebook) but that exceeds the
    diffusion stability bound dt <= dx^2/(4*Du) = 1.25 and diverges on sharp ICs;
    `override_params("gray_scott", dt=1.0)` gives a stable teacher for experiments.
    """
    base = REGISTRY[name]
    new_params = {**base.params, **overrides}
    return PDESpec(base.name, base.channels, base.step, new_params, base.conserves_mass)


# Numerically stable teacher configs for experiments (documented deviations from
# the verbatim notebook params; see research log "Phase 2 / Gray-Scott stability").
STABLE = {
    "gray_scott": override_params("gray_scott", dt=1.0),
}


def rollout(spec: PDESpec, s0: jax.Array, n_steps: int) -> jax.Array:
    """Roll a registry solver forward `n_steps` (final state) via lax.scan."""
    def body(s, _):
        return spec.step(s, spec.params), None
    s_final, _ = jax.lax.scan(body, s0, xs=None, length=n_steps)
    return s_final


def rollout_trajectory(spec: PDESpec, s0: jax.Array, n_steps: int) -> jax.Array:
    """Stacked trajectory (n_steps, B, H, W, C)."""
    def body(s, _):
        s_next = spec.step(s, spec.params)
        return s_next, s_next
    _, traj = jax.lax.scan(body, s0, xs=None, length=n_steps)
    return traj
