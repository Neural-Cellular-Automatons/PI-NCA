"""3-D PDE suite — differentiable JAX reference solvers (NDHWC).

Direct 3-D extensions of the verified 2-D solvers (pdes.py), sharing the same
parameters where applicable. Each is a pure `step(state, params) -> state` on a 5-D
NDHWC array; rollouts use jax.lax.scan. Correctness vs verbatim torch 3-D references
is asserted in tests/test_pde3d_correctness.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .operators3d import laplacian_3d, grad_x, grad_y, grad_z


def heat_step(s, p):
    return s + p["dt"] * p["alpha"] * laplacian_3d(s)


def adv_diff_step(s, p):
    return s + p["dt"] * (p["D"] * laplacian_3d(s)
                          - p["vx"] * grad_x(s) - p["vy"] * grad_y(s) - p["vz"] * grad_z(s))


def allen_cahn_step(s, p):
    return s + p["dt"] * (p["eps2"] * laplacian_3d(s) + s - s ** 3)


def nagumo_step(s, p):
    return s + p["dt"] * (p["D"] * laplacian_3d(s) + s * (1.0 - s) * (s - p["a"]))


def gray_scott_step(s, p):  # (u, v)
    u, v = s[..., 0:1], s[..., 1:2]
    uvv = u * v * v
    du = p["Du"] * laplacian_3d(u) - uvv + p["F"] * (1.0 - u)
    dv = p["Dv"] * laplacian_3d(v) + uvv - (p["F"] + p["k"]) * v
    return jnp.concatenate([u + p["dt"] * du, v + p["dt"] * dv], axis=-1)


def fitzhugh_nagumo_step(s, p):  # (u, v)
    u, v = s[..., 0:1], s[..., 1:2]
    du = p["Du"] * laplacian_3d(u) + (u - u ** 3 / 3.0 - v) / p["tau"]
    dv = p["Dv"] * laplacian_3d(v) + p["eps"] * (u + p["a"] - p["b"] * v)
    return jnp.concatenate([u + p["dt"] * du, v + p["dt"] * dv], axis=-1)


@dataclass(frozen=True)
class PDESpec3D:
    name: str
    channels: int
    step: Callable
    params: dict
    conserves_mass: bool = False


REGISTRY: dict[str, PDESpec3D] = {
    "heat": PDESpec3D("heat", 1, heat_step, dict(alpha=0.5, dt=0.1), conserves_mass=True),
    "adv_diff": PDESpec3D("adv_diff", 1, adv_diff_step,
                          dict(D=0.1, vx=0.3, vy=0.2, vz=0.15, dt=0.06), conserves_mass=True),
    "allen_cahn": PDESpec3D("allen_cahn", 1, allen_cahn_step, dict(eps2=0.01, dt=0.04)),
    "nagumo": PDESpec3D("nagumo", 1, nagumo_step, dict(D=0.1, a=0.3, dt=0.1)),
    # dt=0.5 < 1/(6*Du)=0.83 — 3D diffusion stability (the 3D Laplacian factor is 6, not 4)
    "gray_scott": PDESpec3D("gray_scott", 2, gray_scott_step,
                            dict(Du=0.2, Dv=0.05, F=0.035, k=0.065, dt=0.5)),
    "fitzhugh_nagumo": PDESpec3D("fitzhugh_nagumo", 2, fitzhugh_nagumo_step,
                                 dict(Du=0.5, Dv=0.1, a=0.7, b=0.8, tau=12.5, eps=0.08, dt=0.1)),
}


def rollout(spec: PDESpec3D, s0, n_steps: int):
    def body(s, _):
        return spec.step(s, spec.params), None
    sf, _ = jax.lax.scan(body, s0, xs=None, length=n_steps)
    return sf


def rollout_trajectory(spec: PDESpec3D, s0, n_steps: int):
    def body(s, _):
        s_next = spec.step(s, spec.params)
        return s_next, s_next
    _, traj = jax.lax.scan(body, s0, xs=None, length=n_steps)
    return traj
