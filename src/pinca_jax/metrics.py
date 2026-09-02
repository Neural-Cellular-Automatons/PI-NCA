"""Evaluation metrics shared by every architecture (JAX).

All array metrics operate on NHWC tensors and reduce over all non-batch axes
unless noted. The mandated metric set: L2, relative error, residual, BC
satisfaction, conservation, stability, plus model size (params) and timing.
Multi-seed aggregation (mean ± std) is provided so we never report single runs.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .equations.operators import grad_x, grad_y


# ---- pointwise error metrics ---- #
def mse(pred, target):
    return float(jnp.mean((pred - target) ** 2))


def rmse(pred, target):
    return float(jnp.sqrt(jnp.mean((pred - target) ** 2)))


def mae(pred, target):
    return float(jnp.mean(jnp.abs(pred - target)))


def rel_l2(pred, target, eps=1e-8):
    """Relative L2: ||pred-target||_2 / ||target||_2 (global)."""
    num = jnp.sqrt(jnp.sum((pred - target) ** 2))
    den = jnp.sqrt(jnp.sum(target ** 2)) + eps
    return float(num / den)


def max_abs_error(pred, target):
    """L∞ (worst-cell) error."""
    return float(jnp.max(jnp.abs(pred - target)))


def ssim_like(pred, target):
    """Lightweight global SSIM-style structural similarity (1 = identical)."""
    mp, mt = jnp.mean(pred), jnp.mean(target)
    vp, vt = jnp.var(pred), jnp.var(target)
    cov = jnp.mean((pred - mp) * (target - mt))
    c1, c2 = 1e-4, 9e-4
    return float(((2 * mp * mt + c1) * (2 * cov + c2)) /
                 ((mp ** 2 + mt ** 2 + c1) * (vp + vt + c2)))


def highfreq_error_frac(pred, target):
    """Fraction of the squared error energy living in the upper half of the 2-D
    spatial spectrum — exposes whether a model misses fine scales (e.g. FNO's
    spectral truncation, NCA over-smoothing)."""
    e = (pred - target)[..., 0] if pred.ndim == 4 else (pred - target)
    ef = jnp.fft.fft2(e, axes=(-2, -1))
    p = jnp.abs(ef) ** 2
    N = e.shape[-1]
    coords = jnp.fft.fftfreq(N)
    hi = (jnp.abs(coords)[:, None] > 0.25) | (jnp.abs(coords)[None, :] > 0.25)
    tot = jnp.sum(p) + 1e-12
    return float(jnp.sum(p * hi) / tot)


def psnr(pred, target, data_range=None):
    """Peak signal-to-noise ratio (dB). data_range defaults to target span."""
    if data_range is None:
        data_range = float(jnp.max(target) - jnp.min(target))
        if data_range < 1e-6:  # constant field → fall back to magnitude (or 1)
            data_range = float(jnp.max(jnp.abs(target))) or 1.0
    m = jnp.mean((pred - target) ** 2)
    return float(20.0 * jnp.log10(data_range) - 10.0 * jnp.log10(m + 1e-12))


# ---- physics metrics ---- #
def residual(step_fn, params, u):
    """One-step PDE residual: how far a model step departs from the solver step.
    For an autoregressive emulator g, residual = || g(u) - solver_step(u) ||.
    Pass step_fn = solver step (state,params)->state, evaluated at u."""
    return float(jnp.sqrt(jnp.mean((step_fn(u, params) - u) ** 2)))  # placeholder if no model


def mass(u):
    """Total mass per sample (sum over H,W,C)."""
    return jnp.sum(u, axis=(1, 2, 3))


def conservation_error(pred, u0):
    """|mass(pred) - mass(u0)| averaged over batch — exact-conservation diagnostic."""
    return float(jnp.mean(jnp.abs(mass(pred) - mass(u0))))


def periodic_bc_residual(u):
    """Periodic-BC satisfaction: wrap-around mismatch (≈0 for roll-based/circular models)."""
    lr = jnp.mean(jnp.abs(u[:, :, 0, :] - u[:, :, -1, :]))
    tb = jnp.mean(jnp.abs(u[:, 0, :, :] - u[:, -1, :, :]))
    return float(0.5 * (lr + tb))


def gradient_energy(u):
    """Mean |∇u|² — a stability/roughness proxy; blow-up ⇒ instability."""
    gx, gy = grad_x(u), grad_y(u)
    return float(jnp.mean(gx ** 2 + gy ** 2))


# ---- model-size / timing ---- #
def param_count(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def time_callable(fn, *args, n_warmup=2, n_runs=10):
    """Wall-clock seconds per call (median), blocking on device."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(statistics.median(ts))


# ---- multi-seed aggregation ---- #
@dataclass
class Agg:
    mean: float
    std: float
    n: int

    def __str__(self):
        return f"{self.mean:.4e} ± {self.std:.2e} (n={self.n})"


def aggregate(values) -> Agg:
    vals = [float(v) for v in values]
    n = len(vals)
    mu = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return Agg(mu, sd, n)


def aggregate_runs(run_dicts):
    """List[dict[name->float]] → dict[name->Agg]. For multi-seed metric tables."""
    if not run_dicts:
        return {}
    keys = run_dicts[0].keys()
    return {k: aggregate([r[k] for r in run_dicts]) for k in keys}
