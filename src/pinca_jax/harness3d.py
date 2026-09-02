"""3-D emulator train + eval harness (NDHWC). 3-D analogue of harness.py.

Single fixed seed (42), He-init, zero-init heads, LR warmup, pre-seeding — the same
"better start" protocol as 2-D. Reduced scale (16^3) for CPU feasibility.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from . import ic3d, metrics, physics3d
from .equations import pdes3d


@dataclass(frozen=True)
class Emu3DConfig:
    pde: str = "heat"
    grid_size: int = 16
    batch: int = 8
    rollout_steps: int = 8
    eval_steps: int = 24
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    n_eval: int = 4
    warmup_epochs: int = 20
    preseed_steps: int = 6
    output_clip: tuple | None = None

    def spec(self):
        return pdes3d.REGISTRY[self.pde]


def _rollout(model, params, x0, steps, clip=None):
    def body(x, _):
        x = model.apply(params, x)
        if clip is not None:
            x = jnp.clip(x, clip[0], clip[1])
        return x, x
    _, traj = jax.lax.scan(body, x0, xs=None, length=steps)
    return traj


def train(model_ctor, cfg: Emu3DConfig):
    spec = cfg.spec()
    key = jax.random.PRNGKey(cfg.seed)
    key, ik = jax.random.split(key)
    model = model_ctor()
    params = model.init(ik, ic3d.make_state(ik, cfg.pde, 1, cfg.grid_size))

    sched = (optax.join_schedules(
        [optax.linear_schedule(2e-4, cfg.lr, cfg.warmup_epochs), optax.constant_schedule(cfg.lr)],
        [cfg.warmup_epochs]) if cfg.warmup_epochs > 0 else cfg.lr)
    opt = optax.adamw(sched, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, k):
        x0 = ic3d.make_state(k, cfg.pde, cfg.batch, cfg.grid_size)
        if cfg.preseed_steps > 0:
            x0 = pdes3d.rollout(spec, x0, cfg.preseed_steps)
        target = pdes3d.rollout(spec, x0, cfg.rollout_steps)

        def loss_fn(p):
            def body(x, _):
                x = model.apply(p, x)
                if cfg.output_clip is not None:
                    x = jnp.clip(x, cfg.output_clip[0], cfg.output_clip[1])
                return x, None
            xf, _ = jax.lax.scan(body, x0, xs=None, length=cfg.rollout_steps)
            return jnp.mean((xf - target) ** 2)

        loss, g = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    t0 = time.time()
    for e in range(cfg.epochs):
        key, sk = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, sk)
    return {"params": params, "model": model, "wall_s": time.time() - t0,
            "final_loss": float(loss)}


def evaluate(model, params, cfg: Emu3DConfig):
    spec = cfg.spec()
    key = jax.random.PRNGKey(cfg.seed + 10_000)
    x0 = ic3d.make_state(key, cfg.pde, cfg.n_eval, cfg.grid_size)
    K = cfg.eval_steps
    tgt = pdes3d.rollout_trajectory(spec, x0, K)
    pred = _rollout(model, params, x0, K, cfg.output_clip)
    pf, tf = pred[-1], tgt[-1]
    q1 = max(1, K // 4) - 1
    rel_q1 = metrics.rel_l2(pred[q1], tgt[q1])
    rel_T = metrics.rel_l2(pf, tf)
    one = jax.jit(lambda x: model.apply(params, x))
    infer = metrics.time_callable(one, x0, n_runs=5)
    cm = float(jnp.mean(jnp.abs(physics3d.total_mass(pf) - physics3d.total_mass(x0))))
    return {
        "rel_l2": rel_T, "mse": metrics.mse(pf, tf), "rmse": metrics.rmse(pf, tf),
        "mae": metrics.mae(pf, tf), "max_abs_err": metrics.max_abs_error(pf, tf),
        "psnr": metrics.psnr(pf, tf), "ssim": metrics.ssim_like(pf, tf),
        "rel_l2_t_q1": rel_q1, "rel_l2_t_final": rel_T,
        "error_growth_ratio": rel_T / (rel_q1 + 1e-8),
        "conservation_err": cm, "params": metrics.param_count(params),
        "infer_s_per_step": infer,
    }
