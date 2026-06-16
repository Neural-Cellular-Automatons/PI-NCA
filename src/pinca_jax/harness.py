"""Model-agnostic emulator train + eval harness (JAX).

Every neural architecture in the study (plain NCA, conservative PI-NCA, FNO, ...)
is treated as a one-step **autoregressive emulator** g: state -> next_state and
trained by distillation against a differentiable PDE solver teacher
(equations/pdes.py). This makes the comparison apples-to-apples: same data, same
teacher, same horizon, same metrics. PINNs are evaluated separately (they solve a
single IVP in continuous (x,t) and do not fit the emulator interface) — see the
PINN branch.

Reproducibility: explicit PRNG seeds; multi-seed runs return mean±std (never a
single run). Reduced-scale CPU configs; identical code runs full-scale on GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from . import ic, metrics
from .equations import pdes


@dataclass(frozen=True)
class EmuConfig:
    pde: str = "heat"
    grid_size: int = 32
    batch: int = 16
    rollout_steps: int = 16        # training horizon (BPTT through scan)
    eval_steps: int = 64           # evaluation horizon (generalisation/stability)
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0
    n_eval: int = 8

    def spec(self):
        # use the numerically stable teacher config where one exists (e.g. gray_scott)
        return pdes.STABLE.get(self.pde, pdes.REGISTRY[self.pde])


def _emu_rollout(model, params, x0, steps):
    def body(x, _):
        return model.apply(params, x), None
    xf, _ = jax.lax.scan(body, x0, xs=None, length=steps)
    return xf


def train_emulator(model_ctor, cfg: EmuConfig, verbose=False):
    """Train one emulator. model_ctor() -> Flax module with __call__(state)->state."""
    spec = cfg.spec()
    key = jax.random.PRNGKey(cfg.seed)
    key, ik = jax.random.split(key)
    model = model_ctor()
    dummy = ic.make_state(ik, cfg.pde, 1, cfg.grid_size)
    params = model.init(ik, dummy)

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, k):
        x0 = ic.make_state(k, cfg.pde, cfg.batch, cfg.grid_size)
        target = pdes.rollout(spec, x0, cfg.rollout_steps)

        def loss_fn(p):
            pred = _emu_rollout(model, p, x0, cfg.rollout_steps)
            return jnp.mean((pred - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = []
    t0 = time.time()
    for e in range(cfg.epochs):
        key, sk = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, sk)
        losses.append(float(loss))
        if verbose and (e % max(1, cfg.epochs // 10) == 0 or e == cfg.epochs - 1):
            print(f"  epoch {e:4d} | loss {float(loss):.4e}")
    return {"params": params, "model": model, "losses": losses,
            "wall_s": time.time() - t0}


def evaluate_emulator(model, params, cfg: EmuConfig):
    """Long-horizon eval vs the teacher → metric dict (single seed)."""
    spec = cfg.spec()
    key = jax.random.PRNGKey(cfg.seed + 10_000)
    x0 = ic.make_state(key, cfg.pde, cfg.n_eval, cfg.grid_size)
    target = pdes.rollout(spec, x0, cfg.eval_steps)
    pred = _emu_rollout(model, params, x0, cfg.eval_steps)

    one_step = jax.jit(lambda x: model.apply(params, x))
    return {
        "mse": metrics.mse(pred, target),
        "rel_l2": metrics.rel_l2(pred, target),
        "rmse": metrics.rmse(pred, target),
        "psnr": metrics.psnr(pred, target),
        "conservation_err": metrics.conservation_error(pred, x0),
        "bc_residual": metrics.periodic_bc_residual(pred),
        "grad_energy": metrics.gradient_energy(pred),
        "params": metrics.param_count(params),
        "infer_s_per_step": metrics.time_callable(one_step, x0),
        "train_wall_s": None,  # filled by run_multiseed
    }


def run_multiseed(model_ctor, cfg: EmuConfig, seeds=(0, 1, 2)):
    """Train+eval across seeds → (per-seed list, aggregated mean±std dict)."""
    runs = []
    for s in seeds:
        c = EmuConfig(**{**cfg.__dict__, "seed": s})
        tr = train_emulator(model_ctor, c)
        ev = evaluate_emulator(tr["model"], tr["params"], c)
        ev["train_wall_s"] = tr["wall_s"]
        ev["final_train_loss"] = tr["losses"][-1]
        runs.append(ev)
    agg = metrics.aggregate_runs([{k: v for k, v in r.items() if v is not None} for r in runs])
    return runs, agg
