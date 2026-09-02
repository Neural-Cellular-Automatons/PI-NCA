"""End-to-end JAX/Optax trainer for the conservative heat PI-NCA.

Migrated from the PyTorch training loop in `PI NCA_v1.py`. Demonstrates the
modern-JAX idioms the migration targets:
- jax.lax.scan for the NCA rollout (O(1) python overhead, full BPTT through it),
- optax for Adam + weight decay,
- jax.jit on the whole train step,
- explicit PRNG threading for reproducibility.

Run a smoke check:  python -m pinca_jax.train_nca --smoke
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from .configs import SMOKE, CPU_REDUCED, HeatNCAConfig
from .data import make_blobs
from .equations import heat
from .models.flux_nca import DeepFluxNCA
from .physics import conserve_energy, total_mass


def nca_rollout(params, model, x0, steps, conserve):
    """Roll the NCA forward `steps`, optionally projecting mass each step."""
    target = total_mass(x0)

    def body(x, _):
        x = model.apply(params, x)
        if conserve:
            x = conserve_energy(x, target)
        return x, None

    x_final, _ = jax.lax.scan(body, x0, xs=None, length=steps)
    return x_final


def make_train_step(model, cfg: HeatNCAConfig, optimizer):
    steps = cfg.rollout_steps  # static -> single compile

    @jax.jit
    def train_step(params, opt_state, key):
        x0 = make_blobs(key, cfg.batch_size, cfg.grid_size, cfg.n_blobs,
                        cfg.blob_sigma_frac, cfg.amp_low, cfg.amp_high)
        target = heat.rollout(x0, cfg.alpha_dt, steps)  # differentiable teacher

        def loss_fn(p):
            pred = nca_rollout(p, model, x0, steps, cfg.conserve)
            return jnp.mean((pred - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def train(cfg: HeatNCAConfig, verbose: bool = True):
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)

    model = DeepFluxNCA()
    dummy = jnp.zeros((1, cfg.grid_size, cfg.grid_size, 1))
    params = model.init(init_key, dummy)

    optimizer = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, cfg, optimizer)

    losses = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        key, sk = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, sk)
        losses.append(float(loss))
        if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:4d} | loss {float(loss):.4e}")
    wall = time.time() - t0
    return {"params": params, "losses": losses, "wall_s": wall, "model": model}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check")
    args = ap.parse_args()
    cfg = SMOKE if args.smoke else CPU_REDUCED
    out = train(cfg)
    l0, lf = out["losses"][0], out["losses"][-1]
    print(f"\n[{'SMOKE' if args.smoke else 'CPU_REDUCED'}] "
          f"loss {l0:.3e} -> {lf:.3e}  ({l0/lf:.1f}x)  in {out['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
