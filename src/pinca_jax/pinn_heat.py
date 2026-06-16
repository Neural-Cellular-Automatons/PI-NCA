"""Physics-Informed Neural Network for the 2-D periodic heat IVP (JAX).

Distinct paradigm from the emulators: a continuous field u_φ(x,y,t) trained to
satisfy the PDE residual + initial condition (periodic BCs are *exact* via Fourier
features, so no BC loss term). Solves a SINGLE initial-value problem per training
run (no cross-IC generalisation) — the canonical PINN setting (Raissi et al.,
arXiv:1711.10561).

PDE (grid units, matching equations/heat.py): u_t = α (u_xx + u_yy), α=0.5.
Space x,y ∈ [0,N) periodic; time t ∈ [0, T], T = n_steps·dt (dt=0.1), so the
solver state at step k is the reference at t = k·dt.

Coordinates are normalised to xn,yn ∈ [0,1), tn ∈ [0,1]; chain rule restores
physical derivatives (∂x = ∂xn / N, ∂t = ∂tn / T).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from . import ic
from .equations import pdes


class HeatPINN(nn.Module):
    """u_φ(xn,yn,tn) → scalar. Fourier features in space bake in periodicity."""
    n_modes: int = 6
    width: int = 64
    depth: int = 4

    @nn.compact
    def __call__(self, xn, yn, tn):
        ks = jnp.arange(1, self.n_modes + 1) * 2.0 * jnp.pi
        feats = [jnp.sin(ks * xn), jnp.cos(ks * xn),
                 jnp.sin(ks * yn), jnp.cos(ks * yn), jnp.atleast_1d(tn)]
        h = jnp.concatenate(feats, axis=-1)
        for _ in range(self.depth):
            h = jnp.tanh(nn.Dense(self.width)(h))
        return nn.Dense(1)(h)[..., 0]


@dataclass(frozen=True)
class PINNConfig:
    grid_size: int = 16
    n_steps: int = 32          # reference horizon → T = n_steps*dt
    dt: float = 0.1
    alpha: float = 0.5
    n_collocation: int = 2048
    n_ic: int = 1024
    iters: int = 4000
    lr: float = 2e-3
    lambda_ic: float = 10.0
    seed: int = 0


def _u_scalar(model, params, x, y, t):
    return model.apply(params, x, y, t)


def make_loss(model, cfg, ic_grid):
    N, T = cfg.grid_size, cfg.n_steps * cfg.dt
    # bilinear-periodic sample of the IC grid at normalised coords (for IC loss)
    def ic_at(xn, yn):
        gx = xn * N
        gy = yn * N
        x0 = jnp.floor(gx).astype(int) % N; x1 = (x0 + 1) % N
        y0 = jnp.floor(gy).astype(int) % N; y1 = (y0 + 1) % N
        fx = gx - jnp.floor(gx); fy = gy - jnp.floor(gy)
        g = ic_grid  # (N,N)
        v = (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x1] * fx * (1 - fy)
             + g[y1, x0] * (1 - fx) * fy + g[y1, x1] * fx * fy)
        return v

    def u(params, xn, yn, tn):
        return _u_scalar(model, params, xn, yn, tn)

    def residual(params, xn, yn, tn):
        u_t = jax.grad(u, argnums=3)(params, xn, yn, tn) / T
        u_x = lambda a, b, c: jax.grad(u, argnums=1)(params, a, b, c)
        u_y = lambda a, b, c: jax.grad(u, argnums=2)(params, a, b, c)
        u_xx = jax.grad(u_x, argnums=0)(xn, yn, tn) / (N ** 2)
        u_yy = jax.grad(u_y, argnums=1)(xn, yn, tn) / (N ** 2)
        return u_t - cfg.alpha * (u_xx + u_yy)

    def loss(params, key):
        kc, ki = jax.random.split(key)
        pc = jax.random.uniform(kc, (cfg.n_collocation, 3))
        r = jax.vmap(lambda p: residual(params, p[0], p[1], p[2]))(pc)
        l_pde = jnp.mean(r ** 2)
        pi = jax.random.uniform(ki, (cfg.n_ic, 2))
        u0 = jax.vmap(lambda p: u(params, p[0], p[1], 0.0))(pi)
        tgt = jax.vmap(lambda p: ic_at(p[0], p[1]))(pi)
        l_ic = jnp.mean((u0 - tgt) ** 2)
        return l_pde + cfg.lambda_ic * l_ic, (l_pde, l_ic)

    return loss


def train_and_eval(cfg: PINNConfig):
    key = jax.random.PRNGKey(cfg.seed)
    key, ik = jax.random.split(key)
    # single IC (batch 1) and its solver reference trajectory
    s0 = ic.make_state(ik, "heat", 1, cfg.grid_size)          # (1,N,N,1)
    ic_grid = s0[0, :, :, 0]
    spec = pdes.REGISTRY["heat"]
    ref_T = pdes.rollout(spec, s0, cfg.n_steps)[0, :, :, 0]    # reference at t=T

    model = HeatPINN()
    params = model.init(ik, jnp.array(0.1), jnp.array(0.2), jnp.array(0.0))
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)
    loss_fn = make_loss(model, cfg, ic_grid)

    @jax.jit
    def step(params, opt_state, key):
        (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, key)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, l, parts

    t0 = time.time()
    for it in range(cfg.iters):
        key, sk = jax.random.split(key)
        params, opt_state, l, parts = step(params, opt_state, sk)
        if it % max(1, cfg.iters // 8) == 0 or it == cfg.iters - 1:
            print(f"  it {it:5d} | loss {float(l):.3e} | pde {float(parts[0]):.2e} | ic {float(parts[1]):.2e}")
    wall = time.time() - t0

    # evaluate u_φ(.,T) on the grid vs solver reference
    N = cfg.grid_size
    xs = (jnp.arange(N) + 0.5) / N
    XX, YY = jnp.meshgrid(xs, xs, indexing="xy")
    pred_T = jax.vmap(lambda x, y: model.apply(params, x, y, 1.0))(XX.ravel(), YY.ravel()).reshape(N, N)
    num = jnp.sqrt(jnp.sum((pred_T - ref_T) ** 2))
    den = jnp.sqrt(jnp.sum(ref_T ** 2)) + 1e-8
    rel_l2 = float(num / den)
    return {"rel_l2_at_T": rel_l2, "wall_s": wall,
            "params": int(sum(x.size for x in jax.tree_util.tree_leaves(params)))}


if __name__ == "__main__":
    out = train_and_eval(PINNConfig())
    print(f"\n[PINN heat] rel-L2 at T = {out['rel_l2_at_T']:.3e} | "
          f"{out['params']} params | {out['wall_s']:.1f}s")
