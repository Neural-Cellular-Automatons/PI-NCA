"""DeepONet for the heat solution operator G: u0 ↦ u(·,T)  (JAX).

Operator-learning baseline (Lu et al., arXiv:1910.03193). Unlike the autoregressive
emulators, DeepONet predicts the state at a fixed horizon T directly from the initial
condition and generalises ACROSS initial conditions (like FNO, unlike PINN). Output:

    u(x; u0) ≈ Σ_k  branch_k(u0_sensors) · trunk_k(x)   (+ bias)

- branch: MLP over the IC sampled on the grid (sensors).
- trunk : MLP over query coords with Fourier features (periodic domain).

Trained on (u0, u_T) pairs from the solver; evaluated by relative-L2 at T on held-out
ICs (cross-IC generalisation).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from . import ic, metrics
from .equations import pdes


@dataclass(frozen=True)
class DeepONetConfig:
    grid_size: int = 24
    n_steps: int = 24          # horizon T = n_steps * dt
    p: int = 64                # latent basis size
    trunk_modes: int = 6
    width: int = 128
    batch: int = 32
    iters: int = 3000
    lr: float = 1e-3
    seed: int = 0
    n_eval: int = 32


class Branch(nn.Module):
    p: int; width: int
    @nn.compact
    def __call__(self, u0_flat):           # (B, N*N)
        h = nn.gelu(nn.Dense(self.width)(u0_flat))
        h = nn.gelu(nn.Dense(self.width)(h))
        return nn.Dense(self.p)(h)          # (B, p)


class Trunk(nn.Module):
    p: int; width: int; modes: int
    @nn.compact
    def __call__(self, xy):                 # (Q, 2) in [0,1)
        ks = jnp.arange(1, self.modes + 1) * 2.0 * jnp.pi
        x, y = xy[:, 0:1], xy[:, 1:2]
        feats = jnp.concatenate(
            [jnp.sin(ks * x), jnp.cos(ks * x), jnp.sin(ks * y), jnp.cos(ks * y)], axis=-1)
        h = nn.gelu(nn.Dense(self.width)(feats))
        h = nn.gelu(nn.Dense(self.width)(h))
        return nn.gelu(nn.Dense(self.p)(h))  # (Q, p)


class DeepONet(nn.Module):
    p: int = 64; width: int = 128; trunk_modes: int = 6
    @nn.compact
    def __call__(self, u0_flat, xy):
        b = Branch(self.p, self.width)(u0_flat)      # (B,p)
        t = Trunk(self.p, self.width, self.trunk_modes)(xy)  # (Q,p)
        bias = self.param("bias", nn.initializers.zeros, ())
        return jnp.einsum("bp,qp->bq", b, t) + bias  # (B,Q)


def _grid_coords(N):
    xs = (jnp.arange(N) + 0.5) / N
    XX, YY = jnp.meshgrid(xs, xs, indexing="xy")
    return jnp.stack([XX.ravel(), YY.ravel()], axis=-1)  # (N*N,2)


def train_and_eval(cfg: DeepONetConfig):
    spec = pdes.REGISTRY["heat"]
    N = cfg.grid_size
    xy = _grid_coords(N)
    key = jax.random.PRNGKey(cfg.seed)
    model = DeepONet(cfg.p, cfg.width, cfg.trunk_modes)

    key, ik = jax.random.split(key)
    u0 = ic.make_state(ik, "heat", 1, N)
    params = model.init(ik, u0.reshape(1, -1), xy)
    opt = optax.adam(cfg.lr); opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, key):
        u0 = ic.make_state(key, "heat", cfg.batch, N)             # (B,N,N,1)
        uT = pdes.rollout(spec, u0, cfg.n_steps)[..., 0]          # (B,N,N)
        def loss_fn(p):
            pred = model.apply(p, u0.reshape(cfg.batch, -1), xy)  # (B,N*N)
            return jnp.mean((pred - uT.reshape(cfg.batch, -1)) ** 2)
        l, g = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, l

    t0 = time.time()
    for it in range(cfg.iters):
        key, sk = jax.random.split(key)
        params, opt_state, l = step(params, opt_state, sk)
        if it % max(1, cfg.iters // 6) == 0 or it == cfg.iters - 1:
            print(f"  it {it:5d} | loss {float(l):.3e}")
    wall = time.time() - t0

    # eval on held-out ICs
    ek = jax.random.PRNGKey(cfg.seed + 9999)
    u0 = ic.make_state(ek, "heat", cfg.n_eval, N)
    uT = pdes.rollout(spec, u0, cfg.n_steps)[..., 0].reshape(cfg.n_eval, -1)
    pred = model.apply(params, u0.reshape(cfg.n_eval, -1), xy)
    rel = float(jnp.sqrt(jnp.sum((pred - uT) ** 2)) / (jnp.sqrt(jnp.sum(uT ** 2)) + 1e-8))
    return {"rel_l2_at_T": rel, "wall_s": wall,
            "params": metrics.param_count(params)}


if __name__ == "__main__":
    import statistics
    rels = []
    for s in range(3):
        out = train_and_eval(DeepONetConfig(seed=s))
        print(f"seed {s}: rel-L2@T {out['rel_l2_at_T']:.3e} | {out['params']} params | {out['wall_s']:.1f}s")
        rels.append(out["rel_l2_at_T"])
    print(f"\n[DeepONet heat] rel-L2@T {statistics.fmean(rels):.3e} ± "
          f"{statistics.stdev(rels):.1e} (n={len(rels)})")
