"""Darcy flow — steady elliptic operator learning  G: a(x) ↦ u(x)  (JAX).

-∇·(a(x) ∇u) = f  on the unit square, u=0 on the boundary, f=1. The classic FNO
benchmark. This is a STEADY (not time-dependent) GLOBAL problem: the solution at
every point depends on the whole coefficient field, so it stresses global vs local
inductive biases differently from the autoregressive emulators.

We compare:
- FNO (direct operator a↦u, residual=False) — global spectral mixing.
- an NCA "learned iterative solver": conditioned on a, relax u from 0 over K local
  steps (a local solver must propagate boundary information K cells — the locality
  test on an elliptic problem).

Ground truth u is from a dense sparse-free solve of the 5-point variable-coefficient
operator with Dirichlet BCs.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .models.fno import FNO2d


@dataclass(frozen=True)
class DarcyConfig:
    grid: int = 20            # interior solve is grid^2 unknowns (dense) — keep modest
    n_train: int = 256
    n_test: int = 64
    nca_steps: int = 24
    iters: int = 1500
    batch: int = 32
    lr: float = 1e-3
    seed: int = 0


# ---------- ground-truth Darcy data ---------- #
def _sample_a(rng, n, N):
    """Random positive permeability fields a(x): smoothed noise → thresholded log-normal."""
    z = rng.standard_normal((n, N, N)).astype(np.float32)
    # smooth with a few box blurs (spatial correlation)
    for _ in range(3):
        z = (z + np.roll(z, 1, 1) + np.roll(z, -1, 1) + np.roll(z, 1, 2) + np.roll(z, -1, 2)) / 5.0
    a = np.where(z > 0, 9.0, 3.0).astype(np.float32)  # piecewise high/low contrast
    return a


def _solve_darcy(a):
    """Solve -∇·(a∇u)=1, u=0 on boundary, via dense linear solve (harmonic-mean faces)."""
    N = a.shape[0]
    h = 1.0 / (N + 1)
    idx = lambda i, j: i * N + j
    A = np.zeros((N * N, N * N), np.float32)
    b = np.full(N * N, h * h, np.float32)

    def face(ai, aj):  # harmonic mean of adjacent cell permeabilities
        return 2 * ai * aj / (ai + aj)

    for i in range(N):
        for j in range(N):
            p = idx(i, j); diag = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    w = face(a[i, j], a[ni, nj]); diag += w
                    A[p, idx(ni, nj)] = -w
                else:
                    diag += face(a[i, j], a[i, j])  # Dirichlet u=0 outside
            A[p, p] = diag
    u = np.linalg.solve(A, b)
    return u.reshape(N, N).astype(np.float32)


def make_dataset(cfg: DarcyConfig):
    rng = np.random.default_rng(cfg.seed)
    a = _sample_a(rng, cfg.n_train + cfg.n_test, cfg.grid)
    u = np.stack([_solve_darcy(ai) for ai in a])
    a = a[..., None]; u = u[..., None]
    ntr = cfg.n_train
    return (jnp.asarray(a[:ntr]), jnp.asarray(u[:ntr]),
            jnp.asarray(a[ntr:]), jnp.asarray(u[ntr:]))


# ---------- NCA iterative solver ---------- #
class DarcyNCA(nn.Module):
    """Relax u from 0 over `steps` local updates, conditioned on the fixed a-field."""
    steps: int = 24
    features: int = 48
    hidden: int = 64

    @nn.compact
    def __call__(self, a):                      # a: (B,N,N,1)
        u = jnp.zeros_like(a)
        perceive = nn.Conv(self.features, (3, 3), padding="SAME", name="perceive")
        h1 = nn.Conv(self.hidden, (1, 1), name="h1")
        h2 = nn.Conv(1, (1, 1), use_bias=False, kernel_init=nn.initializers.zeros, name="upd")
        for _ in range(self.steps):
            inp = jnp.concatenate([u, a], axis=-1)
            z = nn.relu(perceive(inp)); z = nn.relu(h1(z))
            u = u + h2(z)
            u = u * _interior_mask(u.shape)     # enforce u=0 on boundary (Dirichlet)
        return u


def _interior_mask(shape):
    B, N, _, C = shape
    m = jnp.ones((N, N)).at[0, :].set(0).at[-1, :].set(0).at[:, 0].set(0).at[:, -1].set(0)
    return m[None, :, :, None]


def _rel_l2(pred, tgt):
    return float(jnp.sqrt(jnp.sum((pred - tgt) ** 2)) / (jnp.sqrt(jnp.sum(tgt ** 2)) + 1e-8))


def train_model(name, model_ctor, atr, utr, ate, ute, cfg):
    key = jax.random.PRNGKey(cfg.seed)
    model = model_ctor()
    params = model.init(key, atr[:1])
    opt = optax.adam(cfg.lr); opt_state = opt.init(params)
    n = atr.shape[0]

    @jax.jit
    def step(params, opt_state, key):
        i = jax.random.randint(key, (cfg.batch,), 0, n)
        a, u = atr[i], utr[i]
        def loss(p): return jnp.mean((model.apply(p, a) - u) ** 2)
        l, g = jax.value_and_grad(loss)(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, l

    t0 = time.time()
    for it in range(cfg.iters):
        key, sk = jax.random.split(key)
        params, opt_state, l = step(params, opt_state, sk)
    wall = time.time() - t0
    rel = _rel_l2(model.apply(params, ate), ute)
    nparams = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"  {name:10s} rel-L2 {rel:.3e} | {nparams} params | {wall:.1f}s")
    return {"rel_l2": rel, "params": nparams, "wall_s": wall}


def run(seeds=(0, 1)):
    out = {"fno": [], "nca_solver": []}
    for s in seeds:
        cfg = DarcyConfig(seed=s)
        atr, utr, ate, ute = make_dataset(cfg)
        print(f"[Darcy seed {s}] dataset {atr.shape} -> {utr.shape}")
        out["fno"].append(train_model(
            "fno", lambda: FNO2d(out_channels=1, width=24, modes=8, depth=4, residual=False),
            atr, utr, ate, ute, cfg))
        out["nca_solver"].append(train_model(
            "nca_solver", lambda: DarcyNCA(steps=cfg.nca_steps),
            atr, utr, ate, ute, cfg))
    for k, runs in out.items():
        rels = [r["rel_l2"] for r in runs]
        print(f"[{k}] rel-L2 {statistics.fmean(rels):.3e} ± "
              f"{(statistics.stdev(rels) if len(rels)>1 else 0):.1e}  ({runs[0]['params']} params)")
    return out


if __name__ == "__main__":
    run()
