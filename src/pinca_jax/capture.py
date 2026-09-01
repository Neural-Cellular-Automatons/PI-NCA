"""Capture trajectories once so figures can be rebuilt forever, without retraining.

Training a model is the expensive part of a figure; drawing it is not. Previously
`viz`, `viz3d` and `viz3d_volume` each retrained from scratch — and `viz3d` and
`viz3d_volume` trained the *same* model twice for the same phenomenon.

This module trains once per phenomenon, rolls the solver (analytic) and the model
forward from one held-out IC, and writes the raw arrays to `results/traj/*.npz`:

    <pde>_2d.npz   solver, model : (T+1, H, W)        channel 0
    <pde>_3d.npz   solver, model : (T+1, D, H, W)     channel 0, full volumes

Every montage, GIF, rotating 3-D render, and anything you write later can then be
built from those files on any machine — no GPU, no jax, no retraining:

    python -m pinca_jax.viz          --npz results/traj/heat_2d.npz
    python -m pinca_jax.viz3d        --npz results/traj/heat_3d.npz
    python -m pinca_jax.viz3d_volume --npz results/traj/heat_3d.npz

The arrays are plain numpy, so `np.load(path)["model"]` is all any other tool needs.

Run:  python -m pinca_jax.capture --dims both
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import ic, ic3d, env
from .equations import pdes, pdes3d
from .harness import EmuConfig, train_emulator
from .harness3d import Emu3DConfig, train as train3d
from .models import registry
from .viz import DEFAULT_ARCH
from .viz3d import DEFAULT as DEFAULT_3D

TRAJ_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "traj")

PDES_2D = ["heat", "allen_cahn", "nagumo", "adv_diff", "gray_scott", "shallow_water",
           "fitzhugh_nagumo", "wave", "cahn_hilliard", "navier_stokes"]
PDES_3D = ["heat", "adv_diff", "allen_cahn", "nagumo", "gray_scott", "fitzhugh_nagumo"]


def _stride_for_budget(n_frames, frame_bytes, max_mb):
    """Keep every k-th frame so 2 x trajectory stays under max_mb. Never drops below 2."""
    if max_mb <= 0:
        return 1
    budget = max_mb * 1e6 / 2.0                      # solver + model
    k = 1
    while n_frames // k > 2 and (n_frames // k) * frame_bytes > budget:
        k += 1
    return k


def _write(path, solver, model, meta, stride):
    os.makedirs(TRAJ_DIR, exist_ok=True)
    np.savez_compressed(path, solver=solver, model=model, stride=np.int32(stride),
                        **{k: np.asarray(v) for k, v in meta.items()})
    mb = os.path.getsize(path) / 1e6
    print(f"  wrote {os.path.relpath(path)}  {solver.shape} x2  ({mb:.1f} MB on disk)")


def _rollout_frames(step_fn, x0, steps, pick):
    """Advance `steps` times, keeping channel 0 of the frames whose index is in `pick`."""
    keep, x = [], x0
    if 0 in pick:
        keep.append(np.asarray(x[0, ..., 0]))
    for i in range(1, steps + 1):
        x = step_fn(x)
        if i in pick:
            keep.append(np.asarray(x[0, ..., 0]))
    return np.stack(keep).astype(np.float32)


def capture_2d(pde, grid, epochs, eval_steps, seed, max_mb):
    arch = DEFAULT_ARCH.get(pde, "multiscale_flux_nca")
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    C = pdes.REGISTRY[pde].channels
    clip = (-1.0, 1.0) if pde in ("cahn_hilliard", "allen_cahn") else None
    cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=min(16, eval_steps),
                    eval_steps=eval_steps, epochs=epochs, seed=seed, output_clip=clip)
    print(f"[capture] {pde} 2-D / {arch}: training ({epochs} ep, grid {grid})...")
    tr = train_emulator(registry.REGISTRY[arch].make(C), cfg)
    model, params = tr["model"], tr["params"]

    x0 = ic.make_state(jax.random.PRNGKey(seed + 777), pde, 1, grid)
    stride = _stride_for_budget(eval_steps + 1, grid * grid * 4, max_mb)
    pick = set(range(0, eval_steps + 1, stride)) | {eval_steps}

    def model_step(x):
        y = model.apply(params, x)
        return jnp.clip(y, clip[0], clip[1]) if clip else y

    solver = _rollout_frames(lambda x: spec.step(x, spec.params), x0, eval_steps, pick)
    mdl = _rollout_frames(model_step, x0, eval_steps, pick)
    _write(os.path.join(TRAJ_DIR, f"{pde}_2d.npz"), solver, mdl,
           {"pde": pde, "arch": arch, "grid": grid, "epochs": epochs,
            "eval_steps": eval_steps, "seed": seed, "dims": 2}, stride)


def capture_3d(pde, grid, epochs, eval_steps, seed, max_mb):
    spec = pdes3d.REGISTRY[pde]
    C = spec.channels
    ctor_fn, clip = DEFAULT_3D[pde]
    cfg = Emu3DConfig(pde=pde, grid_size=grid, rollout_steps=8, eval_steps=eval_steps,
                      epochs=epochs, seed=seed, output_clip=clip)
    arch = type(ctor_fn(C)).__name__
    print(f"[capture] {pde} 3-D / {arch}: training ({epochs} ep, {grid}^3)...")
    tr = train3d(lambda: ctor_fn(C), cfg)
    model, params = tr["model"], tr["params"]

    x0 = ic3d.make_state(jax.random.PRNGKey(seed + 777), pde, 1, grid)
    stride = _stride_for_budget(eval_steps + 1, grid ** 3 * 4, max_mb)
    pick = set(range(0, eval_steps + 1, stride)) | {eval_steps}

    def model_step(x):
        y = model.apply(params, x)
        return jnp.clip(y, clip[0], clip[1]) if clip else y

    solver = _rollout_frames(lambda x: spec.step(x, spec.params), x0, eval_steps, pick)
    mdl = _rollout_frames(model_step, x0, eval_steps, pick)
    _write(os.path.join(TRAJ_DIR, f"{pde}_3d.npz"), solver, mdl,
           {"pde": pde, "arch": arch, "grid": grid, "epochs": epochs,
            "eval_steps": eval_steps, "seed": seed, "dims": 3}, stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="both", choices=["2d", "3d", "both"])
    ap.add_argument("--pdes", default=None, help="comma-separated subset")
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval", type=int, default=64)
    ap.add_argument("--grid3d", type=int, default=16)
    ap.add_argument("--epochs3d", type=int, default=200)
    ap.add_argument("--eval3d", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-mb", type=float, default=64.0,
                    help="per-phenomenon cap; frames are strided out to fit (0 = no cap)")
    args = ap.parse_args()
    env.banner("capture")
    os.makedirs(TRAJ_DIR, exist_ok=True)

    if args.dims in ("2d", "both"):
        for pde in (args.pdes.split(",") if args.pdes else PDES_2D):
            if pde in pdes.REGISTRY:
                capture_2d(pde, args.grid, args.epochs, args.eval, args.seed, args.max_mb)
    if args.dims in ("3d", "both"):
        for pde in (args.pdes.split(",") if args.pdes else PDES_3D):
            if pde in pdes3d.REGISTRY:
                capture_3d(pde, args.grid3d, args.epochs3d, args.eval3d, args.seed,
                           args.max_mb)
    print(f"[capture] done -> {os.path.abspath(TRAJ_DIR)}")


if __name__ == "__main__":
    main()
