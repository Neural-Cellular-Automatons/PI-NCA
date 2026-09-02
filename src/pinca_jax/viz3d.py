"""3-D analytic-vs-model GIFs + error maps (mid-depth slice over time).

A 3-D volume (B,D,H,W,C) can't be a flat GIF, so for each phenomenon we train a 3-D
model, roll out the solver (analytic) and the model, and visualise the **mid-depth
z-slice** (D/2) of channel 0 over time:
  - results/gifs/<pde>_3d_{analytic,model,error}.gif   (local, gitignored)
  - docs/figures/<pde>_3d_comparison.png               (committed montage)

Run:  python -m pinca_jax.viz3d --pde heat
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import ic3d, viz
from .equations import pdes3d
from .harness3d import Emu3DConfig, train
import pinca_jax.models3d as M

# regime-appropriate winner per 3-D phenomenon (from results/bench3d_*.md)
DEFAULT = {
    "heat": (lambda C: M.FluxNCA3D(), None),
    "adv_diff": (lambda C: M.FNO3D(out_channels=C), None),
    "allen_cahn": (lambda C: M.FNO3D(out_channels=C), None),
    "nagumo": (lambda C: M.NCA3D(out_channels=C), None),
    "gray_scott": (lambda C: M.NCA3D(out_channels=C), None),
    "fitzhugh_nagumo": (lambda C: M.FNO3D(out_channels=C), None),
}


def render(pde, grid=16, epochs=120, eval_steps=32, every=2, seed=42, npz=None):
    """Mid-depth-slice montage + GIFs for one 3-D phenomenon.

    `npz` = a trajectory file written by `pinca_jax.capture` (full volumes). The
    z-mid slice is taken from it and nothing is retrained — use this to redraw or
    restyle figures later, on any machine.
    """
    if npz is not None:
        d = np.load(npz)
        sol_v, mdl_v = d["solver"], d["model"]          # (T+1, D, H, W)
        zmid = sol_v.shape[1] // 2
        solver_traj = jnp.asarray(sol_v[:, zmid])
        model_traj = jnp.asarray(mdl_v[:, zmid])
        pde = str(d["pde"]) if "pde" in d else pde
        arch = str(d["arch"]) if "arch" in d else "?"
        eval_steps = solver_traj.shape[0] - 1
        print(f"[viz3d] {pde} / {arch}: from {npz} {sol_v.shape}, no training")
        err = jnp.abs(solver_traj - model_traj)
    else:
        spec = pdes3d.REGISTRY[pde]
        C = spec.channels
        ctor_fn, clip = DEFAULT[pde]
        arch = type(ctor_fn(C)).__name__
        cfg = Emu3DConfig(pde=pde, grid_size=grid, rollout_steps=8, eval_steps=eval_steps,
                          epochs=epochs, seed=seed, output_clip=clip)
        print(f"[viz3d] {pde}: training 3D model ({epochs} ep, {grid}^3)...")
        tr = train(lambda: ctor_fn(C), cfg)
        model, params = tr["model"], tr["params"]

        key = jax.random.PRNGKey(seed + 777)
        x0 = ic3d.make_state(key, pde, 1, grid)
        zmid = grid // 2

        def traj(step_fn):
            outs = [x0]; x = x0
            for _ in range(eval_steps):
                x = step_fn(x); outs.append(x)
            # mid-depth slice of channel 0 -> (T+1, H, W)
            return jnp.concatenate([o[0:1, zmid, :, :, 0] for o in outs], axis=0)

        solver_traj = traj(lambda x: spec.step(x, spec.params))
        if clip:
            model_step = lambda x: jnp.clip(model.apply(params, x), clip[0], clip[1])
        else:
            model_step = lambda x: model.apply(params, x)
        model_traj = traj(model_step)
        err = jnp.abs(solver_traj - model_traj)

    vmin, vmax = float(jnp.min(solver_traj)), float(jnp.max(solver_traj))
    emax = float(jnp.quantile(err, 0.99)) + 1e-6
    os.makedirs(viz.GIF_DIR, exist_ok=True); os.makedirs(viz.FIG_DIR, exist_ok=True)
    name = f"{pde}_3d"
    montage = viz._save_montage(name, arch + " (z-mid slice)", solver_traj, model_traj, err,
                                vmin, vmax, emax, eval_steps)
    try:
        viz._save_gifs(name, solver_traj, model_traj, err, vmin, vmax, emax, every)
    except Exception as e:
        print(f"[viz3d] gif step skipped ({repr(e)[:80]})")
    rel = float(jnp.linalg.norm(model_traj[-1] - solver_traj[-1]) /
                (jnp.linalg.norm(solver_traj[-1]) + 1e-8))
    print(f"[viz3d] {pde}: {montage} | final-slice rel-err {rel:.3e}")
    return {"pde": pde, "arch": arch, "final_rel_err": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--npz", default=None,
                    help="render from a pinca_jax.capture trajectory file (no training)")
    args = ap.parse_args()
    render(args.pde, grid=args.grid, epochs=args.epochs, npz=args.npz)


if __name__ == "__main__":
    main()
