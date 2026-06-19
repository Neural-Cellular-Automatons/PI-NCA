"""TRUE 3-D volumetric visualisations (Axes3D) for 3-D phenomena.

Unlike viz3d.py (which shows a single mid-depth z-slice and therefore "looks 2-D"),
this module renders the full 3-D volume with mpl_toolkits.mplot3d (projection='3d')
so depth + perspective are visible. For a PDE we:
  - train the regime-winner 3-D model (reuse viz3d.DEFAULT),
  - roll out the solver (analytic) and the model from one held-out IC,
  - render volumetric point-clouds at t = 0, T/2, T as a montage, and
  - emit a rotating + evolving animated GIF of the model volume.

Outputs:
  - docs/figures/<pde>_3d_volume.png        (committed montage; rows analytic/model/|error|)
  - results/gifs/<pde>_3d_volume.gif         (local, gitignored)

Run:  python -m pinca_jax.viz3d_volume --pde heat --epochs 60
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from . import ic3d, viz
from .equations import pdes3d
from .harness3d import Emu3DConfig, train
from .viz3d import DEFAULT


def draw_volume(ax, vol, cmap, vmin, vmax, thr_q=0.60):
    """Render a (D,H,W) volume as a semi-transparent point cloud on a 3-D axis.

    Only high-value voxels (>= the thr_q quantile) are drawn, coloured by their
    normalised value. depthshade=True + perspective makes the cloud read as 3-D.
    """
    D, H, W = vol.shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    v = vol.ravel()
    thr = np.quantile(v, thr_q)
    m = v >= thr
    norm = np.clip((v[m] - vmin) / (vmax - vmin + 1e-9), 0, 1)
    ax.scatter(xx.ravel()[m], yy.ravel()[m], zz.ravel()[m], c=norm, cmap=cmap,
               s=18, alpha=0.28, marker='o', depthshade=True, vmin=0, vmax=1)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, D)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=35)


def _model_step_fn(model, params, clip):
    if clip:
        return lambda x: jnp.clip(model.apply(params, x), clip[0], clip[1])
    return lambda x: model.apply(params, x)


def _capture_volumes(step_fn, x0, steps, key_idxs):
    """Roll out `steps` and return the (D,H,W) channel-0 volume at the requested indices."""
    outs = [x0]
    x = x0
    for _ in range(steps):
        x = step_fn(x)
        outs.append(x)
    return {i: np.asarray(outs[i][0, :, :, :, 0]) for i in key_idxs}


def render_volume(pde, grid=16, epochs=80, eval_steps=24, seed=42):
    spec = pdes3d.REGISTRY[pde]
    C = spec.channels
    ctor_fn, clip = DEFAULT[pde]
    cfg = Emu3DConfig(pde=pde, grid_size=grid, rollout_steps=8, eval_steps=eval_steps,
                      epochs=epochs, seed=seed, output_clip=clip)
    arch = type(ctor_fn(C)).__name__
    print(f"[viz3d_volume] {pde}: training {arch} ({epochs} ep, {grid}^3)...")
    tr = train(lambda: ctor_fn(C), cfg)
    model, params = tr["model"], tr["params"]

    key = jax.random.PRNGKey(seed + 777)
    x0 = ic3d.make_state(key, pde, 1, grid)

    solver_step = lambda x: spec.step(x, spec.params)
    model_step = _model_step_fn(model, params, clip)

    # timesteps t = 0, T/2, T
    T = eval_steps
    tcols = [0, T // 2, T]
    sol_vols = _capture_volumes(solver_step, x0, T, tcols)
    mdl_vols = _capture_volumes(model_step, x0, T, tcols)

    # shared field scale from the analytic volumes; separate error scale (99th pct)
    all_sol = np.concatenate([sol_vols[t].ravel() for t in tcols])
    vmin, vmax = float(all_sol.min()), float(all_sol.max())
    err_vols = {t: np.abs(sol_vols[t] - mdl_vols[t]) for t in tcols}
    emax = float(np.quantile(np.concatenate([err_vols[t].ravel() for t in tcols]), 0.99)) + 1e-6

    os.makedirs(viz.GIF_DIR, exist_ok=True)
    os.makedirs(viz.FIG_DIR, exist_ok=True)

    png_path = _save_montage(pde, arch, tcols, sol_vols, mdl_vols, err_vols,
                             vmin, vmax, emax)
    gif_path = _save_rotating_gif(pde, model_step, x0, vmin, vmax)

    print(f"[viz3d_volume] {pde}: montage -> {png_path}")
    print(f"[viz3d_volume] {pde}: gif     -> {gif_path}")
    return {"pde": pde, "arch": arch, "png": png_path, "gif": gif_path}


def _save_montage(pde, arch, tcols, sol_vols, mdl_vols, err_vols, vmin, vmax, emax):
    rows = [("analytic", sol_vols, "inferno", vmin, vmax),
            ("model", mdl_vols, "inferno", vmin, vmax),
            ("|error|", err_vols, "magma", 0.0, emax)]
    nrows, ncols = len(rows), len(tcols)
    fig = plt.figure(figsize=(3.0 * ncols, 3.0 * nrows))
    for r, (label, vols, cmap, lo, hi) in enumerate(rows):
        for c, t in enumerate(tcols):
            idx = r * ncols + c + 1
            ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
            draw_volume(ax, vols[t], cmap, lo, hi)
            if r == 0:
                ax.set_title(f"t={t}", fontsize=10)
            if c == 0:
                # 3-D axes don't take ylabel as a row tag cleanly; use a 2-D text annotation
                ax.text2D(-0.08, 0.5, label, transform=ax.transAxes, rotation=90,
                          va="center", ha="center", fontsize=11)
    fig.suptitle(f"{pde} — 3D volume: analytic vs {arch} vs |error|", fontsize=13)
    fig.tight_layout()
    path = os.path.join(viz.FIG_DIR, f"{pde}_3d_volume.png")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(path, os.path.join(os.path.dirname(__file__), "..", ".."))


def _save_rotating_gif(pde, model_step, x0, vmin, vmax, n_frames=16):
    """Rotating + evolving GIF: advance the sim and rotate azimuth each frame."""
    import imageio.v2 as imageio

    frames = []
    x = x0
    fig = plt.figure(figsize=(4, 4))
    for f in range(n_frames):
        x = model_step(x)
        vol = np.asarray(x[0, :, :, :, 0])
        ax = fig.add_subplot(111, projection='3d')
        draw_volume(ax, vol, "inferno", vmin, vmax)
        ax.view_init(elev=22, azim=20 + 4 * f)
        ax.set_title(f"{pde} (3D) — frame {f}", fontsize=9)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame.copy())
        fig.clf()
    plt.close(fig)
    path = os.path.join(viz.GIF_DIR, f"{pde}_3d_volume.gif")
    imageio.mimsave(path, frames, fps=8)
    return os.path.relpath(path, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    render_volume(args.pde, grid=args.grid, epochs=args.epochs)


if __name__ == "__main__":
    main()
