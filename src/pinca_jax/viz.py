"""Generate analytic-vs-model GIFs + error maps for each trained phenomenon.

For a PDE and architecture: train a model (reduced-scale), roll out the solver
(analytic) and the model from a held-out IC, and emit:
  - results/gifs/<pde>_analytic.gif, _model.gif, _error.gif      (local, gitignored)
  - docs/figures/<pde>_comparison.png  (3-row montage at key timesteps; committed)

Multi-channel states are visualised on channel 0 (vorticity for NS, h for SWE, u for
reaction systems). Run:  python -m pinca_jax.viz --pde heat --arch multiscale_flux_nca
"""
from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from .harness import EmuConfig, train_emulator, _emu_rollout, field_bounds
from .equations import pdes
from .models import registry
from . import ic

GIF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "gifs")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")

# sensible default architecture per phenomenon (from the findings)
DEFAULT_ARCH = {
    "heat": "multiscale_flux_nca", "cahn_hilliard": "bounded_cons_nca",
    "allen_cahn": "multiscale_flux_nca", "gray_scott": "mc_flux_nca",
    "shallow_water": "mc_flux_nca", "fitzhugh_nagumo": "plain_nca",
    "nagumo": "multiscale_flux_nca", "navier_stokes": "fno",
    "wave": "mc_flux_nca", "adv_diff": "multiscale_flux_nca",
}


def _frames(traj_ch):  # (T,H,W) → list of (H,W)
    return [np.asarray(traj_ch[i]) for i in range(traj_ch.shape[0])]


def render(pde, arch=None, grid=32, epochs=150, eval_steps=64, every=4, seed=0,
           clip=None, npz=None):
    """Draw the analytic/model/error montage + GIFs for one phenomenon.

    `npz` = a file written by `pinca_jax.capture`: the trajectories are loaded and
    NOTHING is retrained. That is the cheap path, and the one to use when you want
    to redraw or restyle figures later on a machine with no GPU.
    """
    if npz is not None:
        d = np.load(npz)
        solver_traj = jnp.asarray(d["solver"])
        model_traj = jnp.asarray(d["model"])
        pde = str(d["pde"]) if "pde" in d else pde
        arch = str(d["arch"]) if "arch" in d else (arch or "?")
        eval_steps = solver_traj.shape[0] - 1
        print(f"[viz] {pde} / {arch}: from {npz} ({solver_traj.shape}, no training)")
        err_traj = jnp.abs(solver_traj - model_traj)
    else:
        arch = arch or DEFAULT_ARCH.get(pde, "multiscale_flux_nca")
        spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
        C = pdes.REGISTRY[pde].channels
        # CH winner needs bounds; harness clip matches the solver range
        if pde in ("cahn_hilliard", "allen_cahn") and clip is None:
            clip = (-1.0, 1.0)
        cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=min(16, eval_steps),
                        eval_steps=eval_steps, epochs=epochs, seed=seed, output_clip=clip)
        ctor = registry.REGISTRY[arch].make(C, bounds=field_bounds(pde, grid))
        print(f"[viz] {pde} / {arch}: training ({epochs} ep, grid {grid})...")
        tr = train_emulator(ctor, cfg)
        model, params = tr["model"], tr["params"]

        key = jax.random.PRNGKey(seed + 777)
        x0 = ic.make_state(key, pde, 1, grid)

        # capture trajectories (channel 0)
        def traj(rollout_fn):
            outs = [x0]
            x = x0
            for _ in range(eval_steps):
                x = rollout_fn(x)
                outs.append(x)
            return jnp.concatenate([o[0:1, ..., 0] for o in outs], axis=0)  # (T+1,H,W)

        solver_traj = traj(lambda x: spec.step(x, spec.params))
        def model_step(x):
            y = model.apply(params, x)
            return jnp.clip(y, clip[0], clip[1]) if clip else y
        model_traj = traj(model_step)
        err_traj = jnp.abs(solver_traj - model_traj)

    vmin = float(jnp.min(solver_traj)); vmax = float(jnp.max(solver_traj))
    emax = float(jnp.quantile(err_traj, 0.99)) + 1e-6

    os.makedirs(GIF_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    # Montage (committed artifact) first — matplotlib only, never blocked by imageio.
    montage = _save_montage(pde, arch, solver_traj, model_traj, err_traj,
                            vmin, vmax, emax, eval_steps)
    try:
        _save_gifs(pde, solver_traj, model_traj, err_traj, vmin, vmax, emax, every)
    except Exception as e:
        print(f"[viz] {pde}: gif step skipped ({repr(e)[:80]})")
    final_relerr = float(jnp.linalg.norm(model_traj[-1] - solver_traj[-1]) /
                         (jnp.linalg.norm(solver_traj[-1]) + 1e-8))
    print(f"[viz] {pde}: montage -> {montage} | final-frame rel-err {final_relerr:.3e}")
    return {"pde": pde, "arch": arch, "montage": montage, "final_rel_err": final_relerr}


def _save_gifs(pde, solver, model, err, vmin, vmax, emax, every):
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio

    def colorize(a, lo, hi, cmap):
        x = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        return (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)

    idx = range(0, solver.shape[0], every)
    imageio.mimsave(os.path.join(GIF_DIR, f"{pde}_analytic.gif"),
                    [colorize(np.asarray(solver[i]), vmin, vmax, "inferno") for i in idx], fps=8)
    imageio.mimsave(os.path.join(GIF_DIR, f"{pde}_model.gif"),
                    [colorize(np.asarray(model[i]), vmin, vmax, "inferno") for i in idx], fps=8)
    imageio.mimsave(os.path.join(GIF_DIR, f"{pde}_error.gif"),
                    [colorize(np.asarray(err[i]), 0, emax, "magma") for i in idx], fps=8)


def _save_montage(pde, arch, solver, model, err, vmin, vmax, emax, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [0, T // 4, T // 2, 3 * T // 4, T]
    fig, ax = plt.subplots(3, len(cols), figsize=(2.2 * len(cols), 6.6))
    rows = [("analytic", solver, "inferno", vmin, vmax),
            ("model", model, "inferno", vmin, vmax),
            ("|error|", err, "magma", 0, emax)]
    for r, (label, data, cmap, lo, hi) in enumerate(rows):
        for c, t in enumerate(cols):
            a = ax[r, c]
            a.imshow(np.asarray(data[t]), cmap=cmap, vmin=lo, vmax=hi)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"t={t}", fontsize=9)
            if c == 0:
                a.set_ylabel(label, fontsize=10)
    fig.suptitle(f"{pde} — analytic vs {arch} vs error", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"{pde}_comparison.png")
    fig.savefig(path, dpi=85, bbox_inches="tight"); plt.close(fig)
    return os.path.relpath(path, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval", type=int, default=64)
    ap.add_argument("--npz", default=None,
                    help="render from a pinca_jax.capture trajectory file (no training)")
    args = ap.parse_args()
    render(args.pde, args.arch, grid=args.grid, epochs=args.epochs,
           eval_steps=args.eval, npz=args.npz)


if __name__ == "__main__":
    main()
