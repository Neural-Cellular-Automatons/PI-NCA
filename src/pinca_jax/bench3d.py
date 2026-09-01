"""Uniform 3-D benchmark matrix: EVERY 3-D architecture on EVERY 3-D phenomenon.

The 3-D counterpart of bench_all.py, with the same three properties: a rectangular
matrix (no per-phenomenon architecture lists), resumable per cell, and tolerant of
out-of-memory on a single cell. 3-D is where OOM actually bites -- a 32^3 grid has
8x the cells of 16^3, and the BPTT tape scales with it.

Two 2-D hybrids have no 3-D counterpart and are absent by design, not omission:
SpectralFluxNCA (no 3-D two-stream model is implemented) and the separate
BoundedConsFluxNCA class (its behaviour is available as FluxNCA3D with bounds set).

Run:  python -m pinca_jax.bench3d
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import jax
import jax.numpy as jnp

from . import bench, env
from .harness3d import Emu3DConfig, train, evaluate
from .equations import pdes3d
from . import ic3d
import pinca_jax.models3d as M

RES = bench.RESULTS_DIR

# name -> (out_channels, bounds) -> (() -> module). Same list for every phenomenon.
ARCHS = {
    "plain_nca": lambda C, b: (lambda: M.NCA3D(out_channels=C)),
    "pi_nca": lambda C, b: (lambda: M.FluxNCA3D(out_channels=C)),
    "bounded_cons_nca": lambda C, b: (lambda: M.FluxNCA3D(out_channels=C, bounds=b)),
    "mc_flux_nca": lambda C, b: (lambda: M.MultiChannelFluxNCA3D(out_channels=C)),
    "multiscale_flux_nca": lambda C, b: (lambda: M.MultiScaleFluxNCA3D(out_channels=C)),
    "bounded_multiscale_nca": lambda C, b: (
        lambda: M.MultiScaleFluxNCA3D(out_channels=C, bounds=b)),
    "fno": lambda C, b: (lambda: M.FNO3D(out_channels=C)),
}

PDES = ["heat", "adv_diff", "allen_cahn", "nagumo", "gray_scott", "fitzhugh_nagumo"]


def field_bounds3d(pde, grid, seed=0, steps=24, margin=0.05):
    """The 3-D teacher's measured physical range, for the bounded variants."""
    spec = pdes3d.REGISTRY[pde]
    x0 = ic3d.make_state(jax.random.PRNGKey(seed), pde, 4, grid)
    x, lo, hi = x0, float(jnp.min(x0)), float(jnp.max(x0))
    for _ in range(steps):
        x = spec.step(x, spec.params)
        lo, hi = min(lo, float(jnp.min(x))), max(hi, float(jnp.max(x)))
    if not (jnp.isfinite(jnp.array([lo, hi])).all()):
        return None
    pad = margin * (hi - lo) + 1e-6
    return (lo - pad, hi + pad)


def _write(pde, results, cfg, grid):
    base = os.path.join(RES, f"bench3d_{pde}")
    bench.save_results(base + ".json",
                       {"config": asdict(cfg), "results": results,
                        "device": env.provenance("bench3d")})
    ok = {a: r for a, r in results.items() if "error" not in r}
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(bench.to_markdown(f"{pde} (3D, {grid}^3)", ok, cfg))
    return base


def run_cell(arch, cfg, C, bounds):
    maker = ARCHS[arch]

    def attempt(b):
        c = Emu3DConfig(**{**cfg.__dict__, "batch": b})
        tr = train(maker(C, bounds), c)
        ev = evaluate(tr["model"], tr["params"], c)
        ev["train_wall_s"] = tr["wall_s"]
        return ev

    with bench.CellTimer() as t:
        ev, used = bench.run_with_oom_backoff(attempt, cfg.batch, min_batch=1,
                                              label=f"{cfg.pde}/{arch}")
    rec = {k: {"mean": float(v), "std": 0.0, "n": 1} for k, v in ev.items()}
    rec["_batch_used"] = {"mean": float(used), "std": 0.0, "n": 1}
    rec["_cell_wall_s"] = {"mean": float(t.seconds), "std": 0.0, "n": 1}
    return rec, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdes", default=",".join(PDES))
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8, help="training batch (raise on GPU)")
    ap.add_argument("--archs", default=None, help="comma-separated subset")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    env.require_gpu("bench3d", allow_cpu=args.allow_cpu)
    os.makedirs(RES, exist_ok=True)
    wanted = args.archs.split(",") if args.archs else list(ARCHS)
    failures = []

    for pde in args.pdes.split(","):
        C = pdes3d.REGISTRY[pde].channels
        cfg = Emu3DConfig(pde=pde, grid_size=args.grid, epochs=args.epochs,
                          batch=args.batch)
        bounds = field_bounds3d(pde, args.grid)
        path = os.path.join(RES, f"bench3d_{pde}.json")
        results = {} if args.force else bench.load_results(path)
        todo = [a for a in wanted if a not in results or "error" in results.get(a, {})]
        print(f"[bench3d] {pde} (grid {args.grid}^3, C={C}) — {len(todo)} to run, "
              f"{len(wanted) - len(todo)} already done")

        for arch in todo:
            try:
                rec, used = run_cell(arch, cfg, C, bounds)
                results[arch] = rec
                print(f"  {arch:24s} rel-L2 {rec['rel_l2']['mean']:.4e} | "
                      f"psnr {rec['psnr']['mean']:.2f} | "
                      f"consErr {rec['conservation_err']['mean']:.2e} | "
                      f"params {int(rec['params']['mean'])} | batch {used}")
            except Exception as exc:                       # noqa: BLE001 - recorded
                kind = "oom" if bench.is_oom(exc) else "error"
                results[arch] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                 "kind": kind}
                print(f"  {arch:24s} FAILED ({kind}): {type(exc).__name__}")
                failures.append(f"{pde}/{arch} ({kind})")
            _write(pde, results, cfg, args.grid)           # checkpoint after every cell
            bench.free_device_memory()
        print(f"  wrote results/bench3d_{pde}.md")

    print(f"[bench3d] done. peak device mem {env.peak_mem_mb():.0f} MB")
    for f in failures:
        print(f"  FAILED {f}")


if __name__ == "__main__":
    main()
