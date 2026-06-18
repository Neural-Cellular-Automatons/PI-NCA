"""3-D benchmark driver — detailed tables for the 3-D phenomena.

Reduced scale (16^3) for CPU feasibility; single fixed seed (42) + better-start
protocol. Writes results/bench3d_<pde>.{md,json}.  Run: python -m pinca_jax.bench3d
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from . import metrics, bench
from .harness3d import Emu3DConfig, train, evaluate
from .equations import pdes3d
import pinca_jax.models3d as M

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")

ARCHS = {  # name -> (ctor(C), scalar_only)
    "plain_nca": (lambda C: (lambda: M.NCA3D(out_channels=C)), False),
    "pi_nca": (lambda C: (lambda: M.FluxNCA3D()), True),
    "multiscale_flux_nca": (lambda C: (lambda: M.MultiScaleFluxNCA3D()), True),
    "mc_flux_nca": (lambda C: (lambda: M.MultiChannelFluxNCA3D(out_channels=C)), False),
    "fno": (lambda C: (lambda: M.FNO3D(out_channels=C)), False),
}

PHENOMENA = {
    "heat": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    "allen_cahn": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    "nagumo": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    "adv_diff": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    "gray_scott": ["plain_nca", "mc_flux_nca", "fno"],
    "fitzhugh_nagumo": ["plain_nca", "mc_flux_nca", "fno"],
}


def run_pde(pde, cfg: Emu3DConfig, arch_names):
    C = pdes3d.REGISTRY[pde].channels
    results = {}
    for a in arch_names:
        maker, scalar_only = ARCHS[a]
        if scalar_only and C != 1:
            continue
        ctor = maker(C)
        tr = train(ctor, cfg)
        ev = evaluate(tr["model"], tr["params"], cfg)
        ev["train_wall_s"] = tr["wall_s"]
        results[a] = {k: {"mean": float(v), "std": 0.0, "n": 1} for k, v in ev.items()}
        print(f"  {a:20s} rel-L2 {ev['rel_l2']:.4e} | psnr {ev['psnr']:.2f} | "
              f"consErr {ev['conservation_err']:.2e} | params {ev['params']}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdes", default=",".join(PHENOMENA))
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    for pde in args.pdes.split(","):
        cfg = Emu3DConfig(pde=pde, grid_size=args.grid, epochs=args.epochs)
        print(f"[bench3d] {pde} (grid {args.grid}^3)")
        results = run_pde(pde, cfg, PHENOMENA[pde])
        with open(os.path.join(RES, f"bench3d_{pde}.json"), "w", encoding="utf-8") as f:
            json.dump({"config": asdict(cfg), "results": results}, f, indent=2)
        # reuse the 2-D detailed-table renderer (skips absent metrics gracefully)
        with open(os.path.join(RES, f"bench3d_{pde}.md"), "w", encoding="utf-8") as f:
            f.write(bench.to_markdown(f"{pde} (3D, {args.grid}^3)", results, cfg))
        print(f"  wrote results/bench3d_{pde}.md")
    print("[bench3d] done.")


if __name__ == "__main__":
    main()
