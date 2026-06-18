"""Comprehensive detailed benchmark driver — every phenomenon + ablations A4/A5.

Writes a 20-metric detailed table (results/bench_<pde>_full.{md,json}) per phenomenon
and per ablation, incrementally. Run:  python -m pinca_jax.bench_all --group all
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .harness import EmuConfig, run_multiseed
from .equations import pdes
from .models import registry
from . import bench

RES = bench.RESULTS_DIR

# Per-phenomenon architecture sets (channel-appropriate).
PHENOMENA = {
    # scalar, local/diffusive
    "heat": ["plain_nca", "pi_nca", "multiscale_flux_nca", "spectral_flux_nca", "fno"],
    "allen_cahn": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    "nagumo": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    "adv_diff": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    # globally coupled
    "navier_stokes": ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno"],
    # stiff bounded
    "cahn_hilliard": ["plain_nca", "fno", "bounded_cons_nca", "bounded_multiscale_nca"],
    # multi-channel
    "wave": ["plain_nca", "mc_flux_nca", "fno"],
    "gray_scott": ["plain_nca", "mc_flux_nca", "fno"],
    "shallow_water": ["plain_nca", "mc_flux_nca", "fno"],
    "fitzhugh_nagumo": ["plain_nca", "mc_flux_nca", "fno"],
}

GROUPS = {
    "local": ["heat", "allen_cahn", "nagumo", "adv_diff"],
    "multichannel": ["wave", "gray_scott", "shallow_water", "fitzhugh_nagumo"],
    "special": ["cahn_hilliard", "navier_stokes"],
}


def _write(tag, pde, results, cfg, seeds):
    with open(os.path.join(RES, f"bench_{pde}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "seeds": list(seeds), "results": results}, f, indent=2)
    with open(os.path.join(RES, f"bench_{pde}_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(bench.to_markdown(pde, results, cfg))
    print(f"  wrote results/bench_{pde}_{tag}.md")


# "Start from a better point" protocol (single fixed seed + warmup + pre-seeded
# developed states), ported from the original implementations. preseed_steps is the
# number of solver steps used to pre-evolve training ICs into developed patterns.
WARMUP = 30
PRESEED = 10


def run_phenomena(pdes_list, seeds, epochs, grid):
    os.makedirs(RES, exist_ok=True)
    for pde in pdes_list:
        archs = PHENOMENA[pde]
        eval_steps = 48
        # Cahn-Hilliard coarsens rapidly from fresh ICs; pre-seeding on developed states
        # mismatches eval and regresses the bounded models to the floor → preseed off.
        preseed = 0 if pde == "cahn_hilliard" else PRESEED
        cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=12, eval_steps=eval_steps,
                        epochs=epochs, warmup_epochs=WARMUP, preseed_steps=preseed)
        print(f"[bench_all] {pde}: {archs}")
        results = {}
        for a in archs:
            C = pdes.REGISTRY[pde].channels
            ctor = registry.REGISTRY[a].make(C)
            _, agg = run_multiseed(ctor, EmuConfig(**{**cfg.__dict__, "pde": pde}), seeds=seeds)
            results[a] = {k: {"mean": v.mean, "std": v.std, "n": v.n} for k, v in agg.items()}
            print(f"    {a:22s} rel-L2 {agg['rel_l2']}  psnr {agg['psnr']}")
        _write("full", pde, results, cfg, seeds)


def run_ablations(seeds, epochs, grid):
    os.makedirs(RES, exist_ok=True)
    # A4: conservation on/off at matched width — conservative heat vs non-conservative nagumo
    for pde in ["heat", "nagumo"]:
        cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=12, eval_steps=48, epochs=epochs)
        results = {}
        for a in ["abl_flux", "abl_residual"]:
            ctor = registry.REGISTRY[a].make(1)
            _, agg = run_multiseed(ctor, cfg, seeds=seeds)
            results[a] = {k: {"mean": v.mean, "std": v.std, "n": v.n} for k, v in agg.items()}
            print(f"  A4 {pde} {a:14s} rel-L2 {agg['rel_l2']}")
        _write("A4", pde, results, cfg, seeds)
    # A5: perception size — heat (local) and navier_stokes (global)
    for pde in ["heat", "navier_stokes"]:
        cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=12, eval_steps=48, epochs=epochs)
        results = {}
        for a in ["abl_k3", "abl_k5", "abl_multiscale"]:
            ctor = registry.REGISTRY[a].make(1)
            _, agg = run_multiseed(ctor, cfg, seeds=seeds)
            results[a] = {k: {"mean": v.mean, "std": v.std, "n": v.n} for k, v in agg.items()}
            print(f"  A5 {pde} {a:14s} rel-L2 {agg['rel_l2']}  params {int(agg['params'].mean)}")
        _write("A5", pde, results, cfg, seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    choices=["all", "local", "multichannel", "special", "ablation"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="1 = single fixed seed 42 (default, matches originals); >1 = variance study")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--grid", type=int, default=24)
    args = ap.parse_args()
    seeds = (42,) if args.seeds <= 1 else tuple(range(args.seeds))
    if args.group in ("all", "local"):
        run_phenomena(GROUPS["local"], seeds, args.epochs, args.grid)
    if args.group in ("all", "multichannel"):
        run_phenomena(GROUPS["multichannel"], seeds, args.epochs, args.grid)
    if args.group in ("all", "special"):
        run_phenomena(GROUPS["special"], seeds, args.epochs, args.grid)
    if args.group in ("all", "ablation"):
        run_ablations(seeds, args.epochs, args.grid)
    print("[bench_all] done.")


if __name__ == "__main__":
    main()
