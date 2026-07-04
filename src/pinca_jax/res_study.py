"""Multi-resolution study: train at grid G_train, evaluate at grid G_eval.

Conv-NCA and spectral FNO are size-agnostic, so the same trained params apply at
any resolution — this measures both (a) accuracy vs resolution and (b) zero-shot
cross-resolution transfer (train coarse, eval fine = super-resolution, and vice
versa). Reuses the emulator harness unchanged.

Run:  python -m pinca_jax.res_study --pdes heat,navier_stokes
"""
from __future__ import annotations

import argparse
import json
import os

from .harness import EmuConfig, train_emulator, evaluate_emulator
from .equations import pdes
from .models import registry

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
GRIDS = (16, 24, 32, 48)

ARCHS = {  # per-PDE model set (channel-appropriate, regime-relevant)
    "heat": ["pi_nca", "multiscale_flux_nca", "fno"],
    "allen_cahn": ["multiscale_flux_nca", "fno"],
    "navier_stokes": ["multiscale_flux_nca", "fno"],
}


def run_pde(pde, grids, epochs):
    C = pdes.REGISTRY[pde].channels
    preseed = 0 if pde == "cahn_hilliard" else 10
    out = {}
    for a in ARCHS[pde]:
        ctor = registry.REGISTRY[a].make(C)
        cell = {}
        for gt in grids:
            cfg = EmuConfig(pde=pde, grid_size=gt, rollout_steps=12, eval_steps=48,
                            epochs=epochs, warmup_epochs=30, preseed_steps=preseed)
            tr = train_emulator(ctor, cfg)
            for ge in grids:
                ce = EmuConfig(**{**cfg.__dict__, "grid_size": ge})
                ev = evaluate_emulator(tr["model"], tr["params"], ce)
                cell[f"{gt}->{ge}"] = ev["rel_l2"]
            print(f"  {pde} {a:20s} train {gt:2d} | " +
                  " ".join(f"{ge}:{cell[f'{gt}->{ge}']:.3f}" for ge in grids))
        out[a] = cell
    return out


def to_md(pde, out, grids):
    lines = [f"### {pde} — resolution transfer (rel-L2; rows=train grid, cols=eval grid)"]
    for a, cell in out.items():
        lines += ["", f"**{a}**", "", "| train\\eval | " + " | ".join(f"{g}²" for g in grids) + " |",
                  "|" + "---|" * (len(grids) + 1)]
        for gt in grids:
            row = " | ".join(f"{cell[f'{gt}->{ge}']:.3e}" + ("**" if gt == ge else "")
                             for ge in grids)
            # mark the diagonal (same-res) bold
            cells = []
            for ge in grids:
                v = f"{cell[f'{gt}->{ge}']:.3e}"
                cells.append(f"**{v}**" if gt == ge else v)
            lines.append(f"| {gt}² | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdes", default="heat,navier_stokes")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    for pde in args.pdes.split(","):
        print(f"[res_study] {pde}")
        out = run_pde(pde, GRIDS, args.epochs)
        json.dump({"grids": GRIDS, "results": out},
                  open(os.path.join(RES, f"bench_resolution_{pde}.json"), "w", encoding="utf-8"), indent=2)
        open(os.path.join(RES, f"bench_resolution_{pde}.md"), "w", encoding="utf-8").write(to_md(pde, out, GRIDS))
        print(f"  wrote results/bench_resolution_{pde}.md")
    print("[res_study] done.")


if __name__ == "__main__":
    main()
