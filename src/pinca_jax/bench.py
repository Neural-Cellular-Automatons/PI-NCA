"""Benchmark runner — multi-seed emulator comparison across PDEs.

Produces mean±std tables (never single-run) and writes machine-readable JSON +
human-readable Markdown to results/. Reduced-scale by default (CPU); pass a
larger EmuConfig to reproduce at full scale on GPU.

CLI:  python -m pinca_jax.bench --pde heat --seeds 3 --epochs 200 --grid 24
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .harness import EmuConfig, run_multiseed
from .equations import pdes
from .models import registry

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# Metrics where lower is better (for marking winners).
LOWER_BETTER = {"mse", "rel_l2", "rmse", "conservation_err", "bc_residual",
                "final_train_loss", "train_wall_s", "infer_s_per_step"}


def run_pde(pde: str, cfg: EmuConfig, seeds):
    channels = pdes.REGISTRY[pde].channels
    archs = registry.applicable(channels)
    out = {}
    for name, spec in archs.items():
        ctor = spec.make(channels)
        _, agg = run_multiseed(ctor, EmuConfig(**{**cfg.__dict__, "pde": pde}), seeds=seeds)
        out[name] = {k: {"mean": a.mean, "std": a.std, "n": a.n} for k, a in agg.items()}
        print(f"  {name:10s} relL2={agg['rel_l2']}  consErr={agg['conservation_err']}  "
              f"params={int(agg['params'].mean)}")
    return out


def to_markdown(pde, results, cfg):
    cols = ["rel_l2", "mse", "psnr", "conservation_err", "bc_residual",
            "grad_energy", "params", "train_wall_s", "infer_s_per_step"]
    lines = [f"### {pde}  (grid={cfg.grid_size}, train_steps={cfg.rollout_steps}, "
             f"eval_steps={cfg.eval_steps}, epochs={cfg.epochs}, seeds={results and list(results.values())[0]['rel_l2']['n']})",
             "", "| arch | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    # find winners per column
    best = {}
    for c in cols:
        vals = {m: results[m][c]["mean"] for m in results if c in results[m]}
        if not vals:
            continue
        best[c] = (min if c in LOWER_BETTER else max)(vals, key=vals.get)
    for m, md in results.items():
        cells = []
        for c in cols:
            if c not in md:
                cells.append("—"); continue
            mean, std = md[c]["mean"], md[c]["std"]
            s = f"{mean:.3e}±{std:.1e}" if abs(mean) < 100 or c in ("params",) else f"{mean:.2f}±{std:.1f}"
            if c == "params":
                s = f"{int(mean)}"
            if best.get(c) == m:
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--eval", type=int, default=48)
    args = ap.parse_args()

    cfg = EmuConfig(pde=args.pde, grid_size=args.grid, rollout_steps=args.rollout,
                    eval_steps=args.eval, epochs=args.epochs)
    seeds = tuple(range(args.seeds))
    print(f"Benchmark {args.pde}: archs vs teacher, seeds={seeds}")
    results = run_pde(args.pde, cfg, seeds)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"bench_{args.pde}.json"), "w") as f:
        json.dump({"config": asdict(cfg), "seeds": list(seeds), "results": results}, f, indent=2)
    md = to_markdown(args.pde, results, cfg)
    with open(os.path.join(RESULTS_DIR, f"bench_{args.pde}.md"), "w") as f:
        f.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
