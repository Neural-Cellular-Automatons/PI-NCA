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

# Winner direction.
HIGHER_BETTER = {"psnr", "ssim", "throughput_cells_per_s"}
NEUTRAL = {"grad_energy"}  # not clearly better high or low → not bolded
LOWER_BETTER = {"mse", "rel_l2", "rmse", "mae", "max_abs_err", "highfreq_err_frac",
                "rel_l2_t_q1", "rel_l2_t_half", "rel_l2_t_q3", "rel_l2_t_final",
                "error_growth_ratio", "conservation_err", "bc_residual",
                "final_train_loss", "train_wall_s", "infer_s_per_step", "params"}


def run_pde(pde: str, cfg: EmuConfig, seeds, arch_names=None):
    channels = pdes.REGISTRY[pde].channels
    archs = registry.applicable(channels)
    if arch_names:
        archs = {k: v for k, v in archs.items() if k in arch_names}
    out = {}
    for name, spec in archs.items():
        ctor = spec.make(channels)
        _, agg = run_multiseed(ctor, EmuConfig(**{**cfg.__dict__, "pde": pde}), seeds=seeds)
        out[name] = {k: {"mean": a.mean, "std": a.std, "n": a.n} for k, a in agg.items()}
        print(f"  {name:10s} relL2={agg['rel_l2']}  consErr={agg['conservation_err']}  "
              f"params={int(agg['params'].mean)}")
    return out


# Ordered, grouped metric rows for the detailed transposed table.
METRIC_ROWS = [
    ("rel_l2", "rel-L2 ↓"), ("mse", "MSE ↓"), ("rmse", "RMSE ↓"), ("mae", "MAE ↓"),
    ("max_abs_err", "L∞ ↓"), ("psnr", "PSNR(dB) ↑"), ("ssim", "SSIM ↑"),
    ("highfreq_err_frac", "hi-freq err frac ↓"),
    ("rel_l2_t_q1", "rel-L2 @T/4 ↓"), ("rel_l2_t_half", "rel-L2 @T/2 ↓"),
    ("rel_l2_t_q3", "rel-L2 @3T/4 ↓"), ("rel_l2_t_final", "rel-L2 @T ↓"),
    ("error_growth_ratio", "err-growth T/(T/4) ↓"),
    ("conservation_err", "mass-cons err ↓"), ("bc_residual", "periodic-BC res ↓"),
    ("grad_energy", "grad-energy"),
    ("params", "params ↓"), ("train_wall_s", "train wall(s) ↓"),
    ("infer_s_per_step", "infer s/step ↓"), ("throughput_cells_per_s", "throughput cells/s ↑"),
]


def _fmt(key, mean, std):
    if key == "params":
        return f"{int(mean)}"
    if key in ("psnr", "ssim", "error_growth_ratio", "train_wall_s"):
        return f"{mean:.3g}±{std:.2g}"
    if key == "throughput_cells_per_s":
        return f"{mean:.2e}"
    return f"{mean:.3e}±{std:.1e}"


def to_markdown(pde, results, cfg):
    archs = list(results)
    n = results[archs[0]]["rel_l2"]["n"] if archs else 0
    head = (f"### {pde}  (grid={cfg.grid_size}, train_steps={cfg.rollout_steps}, "
            f"eval_steps={cfg.eval_steps}, epochs={cfg.epochs}, seeds={n}, "
            f"clip={cfg.output_clip})")
    lines = [head, "", "| metric | " + " | ".join(archs) + " |",
             "|" + "---|" * (len(archs) + 1)]
    for key, label in METRIC_ROWS:
        present = [m for m in archs if key in results[m]]
        if not present:
            continue
        vals = {m: results[m][key]["mean"] for m in present}
        if key in NEUTRAL:
            winner = None
        else:
            winner = (max if key in HIGHER_BETTER else min)(vals, key=vals.get)
        cells = []
        for m in archs:
            if key not in results[m]:
                cells.append("—"); continue
            s = _fmt(key, results[m][key]["mean"], results[m][key]["std"])
            cells.append(f"**{s}**" if m == winner else s)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--seeds", type=int, default=1)  # 1 = single fixed seed 42 (matches originals)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--archs", default=None, help="comma-separated arch names (default: all applicable)")
    ap.add_argument("--tag", default=None, help="output filename suffix")
    ap.add_argument("--clip", default=None, help="lo,hi to clip each step (bounded ablation)")
    args = ap.parse_args()

    clip = tuple(float(v) for v in args.clip.split(",")) if args.clip else None
    cfg = EmuConfig(pde=args.pde, grid_size=args.grid, rollout_steps=args.rollout,
                    eval_steps=args.eval, epochs=args.epochs, output_clip=clip,
                    warmup_epochs=30, preseed_steps=10)  # "better start" protocol
    seeds = (42,) if args.seeds <= 1 else tuple(range(args.seeds))
    arch_names = args.archs.split(",") if args.archs else None
    print(f"Benchmark {args.pde}: archs={arch_names or 'all'}, seeds={seeds}, clip={clip}")
    results = run_pde(args.pde, cfg, seeds, arch_names)

    suffix = f"_{args.tag}" if args.tag else ""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"bench_{args.pde}{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "seeds": list(seeds), "results": results}, f, indent=2)
    md = to_markdown(args.pde, results, cfg)
    md_path = os.path.join(RESULTS_DIR, f"bench_{args.pde}{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    try:
        print("\n" + md)
    except UnicodeEncodeError:
        print(f"\n[detailed table written to {md_path}]")


if __name__ == "__main__":
    main()
