"""Are the baselines under-trained? A learning-rate and budget sweep for the big models.

Every number in the benchmark comes from one shared recipe: 1{,}200 epochs of AdamW at
`lr=1e-3` with warmup, identical for a 4{,}576-parameter cellular automaton and a
592{,}897-parameter neural operator. Sharing the recipe is what makes the comparison
controlled, but it also makes one result suspicious: on Cahn--Hilliard the FNO scores
worse than an unconstrained 6{,}784-parameter NCA. A reviewer is right to ask whether
that is a property of the architecture or of the budget it was given.

This sweeps the two knobs that would fix an under-trained baseline -- learning rate and
epoch count -- for the large models on the phenomena where the claim matters, and reports
the training-loss curve so convergence can be seen rather than asserted. The PI-NCA
models are swept on exactly the same grid, because a sweep given only to the baselines
would replace one unfair comparison with another.

What the paper takes from here is the *best* setting per model, and whether the headline
ordering survives it.

    python -m pinca_jax.tuning --pde cahn_hilliard --allow-cpu
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import bench, env, metrics, stats
from .equations import pdes
from .harness import EmuConfig, evaluate_emulator, field_bounds, train_emulator
from .models import registry

RES = bench.RESULTS_DIR
LRS = (3e-4, 1e-3, 3e-3)
DEFAULT_ARCHS = ("fno", "resnet", "pi_nca", "bounded_cons_nca")


def _curve(losses, n=24):
    """Subsample a loss curve to `n` points, keeping the first and last."""
    if len(losses) <= n:
        return [float(v) for v in losses]
    idx = np.unique(np.linspace(0, len(losses) - 1, n).astype(int))
    return [float(losses[i]) for i in idx]


def one(pde, arch, grid, epochs, lr, batch, rollout, eval_steps, n_eval, seed):
    C = pdes.REGISTRY[pde].channels
    bounds = field_bounds(pde, grid)
    cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=rollout, eval_steps=eval_steps,
                    epochs=epochs, batch=batch, n_eval=n_eval, seed=seed, lr=lr,
                    warmup_epochs=min(30, epochs // 4),
                    preseed_steps=0 if pde == "cahn_hilliard" else 10)

    def attempt(b):
        c = EmuConfig(**{**cfg.__dict__, "batch": b})
        tr = train_emulator(registry.REGISTRY[arch].make(C, bounds=bounds), c)
        return tr, evaluate_emulator(tr["model"], tr["params"], c)

    (tr, ev), used = bench.run_with_oom_backoff(attempt, batch, min_batch=2,
                                                label=f"{pde}/{arch}")
    losses = tr["losses"]
    tail = losses[-max(1, len(losses) // 10):]
    head = losses[len(losses) // 2: len(losses) // 2 + max(1, len(losses) // 10)]
    return {
        "rel_l2": ev["rel_l2"],
        "per_ic_rel_l2": list(ev["per_ic_rel_l2"]),
        "final_train_loss": float(np.mean(tail)),
        # Convergence evidence: how much the loss still fell over the second half. A
        # model that is still improving fast was under-trained, whatever its error.
        "second_half_improvement": float(np.mean(head) / (np.mean(tail) + 1e-30)),
        "loss_curve": _curve(losses),
        "params": int(metrics.param_count(tr["params"])),
        "train_wall_s": float(tr["wall_s"]),
        "batch_used": int(used),
        "epochs": int(epochs), "lr": float(lr),
    }


def study(pde, archs, grid=48, base_epochs=1200, mult=4, batch=32, rollout=12,
          eval_steps=48, n_eval=8, seed=0, lrs=LRS):
    """Every (arch, lr) at the base budget, plus the best lr at `mult` times the budget."""
    out = {}
    for arch in archs:
        rows = {}
        for lr in lrs:
            tag = f"lr{lr:g}_e{base_epochs}"
            rows[tag] = one(pde, arch, grid, base_epochs, lr, batch, rollout,
                            eval_steps, n_eval, seed)
            r = rows[tag]
            print(f"  {arch:18s} {tag:18s} rel-L2 {r['rel_l2']:.4e}  "
                  f"train-loss {r['final_train_loss']:.3e}  "
                  f"still-improving {r['second_half_improvement']:.2f}x  "
                  f"{r['train_wall_s']:.0f}s")
        best_lr = min(rows, key=lambda k: rows[k]["rel_l2"])
        lr = rows[best_lr]["lr"]
        long_tag = f"lr{lr:g}_e{base_epochs * mult}"
        rows[long_tag] = one(pde, arch, grid, base_epochs * mult, lr, batch, rollout,
                             eval_steps, n_eval, seed)
        r = rows[long_tag]
        print(f"  {arch:18s} {long_tag:18s} rel-L2 {r['rel_l2']:.4e}  "
              f"train-loss {r['final_train_loss']:.3e}  "
              f"still-improving {r['second_half_improvement']:.2f}x  "
              f"{r['train_wall_s']:.0f}s  <- {mult}x budget at the best lr")
        # The tag the benchmark's shared recipe corresponds to, so the table can say
        # what the sweep changed relative to the number the paper reports.
        shared = f"lr{1e-3:g}_e{base_epochs}"
        out[arch] = {"settings": rows,
                     "best_setting": min(rows, key=lambda k: rows[k]["rel_l2"]),
                     "shared_recipe": shared if shared in rows else None}
    return out


def to_markdown(pde, out, note=""):
    L = [f"### Learning-rate and budget sweep - {pde} {note}", "",
         "Every model in the benchmark is trained with one shared recipe "
         "(`lr=1e-3`, 1{,}200 epochs). This sweep asks whether any of them was simply "
         "under-served by it. `still-improving` is the mean training loss over the middle "
         "tenth of training divided by the mean over the last tenth: a value near 1 means "
         "the loss had flattened, a large value means it was still falling when training "
         "stopped.", "",
         "| model | setting | rel-L2 | final train loss | still-improving | wall (s) |",
         "|---|---|---|---|---|---|"]
    for arch, rec in out.items():
        for tag, r in rec["settings"].items():
            star = " **best**" if tag == rec["best_setting"] else ""
            shared = " (shared recipe)" if tag == rec["shared_recipe"] else ""
            L.append(f"| {arch}{shared} | `{tag}`{star} | {r['rel_l2']:.4e} | "
                     f"{r['final_train_loss']:.3e} | {r['second_half_improvement']:.2f}x | "
                     f"{r['train_wall_s']:.0f} |")
    L += ["", "**Best setting per model, and what it does to the ordering:**", "",
          "| model | shared-recipe rel-L2 | best-setting rel-L2 | change |", "|---|---|---|---|"]
    for arch, rec in out.items():
        s = rec["settings"].get(rec["shared_recipe"])
        b = rec["settings"][rec["best_setting"]]
        if not s:
            continue
        L.append(f"| {arch} | {s['rel_l2']:.4e} | {b['rel_l2']:.4e} | "
                 f"{(s['rel_l2'] - b['rel_l2']) / s['rel_l2']:+.1%} |")
    best = {a: r["settings"][r["best_setting"]] for a, r in out.items()}
    order = sorted(best, key=lambda a: best[a]["rel_l2"])
    L += ["", "Ranking at each model's own best setting: "
              + " < ".join(f"`{a}` {best[a]['rel_l2']:.4e}" for a in order) + ".", "",
          "Loss curves are stored in the JSON under `loss_curve` (subsampled to 24 points) "
          "so convergence can be plotted rather than taken on trust."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="cahn_hilliard")
    ap.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=1200, help="the shared-recipe budget")
    ap.add_argument("--mult", type=int, default=4, help="long-run multiple of that budget")
    ap.add_argument("--lrs", default=",".join(f"{v:g}" for v in LRS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("tuning", allow_cpu=args.allow_cpu)
    archs = [a for a in args.archs.split(",") if a in registry.REGISTRY]
    lrs = tuple(float(v) for v in args.lrs.split(","))
    print(f"[tuning] {args.pde}: {len(archs)} archs x {len(lrs)} learning rates, "
          f"plus {args.mult}x budget at the best; grid {args.grid}")
    out = study(args.pde, archs, grid=args.grid, base_epochs=args.epochs, mult=args.mult,
                batch=args.batch, rollout=args.rollout, eval_steps=args.eval,
                n_eval=args.n_eval, seed=args.seed, lrs=lrs)
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, f"tuning_{args.pde}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"pde": args.pde, "grid": args.grid, "lrs": list(lrs),
                   "base_epochs": args.epochs, "mult": args.mult, "seed": args.seed,
                   "results": out, "device": env.provenance("tuning")}, f, indent=1)
    md = to_markdown(args.pde, out, f"(grid={args.grid}, seed={args.seed})")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"[tuning] wrote {base}.json / .md")


if __name__ == "__main__":
    main()
