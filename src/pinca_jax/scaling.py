"""Scaling curves: does the ranking survive a change of scale?

Every accuracy table in this study is measured at one grid, one training horizon and one
epoch budget. A ranking read off a single operating point is only useful if it is stable
under scale, and there is no reason to assume that a priori: a global spectral operator
has a fixed number of retained modes and should improve differently with resolution than a
local rule whose receptive field is fixed in cells, and a model with two orders of
magnitude more parameters should benefit differently from a larger epoch budget.

Three sweeps, each varying exactly one axis with everything else held fixed:

* **Grid.** Resolution changes what fraction of the solution's spectrum is representable
  and, for a local model with a fixed per-step receptive field, how many steps information
  needs to cross the domain. A local rule that wins at 16 cells and loses at 64 is not a
  win, it is a statement about the domain size.
* **Rollout horizon.** Training with backpropagation through H steps and evaluating over
  K >> H is the setting where autoregressive surrogates fail. Sweeping H separates "this
  architecture is accurate" from "this architecture was trained with enough of the horizon
  to be stable over it".
* **Training budget.** Epochs, at fixed everything else. This is the axis on which an
  apparent architecture win is most often really a convergence difference; a ranking that
  flips as the budget grows was a statement about optimisation, not about the prior.

The output the paper needs is not the curves themselves but whether the *ordering* is
stable along each axis, so `rank_stability` reports Kendall's tau between the ranking at
the smallest and largest setting on each axis, and the table flags any axis where the
ordering changes.

    python -m pinca_jax.scaling --pde heat --allow-cpu
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

from . import bench, env, metrics, stats
from .equations import pdes
from .harness import EmuConfig, field_bounds, run_multiseed
from .models import registry

RES = bench.RESULTS_DIR

DEFAULT_ARCHS = ["plain_nca", "pi_nca", "multiscale_flux_nca", "fno", "resnet",
                 "resnet_iso", "identity"]


def _cfg(pde, grid, rollout, epochs, eval_steps, batch, n_eval, seed=42):
    return EmuConfig(pde=pde, grid_size=grid, rollout_steps=rollout,
                     eval_steps=eval_steps, epochs=epochs, batch=batch, n_eval=n_eval,
                     seed=seed, warmup_epochs=min(30, epochs // 4),
                     preseed_steps=0 if pde == "cahn_hilliard" else 10)


def sweep(pde, archs, axis, values, base, seeds=(42,)):
    """Vary one axis; return {value: {arch: summary}} plus the per-IC samples."""
    C = pdes.REGISTRY[pde].channels
    out = {}
    for v in values:
        kw = dict(base)
        kw[axis] = v
        # The evaluation horizon is held FIXED on every axis, the rollout axis included.
        # Letting it grow with the training horizon would change the task and the training
        # budget at the same time, and the question this axis asks is precisely how much
        # training horizon is needed to stay accurate over a FIXED evaluation horizon.
        if axis == "rollout" and v > kw["eval_steps"]:
            raise ValueError(f"training horizon {v} exceeds the fixed evaluation horizon "
                             f"{kw['eval_steps']}; raise --eval instead of letting the "
                             f"sweep move it")
        bounds = field_bounds(pde, kw["grid"])
        row = {}
        for a in archs:
            cfg = _cfg(pde, kw["grid"], kw["rollout"], kw["epochs"], kw["eval_steps"],
                       kw["batch"], kw["n_eval"])
            try:
                runs, _ = run_multiseed(registry.REGISTRY[a].make(C, bounds=bounds),
                                        cfg, seeds=seeds)
                per_ic = metrics.pool_per_ic(runs, "per_ic_rel_l2")
                s = stats.summarize(per_ic)
                row[a] = {"mean": s.mean, "ci_lo": s.ci_lo, "ci_hi": s.ci_hi,
                          "median": s.median, "n": s.n, "per_ic_rel_l2": per_ic,
                          "params": runs[0]["params"],
                          "train_wall_s": runs[0].get("train_wall_s")}
            except Exception as exc:                  # noqa: BLE001 - recorded, not fatal
                row[a] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        out[v] = row
        ok = {a: r for a, r in row.items() if "error" not in r}
        best = min(ok, key=lambda a: ok[a]["mean"]) if ok else "-"
        print(f"  {axis}={v:<6} best {best:22s} "
              + "  ".join(f"{a}:{r['mean']:.3g}" for a, r in list(ok.items())[:4]))
    return out


def kendall_tau(a_order, b_order):
    """Rank correlation between two orderings of the same architectures.

    +1 = identical ordering, -1 = reversed, 0 = unrelated. Implemented directly rather
    than pulled from scipy because the input is a handful of names and the pair count is
    trivial.
    """
    common = [x for x in a_order if x in b_order]
    if len(common) < 2:
        return float("nan")
    ra = {x: i for i, x in enumerate(a_order)}
    rb = {x: i for i, x in enumerate(b_order)}
    conc = disc = 0
    for x, y in itertools.combinations(common, 2):
        s = (ra[x] - ra[y]) * (rb[x] - rb[y])
        if s > 0:
            conc += 1
        elif s < 0:
            disc += 1
    total = conc + disc
    return float((conc - disc) / total) if total else float("nan")


def rank_stability(axis_result):
    """Does the ordering at the smallest setting match the ordering at the largest?"""
    ks = sorted(axis_result)
    if len(ks) < 2:
        return {"tau": float("nan"), "stable": None}

    def order(v):
        ok = {a: r for a, r in axis_result[v].items() if "error" not in r}
        return sorted(ok, key=lambda a: ok[a]["mean"])

    lo, hi = order(ks[0]), order(ks[-1])
    tau = kendall_tau(lo, hi)
    return {"tau": tau, "from": ks[0], "to": ks[-1], "order_small": lo, "order_large": hi,
            "winner_changed": bool(lo and hi and lo[0] != hi[0]),
            "stable": bool(tau == tau and tau >= 0.8)}


def to_markdown(pde, results, base, seeds):
    L = [f"### Scaling behaviour - {pde}", "",
         f"Each sweep varies one axis with everything else fixed at "
         f"{ {k: v for k, v in base.items()} }, seeds {list(seeds)}. The question is not "
         f"what the errors are, it is whether the *ordering* survives the change of "
         f"scale: a ranking that flips along an axis was a statement about that operating "
         f"point, not about the architectures.", ""]
    for axis, block in results.items():
        st = block["stability"]
        L += [f"#### {axis}", "",
              "| " + axis + " | " + " | ".join(
                  a for a in next(iter(block["values"].values()))) + " |",
              "|" + "---|" * (len(next(iter(block["values"].values()))) + 1)]
        for v, row in block["values"].items():
            cells = [("FAILED" if "error" in r else f"{r['mean']:.3g}")
                     for r in row.values()]
            L.append(f"| {v} | " + " | ".join(cells) + " |")
        verdict = ("ordering is stable" if st.get("stable")
                   else "**ordering changes with scale**")
        L += ["",
              f"Kendall's tau between the ranking at {axis}={st.get('from')} and "
              f"{axis}={st.get('to')}: **{st.get('tau', float('nan')):.2f}** -- {verdict}."
              + (f" The best architecture changes from `{st['order_small'][0]}` to "
                 f"`{st['order_large'][0]}`." if st.get("winner_changed") else ""), ""]
    unstable = [a for a, b in results.items() if not b["stability"].get("stable")]
    if unstable:
        L += ["> **Any headline ranking should be read as conditional on the operating "
              "point**, because the ordering is not stable along: "
              + ", ".join(f"`{a}`" for a in unstable) + ". Reporting a winner without the "
              "scale it was measured at would be misleading."]
    else:
        L += ["> The ordering is stable along every axis swept here, which is what "
              "licenses quoting a single ranking in the paper."]
    return "\n".join(L) + "\n"


def run(pde, archs, grids, rollouts, epoch_budgets, base, seeds=(42,)):
    results = {}
    for axis, values in (("grid", grids), ("rollout", rollouts),
                         ("epochs", epoch_budgets)):
        if len(values) < 2:
            continue
        print(f"[scaling] {pde}: sweeping {axis} over {list(values)}")
        vals = sweep(pde, archs, axis, values, base, seeds=seeds)
        results[axis] = {"values": vals, "stability": rank_stability(vals)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    ap.add_argument("--grids", default="16,24,32")
    ap.add_argument("--rollouts", default="4,8,16")
    ap.add_argument("--epochs-sweep", default="150,400,900")
    ap.add_argument("--grid", type=int, default=24, help="fixed value on the other axes")
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("scaling", allow_cpu=args.allow_cpu)
    seeds = (42,) if args.seeds <= 1 else tuple(range(args.seeds))
    archs = [a for a in args.archs.split(",") if a in registry.REGISTRY]
    base = dict(grid=args.grid, rollout=args.rollout, epochs=args.epochs,
                eval_steps=args.eval, batch=args.batch, n_eval=args.n_eval)
    ints = lambda s: [int(x) for x in s.split(",") if x]   # noqa: E731
    results = run(args.pde, archs, ints(args.grids), ints(args.rollouts),
                  ints(args.epochs_sweep), base, seeds=seeds)
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, f"scaling_{args.pde}")
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump({"pde": args.pde, "base": base, "seeds": list(seeds),
                   "results": results, "device": env.provenance("scaling")}, f, indent=1)
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(args.pde, results, base, seeds))
    print(f"[scaling] wrote {out}.json / .md")
    for axis, b in results.items():
        st = b["stability"]
        print(f"  {axis:8s} tau={st['tau']:.2f}  "
              f"{'stable' if st['stable'] else 'ORDERING CHANGES'}")


if __name__ == "__main__":
    main()
