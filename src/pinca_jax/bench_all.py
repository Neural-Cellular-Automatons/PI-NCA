"""Uniform 2-D benchmark matrix: EVERY architecture on EVERY phenomenon.

Previously each phenomenon had a hand-picked architecture list, so scalar PDEs were
measured with five models and multi-field ones with three. That made the tables
incomparable across rows. Every architecture is now channel-generic, so the matrix is
rectangular: the same competitor set runs on all ten phenomena.

Built for long unattended GPU runs:

* **Resumable.** Each (pde, architecture) cell is written to disk as soon as it
  finishes. Re-running skips completed cells, so a crash costs one model, not a night.
* **OOM-tolerant.** A cell that runs out of device memory is retried at half the batch,
  down to a floor, and the batch actually used is recorded alongside the result.
* **Non-fatal cells.** A model that fails for any other reason is recorded as failed and
  the sweep continues; the failures are listed at the end and in the manifest.
* **GPU-only by default.** No silent CPU fallback -- CPU numbers are not comparable to
  GPU numbers and half a matrix measured on each would be meaningless.

Run:  python -m pinca_jax.bench_all --group all
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

from .harness import EmuConfig, field_bounds, run_multiseed
from .equations import pdes
from .models import registry
from . import bench, env, metrics, stats

RES = bench.RESULTS_DIR

GROUPS = {
    "local": ["heat", "allen_cahn", "nagumo", "adv_diff"],
    "multichannel": ["wave", "gray_scott", "shallow_water", "fitzhugh_nagumo"],
    "special": ["cahn_hilliard", "navier_stokes"],
}
ALL_PDES = GROUPS["local"] + GROUPS["multichannel"] + GROUPS["special"]

# "Start from a better point" protocol, ported from the original implementations.
WARMUP = 30
PRESEED = 10


def _cfg(pde, grid, batch, epochs, rollout, eval_steps):
    # Cahn-Hilliard coarsens rapidly from fresh ICs; pre-seeding on developed states
    # mismatches eval and regresses the bounded models to the floor -> preseed off.
    preseed = 0 if pde == "cahn_hilliard" else PRESEED
    return EmuConfig(pde=pde, grid_size=grid, rollout_steps=rollout,
                     eval_steps=eval_steps, epochs=epochs, batch=batch,
                     warmup_epochs=WARMUP, preseed_steps=preseed)


def stats_markdown(pde, results, cfg, seeds):
    """rel-L2 with bootstrap CIs plus Holm-corrected paired tests against the best model.

    This is the table a claim of the form "A beats B on this PDE" has to come from. The
    ranking table next to it can only order point estimates.
    """
    samples = {a: r["_per_ic_rel_l2"] for a, r in results.items()
               if "error" not in r and r.get("_per_ic_rel_l2")}
    if not samples:
        return ""
    cmp = stats.compare_archs(samples, lower_is_better=True)
    ref = cmp["reference"]
    n = len(next(iter(samples.values())))
    lines = [f"### {pde} - rel-L2 with uncertainty and paired tests", "",
             f"n = {n} paired evaluations ({len(seeds)} seed(s) x {cfg.n_eval} held-out "
             f"initial conditions). CIs are 10,000-sample percentile bootstraps. The "
             f"paired column tests each architecture against **{ref}** (best mean) on the "
             f"SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni "
             f"across the {cmp['n_comparisons']} comparisons in this table; `tie` means "
             f"the difference is not resolvable at this sample size, not that the means "
             f"are equal.", "",
             "Note: the mean here is the unweighted mean of per-IC relative errors, "
             "while the ranking table above reports the batch-reduced ratio of norms "
             "(which weights high-energy initial conditions more heavily). The two are "
             "different estimators of the same quantity and will not print equal "
             "numbers; the paired tests need the per-IC form, so it is the one reported "
             "with uncertainty.", "",
             "| architecture | rel-L2 mean | 95% CI | median | vs " + ref +
             " (mean diff) | p (Holm) | verdict |",
             "|---|---|---|---|---|---|---|"]
    order = sorted(samples, key=lambda a: stats.summarize(samples[a]).mean)
    for a in order:
        s = stats.summarize(samples[a])
        if a == ref:
            diff, pv, verdict = "-", "-", "**reference**"
        else:
            t = cmp["comparisons"][a]
            diff = f"{t['mean_diff']:+.3e}"
            pv = f"{t['p_value']:.2g}"
            verdict = ("worse" if t["holm_significant"] and t["better"] == "b"
                       else "better" if t["holm_significant"] else "tie")
        lines.append(f"| {a} | {s.mean:.4e} | [{s.ci_lo:.3e}, {s.ci_hi:.3e}] | "
                     f"{s.median:.4e} | {diff} | {pv} | {verdict} |")
    ties = [a for a in order if a != ref and
            not cmp["comparisons"][a].get("holm_significant")]
    if ties:
        lines += ["", f"Statistically indistinguishable from {ref} at this sample size: "
                      + ", ".join(f"`{t}`" for t in ties) + "."]
    return "\n".join(lines) + "\n"


def _write(tag, pde, results, cfg, seeds):
    base = os.path.join(RES, f"bench_{pde}_{tag}")
    bench.save_results(base + ".json",
                       {"config": asdict(cfg), "seeds": list(seeds), "results": results,
                        "device": env.provenance("bench_all")})
    ok = {a: r for a, r in results.items() if "error" not in r}
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(bench.to_markdown(pde, ok, cfg))
        f.write("\n" + stats_markdown(pde, ok, cfg, seeds))
    return base


def run_cell(pde, arch, cfg, seeds, C, bounds):
    """Train + evaluate one (pde, architecture) cell. Returns (record, batch_used)."""
    ctor_for = lambda b: registry.REGISTRY[arch].make(C, bounds=bounds)

    def attempt(b):
        c = EmuConfig(**{**cfg.__dict__, "batch": b})
        return run_multiseed(ctor_for(b), c, seeds=seeds)

    with bench.CellTimer() as t:
        (runs, agg), used = bench.run_with_oom_backoff(attempt, cfg.batch, min_batch=2,
                                                       label=f"{pde}/{arch}")
    rec = {k: {"mean": v.mean, "std": v.std, "n": v.n} for k, v in agg.items()}
    # Per-IC values, pooled seed-major. Every architecture in this sweep sees the same
    # seeds and the same evaluation ICs in the same order, which is what licenses the
    # paired tests in `stats_markdown`. Without this the tables can only rank means.
    per_ic = metrics.pool_per_ic(runs, "per_ic_rel_l2")
    rec["_per_ic_rel_l2"] = per_ic
    rec["_rel_l2_stats"] = stats.summarize(per_ic).as_dict()
    rec["_per_channel_cons_err"] = runs[0].get("per_channel_cons_err", [])
    rec["_batch_used"] = {"mean": float(used), "std": 0.0, "n": 1}
    rec["_cell_wall_s"] = {"mean": float(t.seconds), "std": 0.0, "n": 1}
    return rec, used


def run_phenomena(pdes_list, seeds, epochs, grid, batch=16, rollout=12, eval_steps=48,
                  force=False, archs=None, manifest=None, tag="full"):
    os.makedirs(RES, exist_ok=True)
    for pde in pdes_list:
        C = pdes.REGISTRY[pde].channels
        wanted = archs or list(registry.bench_archs(C))
        cfg = _cfg(pde, grid, batch, epochs, rollout, eval_steps)
        # Bounded models get this PDE's measured physical range, not a hardcoded
        # [-1,1] that would be nonsense for a field with amplitudes of 5-10.
        bounds = field_bounds(pde, grid)
        path = os.path.join(RES, f"bench_{pde}_{tag}.json")
        results = {} if force else bench.load_results(path, cfg)
        done = [a for a in wanted if a in results and "error" not in results[a]]
        todo = [a for a in wanted if a not in results or "error" in results.get(a, {})]
        print(f"[bench_all] {pde} (C={C}, bounds={None if bounds is None else tuple(round(b, 2) for b in bounds)})"
              f"  {len(todo)} to run, {len(done)} already done")

        for arch in todo:
            try:
                rec, used = run_cell(pde, arch, cfg, seeds, C, bounds)
                results[arch] = rec
                rl, ps = rec["rel_l2"]["mean"], rec["params"]["mean"]
                print(f"    {arch:24s} rel-L2 {rl:.4e}  params {int(ps):>7d}  "
                      f"batch {used}  {rec['_cell_wall_s']['mean']:.0f}s")
                status = "ok"
            except Exception as exc:                      # noqa: BLE001 - recorded, not fatal
                kind = "oom" if bench.is_oom(exc) else "error"
                results[arch] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                 "kind": kind}
                print(f"    {arch:24s} FAILED ({kind}): {type(exc).__name__}")
                status = kind
            if manifest is not None:
                manifest.append({"stage": "bench2d", "pde": pde, "arch": arch,
                                 "status": status})
            _write(tag, pde, results, cfg, seeds)        # checkpoint after every cell
            bench.free_device_memory()
        print(f"  wrote results/bench_{pde}_{tag}.md")


def run_ablations(seeds, epochs, grid, batch=16, rollout=12, eval_steps=48, force=False,
                  manifest=None):
    """A4 (conservation on/off) and A5 (perception size) at matched backbone width.

    These stay scalar-field studies: they are controls, not competitors, and their
    whole point is that only one factor differs.
    """
    os.makedirs(RES, exist_ok=True)
    # A1: does bounding help, and at what cost to conservation? (stiff bounded fields)
    # A4: conservation on/off at matched backbone width.
    # A5: perception size at matched head and width.
    # A7: HOW mass is restored after the bound clip -- the naive uniform offset re-violates
    #     the bound it just enforced, so this measures what that costs.
    plan = [("A4", ["heat", "nagumo"], ["abl_flux", "abl_residual"]),
            ("A5", ["heat", "navier_stokes"], ["abl_k3", "abl_k5", "abl_multiscale"]),
            ("A7", ["cahn_hilliard", "allen_cahn"],
             ["abl_proj_none", "abl_proj_uniform", "abl_proj_headroom"]),
            ("A1", ["cahn_hilliard", "allen_cahn"],
             ["multiscale_flux_nca", "bounded_multiscale_nca"])]
    for tag, pde_list, archs in plan:
        for pde in pde_list:
            C = pdes.REGISTRY[pde].channels
            cfg = _cfg(pde, grid, batch, epochs, rollout, eval_steps)
            path = os.path.join(RES, f"bench_{pde}_{tag}.json")
            results = {} if force else bench.load_results(path, cfg)
            todo = [a for a in archs if a not in results or "error" in results.get(a, {})]
            # A1 and A7 are about bounds, so they need the measured physical range;
            # A4 and A5 are unbounded controls and must not get one.
            bounds = field_bounds(pde, grid) if tag in ("A1", "A7") else None
            for arch in todo:
                try:
                    rec, used = run_cell(pde, arch, cfg, seeds, C, bounds)
                    results[arch] = rec
                    print(f"  {tag} {pde} {arch:16s} rel-L2 {rec['rel_l2']['mean']:.4e} "
                          f"params {int(rec['params']['mean'])}")
                    status = "ok"
                except Exception as exc:                  # noqa: BLE001
                    kind = "oom" if bench.is_oom(exc) else "error"
                    results[arch] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                     "kind": kind}
                    print(f"  {tag} {pde} {arch:16s} FAILED ({kind})")
                    status = kind
                if manifest is not None:
                    manifest.append({"stage": f"ablation_{tag}", "pde": pde,
                                     "arch": arch, "status": status})
                _write(tag, pde, results, cfg, seeds)
                bench.free_device_memory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    choices=["all", "local", "multichannel", "special", "ablation"])
    ap.add_argument("--seeds", type=int, default=1,
                    help="1 = single fixed seed 42; >1 = mean +/- std over seeds")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16, help="training batch (raise on GPU)")
    ap.add_argument("--rollout", type=int, default=12, help="training BPTT horizon")
    ap.add_argument("--eval", type=int, default=48, help="evaluation rollout horizon")
    ap.add_argument("--archs", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--pdes", default=None,
                    help="comma-separated phenomena (default: the group's full list)")
    ap.add_argument("--tag", default="full",
                    help="results/bench_<pde>_<tag>.json; use a distinct tag for a "
                         "higher-seed headline run so it does not overwrite the matrix")
    ap.add_argument("--force", action="store_true", help="recompute cells already on disk")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit the CPU backend (results are NOT comparable to GPU runs)")
    args = ap.parse_args()

    env.require_gpu("bench_all", allow_cpu=args.allow_cpu)
    seeds = (42,) if args.seeds <= 1 else tuple(range(args.seeds))
    archs = args.archs.split(",") if args.archs else None
    manifest = []
    common = dict(seeds=seeds, epochs=args.epochs, grid=args.grid, batch=args.batch,
                  rollout=args.rollout, eval_steps=args.eval, force=args.force,
                  manifest=manifest)

    if args.pdes:
        wanted = [p for p in args.pdes.split(",") if p in ALL_PDES]
        missing = [p for p in args.pdes.split(",") if p not in ALL_PDES]
        if missing:
            raise SystemExit(f"[bench_all] unknown phenomena: {missing}; "
                             f"known: {ALL_PDES}")
        run_phenomena(wanted, archs=archs, tag=args.tag, **common)
    else:
        for key in ("local", "multichannel", "special"):
            if args.group in ("all", key):
                run_phenomena(GROUPS[key], archs=archs, tag=args.tag, **common)
    if args.group in ("all", "ablation"):
        run_ablations(seeds=seeds, epochs=args.epochs, grid=args.grid, batch=args.batch,
                      rollout=args.rollout, eval_steps=args.eval, force=args.force,
                      manifest=manifest)

    bad = [m for m in manifest if m["status"] != "ok"]
    print(f"[bench_all] done. {len(manifest) - len(bad)} cells ok, {len(bad)} failed. "
          f"peak device mem {env.peak_mem_mb():.0f} MB")
    for m in bad:
        print(f"  FAILED {m['stage']} {m['pde']}/{m['arch']} ({m['status']})")


if __name__ == "__main__":
    main()
