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
from . import bench, env

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


def _write(tag, pde, results, cfg, seeds):
    base = os.path.join(RES, f"bench_{pde}_{tag}")
    bench.save_results(base + ".json",
                       {"config": asdict(cfg), "seeds": list(seeds), "results": results,
                        "device": env.provenance("bench_all")})
    ok = {a: r for a, r in results.items() if "error" not in r}
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(bench.to_markdown(pde, ok, cfg))
    return base


def run_cell(pde, arch, cfg, seeds, C, bounds):
    """Train + evaluate one (pde, architecture) cell. Returns (record, batch_used)."""
    ctor_for = lambda b: registry.REGISTRY[arch].make(C, bounds=bounds)

    def attempt(b):
        c = EmuConfig(**{**cfg.__dict__, "batch": b})
        _, agg = run_multiseed(ctor_for(b), c, seeds=seeds)
        return agg

    with bench.CellTimer() as t:
        agg, used = bench.run_with_oom_backoff(attempt, cfg.batch, min_batch=2,
                                               label=f"{pde}/{arch}")
    rec = {k: {"mean": v.mean, "std": v.std, "n": v.n} for k, v in agg.items()}
    rec["_batch_used"] = {"mean": float(used), "std": 0.0, "n": 1}
    rec["_cell_wall_s"] = {"mean": float(t.seconds), "std": 0.0, "n": 1}
    return rec, used


def run_phenomena(pdes_list, seeds, epochs, grid, batch=16, rollout=12, eval_steps=48,
                  force=False, archs=None, manifest=None):
    os.makedirs(RES, exist_ok=True)
    for pde in pdes_list:
        C = pdes.REGISTRY[pde].channels
        wanted = archs or list(registry.bench_archs(C))
        cfg = _cfg(pde, grid, batch, epochs, rollout, eval_steps)
        # Bounded models get this PDE's measured physical range, not a hardcoded
        # [-1,1] that would be nonsense for a field with amplitudes of 5-10.
        bounds = field_bounds(pde, grid)
        path = os.path.join(RES, f"bench_{pde}_full.json")
        results = {} if force else bench.load_results(path)
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
            _write("full", pde, results, cfg, seeds)     # checkpoint after every cell
            bench.free_device_memory()
        print(f"  wrote results/bench_{pde}_full.md")


def run_ablations(seeds, epochs, grid, batch=16, rollout=12, eval_steps=48, force=False,
                  manifest=None):
    """A4 (conservation on/off) and A5 (perception size) at matched backbone width.

    These stay scalar-field studies: they are controls, not competitors, and their
    whole point is that only one factor differs.
    """
    os.makedirs(RES, exist_ok=True)
    plan = [("A4", ["heat", "nagumo"], ["abl_flux", "abl_residual"]),
            ("A5", ["heat", "navier_stokes"], ["abl_k3", "abl_k5", "abl_multiscale"])]
    for tag, pde_list, archs in plan:
        for pde in pde_list:
            C = pdes.REGISTRY[pde].channels
            cfg = _cfg(pde, grid, batch, epochs, rollout, eval_steps)
            path = os.path.join(RES, f"bench_{pde}_{tag}.json")
            results = {} if force else bench.load_results(path)
            todo = [a for a in archs if a not in results or "error" in results.get(a, {})]
            for arch in todo:
                try:
                    rec, used = run_cell(pde, arch, cfg, seeds, C, None)
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

    for key in ("local", "multichannel", "special"):
        if args.group in ("all", key):
            run_phenomena(GROUPS[key], archs=archs, **common)
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
