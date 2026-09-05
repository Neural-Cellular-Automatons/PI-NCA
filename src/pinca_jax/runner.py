"""Single entry point for the full benchmark run.

    python -m pinca_jax.runner

That is the whole command. It runs, in order: the correctness gate, the uniform 2-D
matrix, the ablations, the uniform 3-D matrix, the resolution study, the continuous
baselines, trajectory capture, the field figures, the benchmark plots, and finally
regenerates the report (Markdown + PDF) from whatever results exist.

Design notes, all of which exist because this runs unattended for hours:

* **GPU or nothing.** The run refuses to start on the CPU backend. Mixing CPU and GPU
  measurements inside one table is worse than having no table.
* **Every stage is resumable.** Benchmarks checkpoint per (pde, architecture) cell, so
  re-running after a crash or a Ctrl-C picks up where it stopped.
* **Only the measurement stages are fatal.** Figures, baselines and the report cannot
  destroy hours of completed benchmarking; their failures are collected and reported.
* **Timings and failures are written to results/run_manifest.json**, so the run can be
  audited afterwards without scrolling the log.

Useful flags:
    --profile smoke|bench|full   scale preset (default full)
    --only bench2d,plots         run just these stages
    --skip figures               skip stages
    --force                      recompute cells already on disk
    --allow-cpu                  development escape hatch; NOT for real numbers
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from . import bench, env

RES = bench.RESULTS_DIR
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# grid / batch / epochs per scale preset.
PROFILES = {
    # ~2 minutes end to end: proves every stage wires up, numbers are meaningless.
    "smoke": dict(seeds=1, epochs=40, grid=16, batch=8, rollout=4, eval=12,
                  grid3d=8, epochs3d=20, batch3d=2, res_epochs=20,
                  viz_grid=16, viz_epochs=40, viz3d_grid=8, viz3d_epochs=20, max_mb=8),
    # measurements only, no field figures.
    "bench": dict(seeds=3, epochs=2000, grid=64, batch=64, rollout=12, eval=48,
                  grid3d=32, epochs3d=800, batch3d=16, res_epochs=600,
                  viz_grid=48, viz_epochs=400, viz3d_grid=16, viz3d_epochs=200,
                  max_mb=64),
    "full": dict(seeds=3, epochs=2000, grid=64, batch=64, rollout=12, eval=48,
                 grid3d=32, epochs3d=800, batch3d=16, res_epochs=600,
                 viz_grid=48, viz_epochs=400, viz3d_grid=16, viz3d_epochs=200,
                 max_mb=64),
}

VIZ_2D = ["heat", "allen_cahn", "nagumo", "adv_diff", "gray_scott", "shallow_water",
          "fitzhugh_nagumo", "wave", "cahn_hilliard", "navier_stokes"]
VIZ_3D = ["heat", "adv_diff", "allen_cahn", "nagumo", "gray_scott", "fitzhugh_nagumo"]

REPORT_MD = os.path.join(ROOT, "docs", "PI-NCA_Architectures_and_Results.md")


class Run:
    """Executes stages as subprocesses and records what happened.

    Subprocesses, not in-process calls, because each benchmark stage allocates a lot of
    device memory; letting the process exit is the only reliable way to hand every byte
    back before the next stage starts.
    """

    def __init__(self, allow_cpu=False, force=False):
        self.allow_cpu = allow_cpu
        self.force = force
        self.manifest = []
        self.failures = []

    def _cmd(self, module, args):
        cmd = [sys.executable, "-u", "-m", f"pinca_jax.{module}"] + [str(a) for a in args]
        if self.force and module in ("bench_all", "bench3d"):
            cmd.append("--force")
        if self.allow_cpu and module in ("bench_all", "bench3d", "res_study"):
            cmd.append("--allow-cpu")
        return cmd

    def stage(self, name, module, args=(), fatal=True, raw=None):
        cmd = raw or self._cmd(module, args)
        print(f"\n{'=' * 72}\n== {name}\n{'=' * 72}", flush=True)
        t0 = time.time()
        # A child writing to a pipe is block-buffered, so `bash run_gpu.sh | tee log`
        # would show nothing for hours. Force line-by-line output so progress is live.
        child_env = dict(os.environ, PYTHONUNBUFFERED="1")
        rc = subprocess.call(cmd, cwd=ROOT, env=child_env)
        dt = time.time() - t0
        ok = rc == 0
        self.manifest.append({"stage": name, "seconds": round(dt, 1), "rc": rc,
                              "ok": ok, "cmd": " ".join(cmd)})
        if not ok:
            self.failures.append(name)
            if fatal:
                print(f"\n!! {name} failed (exit {rc}) and is required. Stopping.")
                self.write_manifest()
                sys.exit(rc)
            print(f"!! {name} failed (exit {rc}); continuing — later stages do not "
                  f"depend on it.")
        else:
            print(f"-- {name} finished in {dt / 60:.1f} min")
        return ok

    def write_manifest(self):
        total = sum(m["seconds"] for m in self.manifest)
        bench.save_results(os.path.join(RES, "run_manifest.json"),
                           {"results": {}, "stages": self.manifest,
                            "total_seconds": round(total, 1),
                            "failures": self.failures,
                            "device": env.provenance("runner")})


def main():
    ap = argparse.ArgumentParser(description="Run the whole PI-NCA benchmark suite.")
    ap.add_argument("--profile", default="full", choices=list(PROFILES))
    ap.add_argument("--only", default=None, help="comma-separated stage names")
    ap.add_argument("--skip", default=None, help="comma-separated stage names")
    ap.add_argument("--force", action="store_true", help="recompute finished cells")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="development only; the numbers are not comparable to a GPU run")
    ap.add_argument("--no-gate", action="store_true", help="skip the test suite")
    args = ap.parse_args()

    env.configure_memory()
    P = PROFILES[args.profile]
    # Fail here, before anything expensive, rather than three hours in.
    env.require_gpu("runner", allow_cpu=args.allow_cpu)

    r = Run(allow_cpu=args.allow_cpu, force=args.force)
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    if args.profile == "bench":
        skip |= {"capture", "figures"}

    def want(name):
        return (only is None or name in only) and name not in skip

    t_start = time.time()

    if want("gate") and not args.no_gate:
        r.stage("gate: correctness tests", None, fatal=True,
                raw=[sys.executable, "-u", "-m", "pytest", "tests/", "-q"])

    if want("bench2d"):
        r.stage("2-D matrix: every architecture x every phenomenon", "bench_all",
                ["--group", "all", "--seeds", P["seeds"], "--epochs", P["epochs"],
                 "--grid", P["grid"], "--batch", P["batch"], "--rollout", P["rollout"],
                 "--eval", P["eval"]], fatal=True)

    if want("bench3d"):
        r.stage("3-D matrix: every architecture x every phenomenon", "bench3d",
                ["--grid", P["grid3d"], "--epochs", P["epochs3d"],
                 "--batch", P["batch3d"]], fatal=True)

    if want("resolution"):
        r.stage("resolution transfer study", "res_study",
                ["--pdes", "heat,allen_cahn,navier_stokes",
                 "--epochs", P["res_epochs"]], fatal=False)

    # Plot as soon as measurements exist, so a later crash still leaves figures.
    if want("plots"):
        r.stage("benchmark plots (interim)", "plots", fatal=False)

    if want("baselines"):
        for mod in ("pinn_heat", "deeponet_heat", "darcy"):
            r.stage(f"baseline: {mod}", mod, fatal=False)

    if want("capture"):
        r.stage("capture trajectories for figures", "capture",
                ["--dims", "both", "--grid", P["viz_grid"], "--epochs", P["viz_epochs"],
                 "--grid3d", P["viz3d_grid"], "--epochs3d", P["viz3d_epochs"],
                 "--max-mb", P["max_mb"]], fatal=False)

    if want("figures"):
        traj = os.path.join(RES, "traj")
        for pde in VIZ_2D:
            f = os.path.join(traj, f"{pde}_2d.npz")
            if os.path.exists(f):
                r.stage(f"figure: {pde} (2-D)", "viz", ["--npz", f], fatal=False)
        for pde in VIZ_3D:
            f = os.path.join(traj, f"{pde}_3d.npz")
            if os.path.exists(f):
                r.stage(f"figure: {pde} (3-D slice)", "viz3d", ["--npz", f], fatal=False)
                r.stage(f"figure: {pde} (3-D volume)", "viz3d_volume", ["--npz", f],
                        fatal=False)

    if want("plots"):
        r.stage("benchmark plots (final)", "plots", fatal=False)

    if want("report"):
        r.stage("architecture diagrams", "arch_figs", fatal=False)
        r.stage("report: regenerate Markdown from results", "report", fatal=False)
        r.stage("report: render PDF", "md2pdf", [REPORT_MD], fatal=False)

    r.write_manifest()
    total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"RUN COMPLETE in {total / 3600:.2f} h ({total / 60:.0f} min)")
    print(f"  backend        {env.provenance('runner')['backend']}  "
          f"peak {env.peak_mem_mb():.0f} MB")
    print(f"  tables         results/*.md")
    print(f"  plots          docs/figures/bench/*.png")
    print(f"  report         docs/PI-NCA_Architectures_and_Results.{{md,pdf}}")
    print(f"  raw data       results/traj/*.npz")
    print(f"  audit          results/run_manifest.json")
    if r.failures:
        print(f"\n  {len(r.failures)} non-fatal stage(s) failed:")
        for f in r.failures:
            print(f"    - {f}")
        print("  The benchmark tables above are unaffected.")
    else:
        print("\n  No failures.")
    _summarise_cells()


def _summarise_cells():
    """Count completed vs failed matrix cells across every results file."""
    import glob
    ok = failed = 0
    bad = []
    for path in glob.glob(os.path.join(RES, "bench*_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                res = json.load(f).get("results", {})
        except (OSError, json.JSONDecodeError):
            continue
        for arch, rec in res.items():
            if isinstance(rec, dict) and "error" in rec:
                failed += 1
                bad.append(f"{os.path.basename(path)[:-5]}/{arch} ({rec.get('kind')})")
            else:
                ok += 1
    print(f"\n  matrix cells: {ok} succeeded, {failed} failed")
    for b in bad[:20]:
        print(f"    x {b}")


if __name__ == "__main__":
    main()
