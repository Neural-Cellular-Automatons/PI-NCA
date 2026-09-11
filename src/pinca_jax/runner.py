"""One command produces every number, table and figure in the paper.

    python -m pinca_jax.runner

That is the whole thing. On a GPU box it runs the correctness gate, the uniform 2-D
matrix, the ablations, the multi-seed headline comparison, the teacher-error study, the
out-of-distribution study, the stability stress test, the scaling / rank-stability study,
the uniform 3-D matrix, the resolution study, the continuous and matched baselines,
trajectory capture, the field figures, the benchmark plots, the claims audit, the
bibliography check, the generated paper tables, the Markdown report and -- if a LaTeX
toolchain is present -- the paper PDF.

Design notes, all of which exist because this runs unattended for many hours:

* **GPU or nothing.** The run refuses to start on the CPU backend. Mixing CPU and GPU
  measurements inside one table is worse than having no table.
* **Everything is resumable, at two levels, and resume is condition-aware.** Benchmarks
  checkpoint per (pde, architecture) cell, and the runner additionally skips a whole stage
  whose outputs are already on disk. Re-running after a crash, a Ctrl-C or a reboot picks
  up where it stopped instead of repeating a night of compute.

  Crucially, "already on disk" is not the same as "reusable". A stored result is reused
  only when its backend and its scale match the run about to happen; otherwise it is
  recomputed and the reason is printed. Without that, the first GPU run in a checkout that
  ships CPU results would silently keep them and emit a table that is half one backend and
  half the other, and a `paper` run after a `smoke` run would inherit the smoke numbers.
  `--force` recomputes everything regardless.
* **Only the 2-D matrix is fatal.** Every other stage records its failure and the run
  continues, because a 3-D out-of-memory error must not destroy a completed 2-D sweep.
* **The paper artifacts are always regenerated**, even if a measurement stage failed
  earlier, so what is on disk always reflects what was actually measured. That happens in
  a `finally` block.
* **Timings, failures and provenance go to results/run_manifest_<backend>.json**, so a
  CPU wiring check cannot overwrite the record of a GPU run, and the run can be
  audited afterwards without scrolling the log.

Know the cost before starting it:

    python -m pinca_jax.runner --estimate

That trains two real cells at the chosen profile, one cheap architecture and one
expensive, and projects the wall-clock per stage from the measured rate. It is the honest
way to find out whether a preset is an overnight run or a three-day one on *your* card.

Flags:
    --profile paper|full|bench|smoke   scale preset (default: paper)
    --estimate                         measure and project the cost, then exit
    --list-stages                      print the stage names --only/--skip accept
    --only bench2d,plots               run just these stages
    --skip figures                     skip stages
    --force                            recompute finished cells and stages
    --allow-cpu                        development escape hatch; NOT for real numbers
    --no-gate                          skip the correctness suite (do not, for real runs)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

from . import bench, env

RES = bench.RESULTS_DIR
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPER_DIR = os.path.join(ROOT, "paper")
REPORT_MD = os.path.join(ROOT, "docs", "PI-NCA_Architectures_and_Results.md")

# The three regimes the regime map distinguishes: smooth diffusive, stiff bounded
# phase-separating, and advective/turbulent. The expensive high-seed-count studies run on
# these rather than on all ten, because a 10-seed x 14-architecture sweep over the whole
# suite costs more than the rest of the pipeline combined and buys resolution on phenomena
# whose ranking the 5-seed matrix already settles.
HEADLINE_PDES = "heat,cahn_hilliard,navier_stokes"

VIZ_2D = ["heat", "allen_cahn", "nagumo", "adv_diff", "gray_scott", "shallow_water",
          "fitzhugh_nagumo", "wave", "cahn_hilliard", "navier_stokes"]
VIZ_3D = ["heat", "adv_diff", "allen_cahn", "nagumo", "gray_scott", "fitzhugh_nagumo"]

# Architecture subsets for the studies that would otherwise dominate the run. Each spans
# the three families under test -- unconstrained local, conservative local, global
# spectral -- plus a physics-free control, so the subset can still answer the question the
# study exists to ask.
OOD_ARCHS = "plain_nca,pi_nca,multiscale_flux_nca,fno,resnet,unet"
STAB_ARCHS = ("plain_nca,pi_nca,multiscale_flux_nca,bounded_multiscale_nca,"
              "abl_proj_uniform,abl_proj_headroom,fno,resnet,unet,identity")
SCALING_ARCHS = "plain_nca,pi_nca,multiscale_flux_nca,fno,resnet_iso,identity"

# grid / batch / epochs per scale preset.
PROFILES = {
    # Roughly twenty minutes on a laptop CPU, a few on a GPU: proves every stage wires
    # up. The numbers are meaningless and every driver says so in its own output.
    "smoke": dict(seeds=1, epochs=25, grid=12, batch=4, rollout=3, eval=8,
                  grid3d=8, epochs3d=15, batch3d=2, res_epochs=15,
                  viz_grid=12, viz_epochs=25, viz3d_grid=8, viz3d_epochs=15, max_mb=8,
                  headline_seeds=2, n_eval=4, ood_epochs=25, stab_epochs=25,
                  scal_grids="12,16", scal_rollouts="2,3", scal_epochs="15,25",
                  pinn_grid=12, pinn_iters=150, deeponet_seeds=1,
                  darcy_seeds=1, darcy_iters=60, darcy_ntrain=32,
                  jepa_epochs=30, probe_epochs=20,
                  sweep_grid=12, sweep_epochs=6, sweep_jepa_epochs=6,
                  sweep_probe_epochs=4, sweep_seeds=1,
                  sweep_variants="fno_oneshot,fno_multi,nca_multi",
                  # A wiring check must still touch every phenomenon -- that is where
                  # shape and channel bugs live -- but it does not need every
                  # architecture, and running all fourteen turns "minutes" into an hour.
                  archs="plain_nca,pi_nca,resnet_iso,identity",
                  ood_archs="plain_nca,fno,identity",
                  stab_archs="plain_nca,bounded_multiscale_nca,identity",
                  scal_archs="plain_nca,fno,identity",
                  res_archs="plain_nca,fno"),
    # The intended preset for producing the paper. Sized so the whole pipeline finishes
    # overnight on one mid-range GPU rather than over a long weekend; --estimate will tell
    # you what it actually costs on your card before you commit to it.
    "paper": dict(seeds=5, epochs=1200, grid=48, batch=32, rollout=12, eval=48,
                  grid3d=24, epochs3d=600, batch3d=8, res_epochs=500,
                  viz_grid=48, viz_epochs=400, viz3d_grid=16, viz3d_epochs=200,
                  max_mb=64, headline_seeds=10, n_eval=32, ood_epochs=800,
                  stab_epochs=800,
                  scal_grids="24,32,48", scal_rollouts="4,8,12",
                  scal_epochs="300,600,1200",
                  pinn_grid=32, pinn_iters=6000, deeponet_seeds=3,
                  darcy_seeds=3, darcy_iters=2000, darcy_ntrain=256,
                  jepa_epochs=800, probe_epochs=400,
                  # The screen stays deliberately smaller than the headline
                  # budget: its job is to rank ten variants and rule most of
                  # them out, after which the survivor is worth the full
                  # `jepa` stage at the paper budget.
                  sweep_grid=32, sweep_epochs=400, sweep_jepa_epochs=300,
                  sweep_probe_epochs=200, sweep_seeds=2, sweep_variants=None,
                  archs=None, ood_archs=OOD_ARCHS,
                  stab_archs=STAB_ARCHS, scal_archs=SCALING_ARCHS,
                  res_archs=None),
    # Everything at the largest scale attempted. Expect multiple days on one GPU.
    "full": dict(seeds=5, epochs=2000, grid=64, batch=64, rollout=12, eval=48,
                 grid3d=32, epochs3d=800, batch3d=16, res_epochs=600,
                 viz_grid=48, viz_epochs=400, viz3d_grid=16, viz3d_epochs=200,
                 max_mb=64, headline_seeds=10, n_eval=32, ood_epochs=1200,
                 stab_epochs=1200,
                 scal_grids="32,48,64", scal_rollouts="4,8,12",
                 scal_epochs="500,1000,2000",
                 pinn_grid=32, pinn_iters=8000, deeponet_seeds=3,
                 darcy_seeds=3, darcy_iters=3000, darcy_ntrain=512,
                 jepa_epochs=1200, probe_epochs=600,
                 sweep_grid=48, sweep_epochs=800, sweep_jepa_epochs=600,
                 sweep_probe_epochs=400, sweep_seeds=3, sweep_variants=None,
                  archs=None, ood_archs=OOD_ARCHS,
                  stab_archs=STAB_ARCHS, scal_archs=SCALING_ARCHS,
                  res_archs=None),
}
# `bench` is `paper` without the field figures, for when only the measurements are wanted.
PROFILES["bench"] = dict(PROFILES["paper"])

# Ordered, with the one-line description printed by --list-stages.
STAGES = [
    ("gate", "correctness suite (245 tests) -- fatal, nothing downstream is trustworthy without it"),
    ("bench2d", "uniform 2-D matrix + ablations A1/A4/A5/A7 -- the only other fatal stage"),
    ("headline", "high-seed-count paired comparison on the three regime representatives"),
    ("teacher", "the reference solver's own error, and whether it converges at all"),
    ("ood", "held-out initial-condition families, PDE coefficients and horizons"),
    ("stability", "long-horizon failure rates with the divergence guard DISABLED"),
    ("scaling", "does the ranking survive a change of grid, horizon and training budget?"),
    ("jepa", "latent self-supervised pretraining vs distillation, same architecture"),
    ("jepa_sweep", "OPT-IN (--only jepa_sweep): screen ten latent world-model variants"),
    ("bench3d", "the same comparison in three dimensions"),
    ("resolution", "train at one grid, evaluate at every other"),
    ("baselines", "PINN, DeepONet, Darcy, and the matched PINN-vs-emulator comparison"),
    ("capture", "train once per phenomenon and archive raw trajectories for the figures"),
    ("figures", "field montages and 3-D volume renders, from the captured trajectories"),
    ("plots", "benchmark plots (runs twice: once early, once at the end)"),
    ("claims", "claims audit, bibliography verification, generated paper tables"),
    ("report", "architecture diagrams + the Markdown/PDF architectures-and-results report"),
    ("pdf", "compile paper/main.tex, if a LaTeX toolchain is installed"),
]
FINALISATION = {"plots", "claims", "report", "pdf"}
# Stages that run only when named in --only. A variant screen is exploratory: it is a
# sized-to-fit-a-small-GPU ranking, not a paper number, and it must not silently add
# hours to the one-command run that produces the paper.
OPT_IN = {"jepa_sweep"}


class Run:
    """Executes stages as subprocesses and records what happened.

    Subprocesses, not in-process calls, because each benchmark stage allocates a lot of
    device memory; letting the process exit is the only reliable way to hand every byte
    back before the next stage starts.
    """

    def __init__(self, allow_cpu=False, force=False, grid=None):
        self.allow_cpu = allow_cpu
        self.force = force
        self.grid = grid
        self.manifest = []
        self.failures = []
        self.skipped = []

    def _cmd(self, module, args):
        cmd = [sys.executable, "-u", "-m", f"pinca_jax.{module}"] + [str(a) for a in args]
        if self.force and module in ("bench_all", "bench3d"):
            cmd.append("--force")
        if self.allow_cpu and module in ("bench_all", "bench3d", "res_study", "ood",
                                         "stability", "teacher_error", "matched",
                                         "scaling", "jepa"):
            cmd.append("--allow-cpu")
        return cmd

    def stage(self, name, module, args=(), fatal=True, raw=None, outputs=()):
        """Run one stage. `outputs` are files whose presence means it is already done.

        Stage-level resume matters as much as cell-level resume here: the studies that do
        not checkpoint internally (OOD, stability, scaling, matched) would otherwise redo
        hours of work every time the run is restarted after a crash further along.
        """
        if outputs and not self.force and all(_reusable(p, self.grid) for p in outputs):
            print(f"\n-- {name}: already on disk from a matching run, skipping "
                  f"(--force recomputes)")
            self.skipped.append(name)
            self.manifest.append({"stage": name, "seconds": 0.0, "rc": 0, "ok": True,
                                  "skipped": True})
            return True
        cmd = raw or self._cmd(module, args)
        print(f"\n{'=' * 72}\n== {name}\n{'=' * 72}", flush=True)
        t0 = time.time()
        # A child writing to a pipe is block-buffered, so `bash run_paper.sh | tee log`
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
                raise FatalStage(name)
            print(f"!! {name} failed (exit {rc}); continuing -- later stages do not "
                  f"depend on it.")
        else:
            print(f"-- {name} finished in {dt / 60:.1f} min")
        return ok

    def write_manifest(self):
        # Suffixed by backend. A CPU wiring check writing to a shared filename silently
        # destroys the manifest of the GPU run the paper cites -- which happened once
        # here, and cost a restore from a copy that happened to still exist.
        total = sum(m["seconds"] for m in self.manifest)
        backend = env.provenance("runner")["backend"]
        bench.save_results(os.path.join(RES, f"run_manifest_{backend}.json"),
                           {"results": {}, "stages": self.manifest,
                            "total_seconds": round(total, 1),
                            "failures": self.failures, "skipped": self.skipped,
                            "device": env.provenance("runner")})


def _reusable(path, grid=None):
    """Can this stage output be reused, or must the stage be re-run?

    Existence is not enough. This repository ships CPU results, so a stage that merely
    checked for the file would let the first GPU run keep them and produce a study that is
    half one backend and half the other. The stored backend must match the current one,
    and -- for the studies that record it -- so must the grid, or a `paper` run after a
    `smoke` run would silently inherit the smoke numbers.
    """
    if not os.path.exists(path):
        return False
    try:
        import jax
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:                                  # noqa: BLE001 - recompute on doubt
        return False
    stored = (blob.get("device") or {}).get("backend")
    if stored and stored != jax.default_backend():
        print(f"   [resume] {os.path.basename(path)} was produced on '{stored}', now on "
              f"'{jax.default_backend()}' -- recomputing")
        return False
    old_grid = blob.get("grid", (blob.get("config") or {}).get("grid_size"))
    if grid is not None and old_grid is not None and int(old_grid) != int(grid):
        print(f"   [resume] {os.path.basename(path)} was produced at grid {old_grid}, "
              f"now {grid} -- recomputing")
        return False
    return True


class FatalStage(RuntimeError):
    """A required stage failed. Finalisation still runs; the run then exits non-zero."""


# ----------------------------------------------------------------- estimation ---
def _n_archs():
    from .models import registry
    return len(registry.BENCH_ARCHS)


def _n(spec, default):
    """Size of an architecture subset, or the full matrix when the profile sets none."""
    return len(spec.split(",")) if spec else default


def cell_counts(P):
    """Training runs per stage, so the estimate is arithmetic rather than a guess."""
    from .equations import pdes
    n_arch = _n(P.get("archs"), _n_archs())
    n_pde = len(pdes.REGISTRY)
    n_head = len(HEADLINE_PDES.split(","))
    seeds, hseeds = P["seeds"], P["headline_seeds"]
    small = min(3, seeds)
    n_scal = (len(P["scal_grids"].split(",")) + len(P["scal_rollouts"].split(","))
              + len(P["scal_epochs"].split(",")))
    return {
        "bench2d": n_pde * n_arch * seeds,
        # A4 2x2 + A5 2x3 + A7 2x3 + A1 2x2 = 20 cells, run by the same stage
        "ablations": 20 * seeds,
        "headline": n_head * n_arch * hseeds,
        "ood": n_head * _n(P.get("ood_archs"), 6) * small,
        "stability": n_head * _n(P.get("stab_archs"), 10) * small,
        "scaling": n_head * n_scal * _n(P.get("scal_archs"), 6),
        "bench3d": 6 * n_arch,
        "resolution": 3 * 4 * _n(P.get("res_archs"), 4),
        "capture": len(VIZ_2D) + len(VIZ_3D),
        "matched (PINN runs)": P["n_eval"] + 1,
    }


def measure_rate(P, allow_cpu=False):
    """Time one cheap and one expensive training at this profile's settings.

    Two architectures rather than one because the spread between a 5e3-parameter cellular
    automaton and a 5.9e5-parameter spectral operator is the dominant uncertainty in any
    projection, and quoting a single number would hide it.
    """
    from .harness import EmuConfig, train_emulator
    from .models import registry
    import jax
    out = {}
    for arch in ("plain_nca", "fno"):
        cfg = EmuConfig(pde="heat", grid_size=P["grid"], batch=P["batch"],
                        rollout_steps=P["rollout"], eval_steps=P["eval"],
                        epochs=P["epochs"], warmup_epochs=min(30, P["epochs"] // 4))
        t0 = time.time()
        tr = train_emulator(registry.REGISTRY[arch].make(1), cfg)
        jax.block_until_ready(tr["params"])
        out[arch] = time.time() - t0
        print(f"  {arch:12s} {out[arch]:7.1f} s / training "
              f"(grid {P['grid']}, {P['epochs']} epochs, batch {P['batch']})")
    return out


def estimate(profile, allow_cpu=False):
    P = PROFILES[profile]
    print(f"[estimate] profile '{profile}': grid {P['grid']}, {P['epochs']} epochs, "
          f"batch {P['batch']}, {P['seeds']} seeds "
          f"({P['headline_seeds']} for the headline stage)")
    print(f"[estimate] backend {env.provenance('estimate')['backend']}; "
          f"timing two real trainings...")
    rate = measure_rate(P, allow_cpu)
    lo, hi = min(rate.values()), max(rate.values())
    counts = cell_counts(P)
    total_cells = sum(counts.values())
    print(f"\n{'stage':<28}{'trainings':>10}{'low (h)':>10}{'high (h)':>10}")
    print("-" * 58)
    for k, v in counts.items():
        mark = "  (runs inside bench2d)" if k == "ablations" else ""
        print(f"{k:<28}{v:>10}{v * lo / 3600:>10.1f}{v * hi / 3600:>10.1f}{mark}")
    print("-" * 58)
    print(f"{'TOTAL':<28}{total_cells:>10}{total_cells * lo / 3600:>10.1f}"
          f"{total_cells * hi / 3600:>10.1f}")
    print(f"\nThe two columns are the cheapest and the most expensive architecture in the\n"
          f"matrix; the truth is between them and nearer the low end, because most of the\n"
          f"architectures are small. Figures, plots, the report and the audit add minutes,\n"
          f"not hours.\n"
          f"\nIf that is too long: --profile bench drops the field figures, and\n"
          f"--skip scaling,resolution,bench3d removes the three least central studies.\n"
          f"Every stage resumes, so starting it and stopping it later is safe.")


# ---------------------------------------------------------------------- paper ---
def build_pdf(r: Run):
    """Compile paper/main.tex if a LaTeX toolchain is installed; say so clearly if not."""
    tex = shutil.which("pdflatex") or shutil.which("xelatex")
    if not tex:
        print("\n-- pdf: no pdflatex/xelatex on PATH; skipping the PDF build.\n"
              "   Everything it needs is already generated:\n"
              "     paper/main.tex, paper/appendix.tex, paper/refs.bib,\n"
              "     paper/generated/*.tex  (all tables and every quoted number)\n"
              "   Compile anywhere with:  cd paper && pdflatex main && bibtex main && "
              "pdflatex main && pdflatex main")
        r.skipped.append("pdf (no LaTeX toolchain)")
        return False
    name = os.path.basename(tex)
    steps = [[tex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]
    if shutil.which("bibtex"):
        steps.append(["bibtex", "main"])
        steps.append([tex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    steps.append([tex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    for i, cmd in enumerate(steps, 1):
        rc = subprocess.call(cmd, cwd=PAPER_DIR,
                             stdout=subprocess.DEVNULL if i < len(steps) else None)
        if rc != 0 and cmd[0] != "bibtex":   # bibtex warns noisily on a first pass
            print(f"!! pdf: {name} pass {i} failed (exit {rc}); see paper/main.log")
            r.failures.append("pdf")
            return False
    print(f"-- pdf: wrote {os.path.join('paper', 'main.pdf')}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Run the whole PI-NCA study and produce every paper artifact.")
    ap.add_argument("--profile", default="paper", choices=list(PROFILES))
    ap.add_argument("--estimate", action="store_true",
                    help="measure two real trainings and project the cost, then exit")
    ap.add_argument("--list-stages", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated stage names")
    ap.add_argument("--skip", default=None, help="comma-separated stage names")
    ap.add_argument("--force", action="store_true",
                    help="recompute finished cells and finished stages")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="development only; the numbers are not comparable to a GPU run")
    ap.add_argument("--no-gate", action="store_true", help="skip the test suite")
    args = ap.parse_args()

    if args.list_stages:
        print("Stage names accepted by --only and --skip, in execution order:\n")
        for n, d in STAGES:
            print(f"  {n:<12} {d}")
        print("\nThe last four always run, even after a failure, so the paper artifacts\n"
              "on disk always reflect what was actually measured.")
        return

    env.configure_memory()
    P = PROFILES[args.profile]

    if args.estimate:
        env.require_gpu("estimate", allow_cpu=args.allow_cpu)
        estimate(args.profile, args.allow_cpu)
        return

    # Fail here, before anything expensive, rather than three hours in.
    env.require_gpu("runner", allow_cpu=args.allow_cpu)

    r = Run(allow_cpu=args.allow_cpu, force=args.force, grid=P["grid"])
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    if args.profile == "bench":
        skip |= {"capture", "figures"}
    known = {n for n, _ in STAGES}
    for s in (only or set()) | skip:
        if s not in known:
            raise SystemExit(f"unknown stage {s!r}. Run --list-stages for the list.")

    def want(name):
        if name in OPT_IN:
            return bool(only) and name in only        # opt-in: never runs by default
        return (only is None or name in only) and name not in skip

    def res(*names):
        return [os.path.join(RES, n) for n in names]

    t_start = time.time()
    fatal_failure = None

    try:
        if want("gate") and not args.no_gate:
            r.stage("gate: correctness tests", None, fatal=True,
                    raw=[sys.executable, "-u", "-m", "pytest", "tests/", "-q"])

        arch_arg = ["--archs", P["archs"]] if P.get("archs") else []

        if want("bench2d"):
            r.stage("2-D matrix: every architecture x every phenomenon, + ablations",
                    "bench_all",
                    ["--group", "all", "--seeds", P["seeds"], "--epochs", P["epochs"],
                     "--grid", P["grid"], "--batch", P["batch"], "--rollout", P["rollout"],
                     "--eval", P["eval"]] + arch_arg, fatal=True)

        if want("headline"):
            # Same matrix, more seeds, on the three regime representatives. This is the
            # table the paired statistics and the headline sentences come from; the
            # 5-seed full matrix is the breadth result.
            r.stage(f"headline: {P['headline_seeds']} seeds on {HEADLINE_PDES}",
                    "bench_all",
                    ["--group", "all", "--seeds", P["headline_seeds"],
                     "--epochs", P["epochs"], "--grid", P["grid"], "--batch", P["batch"],
                     "--rollout", P["rollout"], "--eval", P["eval"],
                     "--pdes", HEADLINE_PDES, "--tag", "headline"] + arch_arg,
                    fatal=False)

        if want("teacher"):
            # Cheap, and it decides whether any ranking is meaningful at all, so it runs
            # early rather than as an afterthought.
            r.stage("teacher error: what the distillation target itself gets wrong",
                    "teacher_error", ["--grid", P["grid"], "--steps", P["eval"]],
                    fatal=False, outputs=res("teacher_error.json"))

        if want("ood"):
            for pde in HEADLINE_PDES.split(","):
                r.stage(f"OOD generalisation: {pde}", "ood",
                        ["--pde", pde, "--archs", P["ood_archs"], "--grid", P["grid"],
                         "--epochs", P["ood_epochs"], "--eval", P["eval"],
                         "--n-eval", P["n_eval"], "--batch", P["batch"],
                         "--seeds", min(3, P["seeds"])],
                        fatal=False, outputs=res(f"ood_{pde}.json"))

        if want("stability"):
            for pde in HEADLINE_PDES.split(","):
                r.stage(f"stability stress (guard OFF): {pde}", "stability",
                        ["--pde", pde, "--archs", P["stab_archs"], "--grid", P["grid"],
                         "--epochs", P["stab_epochs"], "--eval", P["eval"],
                         "--n-ic", P["n_eval"], "--batch", P["batch"],
                         "--seeds", min(3, P["seeds"])],
                        fatal=False, outputs=res(f"stability_{pde}.json"))

        if want("scaling"):
            for pde in HEADLINE_PDES.split(","):
                r.stage(f"scaling: does the ranking survive scale? ({pde})", "scaling",
                        ["--pde", pde, "--archs", P["scal_archs"],
                         "--grids", P["scal_grids"], "--rollouts", P["scal_rollouts"],
                         "--epochs-sweep", P["scal_epochs"],
                         "--grid", P["grid"], "--rollout", P["rollout"],
                         "--epochs", P["epochs"], "--eval", P["eval"],
                         "--batch", P["batch"], "--n-eval", P["n_eval"]],
                        fatal=False, outputs=res(f"scaling_{pde}.json"))

        if want("jepa"):
            # A latent training objective cannot be scored by its own loss against the
            # field-space errors in every other table, so this stage freezes the
            # pretrained encoder, fits only the decoder, and evaluates in field space
            # through the same harness. It also carries its own floor: a deliberately
            # constant encoder, which bounds what a collapsed representation scores.
            for pde in HEADLINE_PDES.split(","):
                r.stage(f"JEPA latent pretraining vs distillation: {pde}", "jepa",
                        ["--pde", pde, "--grid", P["grid"], "--epochs", P["epochs"],
                         "--jepa-epochs", P["jepa_epochs"],
                         "--probe-epochs", P["probe_epochs"],
                         "--batch", P["batch"], "--rollout", P["rollout"],
                         "--eval", P["eval"], "--n-eval", P["n_eval"],
                         "--seeds", min(3, P["seeds"])],
                        fatal=False, outputs=res(f"jepa_{pde}.json"))

        if want("jepa_sweep"):
            # Screens several latent world models against the two controls that decide
            # whether any of them has promise: the distillation control of the same
            # architecture (does pretraining pay?) and the collapse floor of the same
            # architecture (did the representation learn anything?). Resumable per
            # variant, because this is the stage most likely to be interrupted.
            for pde in HEADLINE_PDES.split(","):
                r.stage(f"latent world-model variant screen: {pde}", "jepa",
                        ["--sweep", "--pde", pde, "--grid", P["sweep_grid"],
                         "--epochs", P["sweep_epochs"],
                         "--jepa-epochs", P["sweep_jepa_epochs"],
                         "--probe-epochs", P["sweep_probe_epochs"],
                         "--batch", P["batch"], "--rollout", P["rollout"],
                         "--eval", P["eval"], "--n-eval", P["n_eval"],
                         "--seeds", P["sweep_seeds"]]
                        + (["--variants", P["sweep_variants"]] if P.get("sweep_variants") else []),
                        fatal=False, outputs=res(f"jepa_sweep_{pde}.json"))

        if want("bench3d"):
            # Non-fatal on purpose: a 3-D out-of-memory error must not discard a completed
            # 2-D sweep that may have taken most of a night.
            r.stage("3-D matrix: every architecture x every phenomenon", "bench3d",
                    ["--grid", P["grid3d"], "--epochs", P["epochs3d"],
                     "--batch", P["batch3d"]], fatal=False)

        if want("resolution"):
            res_arch = ["--archs", P["res_archs"]] if P.get("res_archs") else []
            r.stage("resolution transfer study", "res_study",
                    ["--pdes", "heat,allen_cahn,navier_stokes",
                     "--epochs", P["res_epochs"]] + res_arch, fatal=False,
                    outputs=res("bench_resolution_heat.json",
                                "bench_resolution_allen_cahn.json",
                                "bench_resolution_navier_stokes.json"))

        # Plot as soon as measurements exist, so a later crash still leaves figures.
        if want("plots"):
            r.stage("benchmark plots (interim)", "plots", fatal=False)

        if want("baselines"):
            # These three hardcoded their own scale until now, which made the smoke
            # profile spend minutes on a PINN and gave a paper run no way to buy it a
            # bigger budget. Their existing config fields are simply exposed.
            r.stage("baseline: PINN (heat)", "pinn_heat",
                    ["--grid", P["pinn_grid"], "--steps", P["eval"],
                     "--iters", P["pinn_iters"]], fatal=False)
            r.stage("baseline: DeepONet (heat)", "deeponet_heat",
                    ["--seeds", P["deeponet_seeds"]], fatal=False)
            r.stage("baseline: Darcy (steady operator)", "darcy",
                    ["--seeds", P["darcy_seeds"], "--iters", P["darcy_iters"],
                     "--n-train", P["darcy_ntrain"]], fatal=False)
            # The only PINN-vs-emulator comparison that is well posed: same PDE, same
            # ICs, same horizon, same metric, with both cost structures reported.
            r.stage("matched PINN vs emulator (same task, both cost structures)",
                    "matched",
                    ["--pde", "heat", "--k", P["n_eval"], "--grid", P["pinn_grid"],
                     "--steps", P["eval"], "--epochs", P["epochs"],
                     "--pinn-iters", P["pinn_iters"]],
                    fatal=False, outputs=res("matched_heat.json"))

        if want("capture"):
            r.stage("capture trajectories for figures", "capture",
                    ["--dims", "both", "--grid", P["viz_grid"],
                     "--epochs", P["viz_epochs"], "--grid3d", P["viz3d_grid"],
                     "--epochs3d", P["viz3d_epochs"], "--max-mb", P["max_mb"]],
                    fatal=False)

        if want("figures"):
            traj = os.path.join(RES, "traj")
            for pde in VIZ_2D:
                f = os.path.join(traj, f"{pde}_2d.npz")
                if os.path.exists(f):
                    r.stage(f"figure: {pde} (2-D)", "viz", ["--npz", f], fatal=False)
            for pde in VIZ_3D:
                f = os.path.join(traj, f"{pde}_3d.npz")
                if os.path.exists(f):
                    r.stage(f"figure: {pde} (3-D slice)", "viz3d", ["--npz", f],
                            fatal=False)
                    r.stage(f"figure: {pde} (3-D volume)", "viz3d_volume", ["--npz", f],
                            fatal=False)
    except FatalStage as exc:
        fatal_failure = str(exc)

    # ---- Finalisation. Always runs, so the artifacts on disk match what was measured. --
    print(f"\n{'#' * 72}\n# finalising: audit, tables, report"
          + ("  (after a fatal failure -- these describe what DID complete)"
             if fatal_failure else "") + f"\n{'#' * 72}")

    if want("plots"):
        r.stage("benchmark plots (final)", "plots", fatal=False)
    if want("claims"):
        r.stage("claims audit: prose vs measured inventory", "claims", fatal=False)
        r.stage("bibliography: re-verify every citation against arXiv", "bib",
                fatal=False)
        r.stage("paper: regenerate every table and every quoted number", "paper",
                fatal=False)
    if want("report"):
        r.stage("architecture diagrams", "arch_figs", fatal=False)
        r.stage("report: regenerate Markdown from results", "report", fatal=False)
        r.stage("report: render PDF", "md2pdf", [REPORT_MD], fatal=False)
    if want("pdf"):
        build_pdf(r)

    r.write_manifest()
    total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"RUN {'INCOMPLETE' if fatal_failure else 'COMPLETE'} in "
          f"{total / 3600:.2f} h ({total / 60:.0f} min)")
    print(f"  backend        {env.provenance('runner')['backend']}  "
          f"peak {env.peak_mem_mb():.0f} MB")
    print("\n  PAPER ARTIFACTS")
    print("    paper/main.tex + paper/generated/*.tex   every table and quoted number")
    print("    paper/main.pdf                           if a LaTeX toolchain was present")
    print("    docs/claims_audit.md                     what the results actually support")
    print("    docs/bibliography.md                     every citation verified vs arXiv")
    print("\n  SUPPORTING")
    print("    results/*.md                             human-readable tables")
    print("    results/*.json                           raw, with device + config stamps")
    print("    docs/figures/bench/*.png                 benchmark plots")
    print("    docs/figures/*.png                       field montages and 3-D renders")
    print("    docs/PI-NCA_Architectures_and_Results.{md,pdf}")
    print("    results/run_manifest.json                per-stage timings and failures")
    if r.skipped:
        print(f"\n  {len(r.skipped)} stage(s) skipped as already complete "
              f"(--force to redo): {', '.join(r.skipped[:6])}"
              + (" ..." if len(r.skipped) > 6 else ""))
    if r.failures:
        print(f"\n  {len(r.failures)} stage(s) failed:")
        for f in r.failures:
            print(f"    - {f}")
    else:
        print("\n  No failures.")
    _summarise_cells()
    print("\n  Next: read docs/claims_audit.md. It is generated from results/ and is the\n"
          "  authority on which claims the run actually supports.")
    if fatal_failure:
        print(f"\n  A required stage failed ({fatal_failure}). Fix it and re-run the same\n"
              "  command: finished cells and stages are skipped automatically.")
        sys.exit(1)


def _summarise_cells():
    """Count completed vs failed matrix cells across every results file."""
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
