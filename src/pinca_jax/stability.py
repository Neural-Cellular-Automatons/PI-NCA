"""Long-horizon stability: failure rates, robustness, and sensitivity -- guard OFF.

`EmuConfig.safety_factor` clamps rollouts to the teacher's physical range so a diverging
model cannot emit meaningless metrics (a negative PSNR). That guard is correct for the
accuracy tables and wrong for this one: it converts a blow-up into a saturated field and
a merely-bad number, which would let the paper report stability it did not have. Every
measurement in this module therefore runs with `safety_factor=0.0` and `output_clip=None`
and reports what actually happens.

Four axes:

* **Failure rate.** Fraction of initial conditions whose rollout goes non-finite, or whose
  amplitude exceeds a multiple of the teacher's physical range, as a function of horizon
  out to several times the training horizon. Reported as a curve, and as the horizon at
  which the first failure occurs.
* **Perturbation robustness.** The rollout is restarted from a perturbed initial condition
  and, separately, perturbed mid-rollout. The growth of the induced difference relative to
  the perturbation size is a discrete Lyapunov-style amplification factor. A surrogate that
  amplifies a 1e-3 perturbation into an O(1) difference is not usable for the ensemble and
  data-assimilation workloads that motivate surrogates in the first place.
* **Timestep sensitivity.** The teacher is re-run at a different dt and the emulator --
  which was trained at one dt and has no notion of dt -- is scored against it. This
  quantifies how much of the learned map is the specific timestep rather than the operator.
* **Boundedness violation.** Fraction of cells that leave the teacher's physical range,
  which is the property the bounded variants are supposed to buy and which a rel-L2 column
  cannot show.

The headline number the paper needs from here is a *failure count*, not an average over
survivors: means computed only over runs that did not blow up are the standard way to
accidentally report a stable model.
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import bench, env, ic, metrics, stats
from .equations import pdes
from .harness import EmuConfig, field_bounds, train_emulator, _emu_traj
from .models import registry

RES = bench.RESULTS_DIR

# A rollout counts as failed when it goes non-finite, or when its amplitude exceeds the
# teacher's measured range by this factor. 10x is deliberately generous: the point is to
# catch blow-up, not to penalise a model that overshoots.
BLOWUP_FACTOR = 10.0


def _unguarded_cfg(pde, grid, **kw):
    """Config with every clamp disabled, so divergence is observable."""
    return EmuConfig(pde=pde, grid_size=grid, output_clip=None, safety_factor=0.0, **kw)


def _raw_traj(model, params, x0, steps):
    """Unclamped trajectory (steps, B, H, W, C). NaNs are preserved, not swallowed."""
    return _emu_traj(model, params, x0, steps, clip=None)


def failure_curve(model, params, x0, steps, bounds, probe_every=8):
    """Per-IC survival along a rollout.

    Returns the fraction of ICs that have failed by each probe point, and the step index
    at which each IC first fails (`steps` if it never does).
    """
    traj = _raw_traj(model, params, x0, steps)
    lo, hi = bounds if bounds is not None else (-jnp.inf, jnp.inf)
    span = (hi - lo) if bounds is not None else 1.0
    mid = 0.5 * (hi + lo) if bounds is not None else 0.0
    ax = tuple(range(2, traj.ndim))          # reduce space+channels, KEEP (T, B)
    peak = jnp.max(jnp.abs(traj - mid), axis=ax)                     # (T,B)
    bad = (~jnp.isfinite(peak)) | (peak > BLOWUP_FACTOR * 0.5 * span)
    bad = np.asarray(bad)
    ever = np.maximum.accumulate(bad, axis=0)                        # once failed, stays failed
    first = np.where(ever.any(axis=0), ever.argmax(axis=0), steps)
    probes = list(range(probe_every, steps + 1, probe_every))
    curve = {int(t): float(ever[t - 1].mean()) for t in probes}
    return {"failure_curve": curve,
            "failure_rate_final": float(ever[-1].mean()),
            "first_failure_step": [int(v) for v in first],
            "median_survival": float(np.median(first)),
            "n_ic": int(bad.shape[1])}


def bounded_violation(model, params, x0, steps, bounds):
    """Fraction of cells outside the teacher's physical range at the end of the rollout."""
    if bounds is None:
        return float("nan")
    y = _raw_traj(model, params, x0, steps)[-1]
    lo, hi = bounds
    out = (y < lo) | (y > hi) | (~jnp.isfinite(y))
    return float(jnp.mean(out.astype(jnp.float32)))


def perturbation_growth(model, params, x0, steps, eps=1e-3, seed=0, mid_rollout=False):
    """Amplification of a small perturbation over the rollout.

    `mid_rollout=True` injects the perturbation halfway instead of at t=0, which
    distinguishes sensitivity of the learned map from sensitivity of the initial fit.
    Returns the ratio of final to injected relative difference; >1 means the model
    amplifies, and an exponential-fit rate per step for comparison across horizons.
    """
    k = jax.random.PRNGKey(seed)
    scale = eps * (jnp.std(x0) + 1e-12)
    if not mid_rollout:
        xa, xb, run = x0, x0 + scale * jax.random.normal(k, x0.shape), steps
    else:
        half = max(1, steps // 2)
        xa = _raw_traj(model, params, x0, half)[-1]
        xb = xa + scale * jax.random.normal(k, xa.shape)
        run = steps - half
    d0 = float(jnp.linalg.norm(xb - xa) / (jnp.linalg.norm(xa) + 1e-12))
    ya = _raw_traj(model, params, xa, run)[-1]
    yb = _raw_traj(model, params, xb, run)[-1]
    if not bool(jnp.all(jnp.isfinite(ya)) and jnp.all(jnp.isfinite(yb))):
        return {"amplification": float("inf"), "rate_per_step": float("inf"),
                "d0": d0, "finite": False}
    d1 = float(jnp.linalg.norm(yb - ya) / (jnp.linalg.norm(ya) + 1e-12))
    amp = d1 / (d0 + 1e-15)
    return {"amplification": amp, "d0": d0, "d1": d1, "finite": True,
            "rate_per_step": float(np.log(max(amp, 1e-12)) / max(run, 1))}


def dt_sensitivity(model, params, x0, cfg, steps, factors=(0.5, 2.0)):
    """Score the emulator against teachers run at a different timestep.

    The emulator absorbed one dt into its weights. Re-timing the teacher and re-scoring
    measures how much of what it learned is the operator and how much is that dt. Both
    teachers are run to the SAME physical time, so the comparison is well posed.
    """
    spec = pdes.STABLE.get(cfg.pde, pdes.REGISTRY[cfg.pde])
    out = {}
    base_pred = _raw_traj(model, params, x0, steps)[-1]
    out["1.0"] = metrics.rel_l2(base_pred, pdes.rollout(spec, x0, steps))
    for f in factors:
        alt = pdes.PDESpec(spec.name, spec.channels, spec.step,
                           {**spec.params, "dt": spec.params["dt"] * f}, spec.conserves_mass)
        n = int(round(steps / f))
        if n < 1:
            continue
        # the emulator has no dt input: it takes the number of steps the TEACHER needs at
        # this dt, which is the only honest way to ask it for the same physical time.
        pred = _raw_traj(model, params, x0, n)[-1]
        out[str(f)] = metrics.rel_l2(pred, pdes.rollout(alt, x0, n))
    return out


def study(pde, archs, grid=24, epochs=150, rollout=12, train_eval=48, horizon_mult=8,
          n_ic=16, seeds=(42,), batch=16):
    """Train each architecture, then stress it with every clamp disabled."""
    C = pdes.REGISTRY[pde].channels
    bounds = field_bounds(pde, grid)
    long_h = train_eval * horizon_mult
    out = {}
    for arch in archs:
        recs = []
        for seed in seeds:
            cfg = _unguarded_cfg(pde, grid, rollout_steps=rollout, eval_steps=train_eval,
                                 epochs=epochs, batch=batch, n_eval=n_ic, seed=seed,
                                 warmup_epochs=30,
                                 preseed_steps=0 if pde == "cahn_hilliard" else 10)
            tr = train_emulator(registry.REGISTRY[arch].make(C, bounds=bounds), cfg)
            model, prm = tr["model"], tr["params"]
            x0 = ic.make_state(jax.random.PRNGKey(seed + 10_000), pde, n_ic, grid)
            r = failure_curve(model, prm, x0, long_h, bounds)
            r["bounded_violation_frac"] = bounded_violation(model, prm, x0, train_eval, bounds)
            r["perturb_t0"] = perturbation_growth(model, prm, x0, train_eval, seed=seed)
            r["perturb_mid"] = perturbation_growth(model, prm, x0, train_eval, seed=seed,
                                                   mid_rollout=True)
            r["dt_sensitivity"] = dt_sensitivity(model, prm, x0, cfg, train_eval)
            r["params"] = metrics.param_count(prm)
            recs.append(r)
        out[arch] = _fold(recs, long_h)
        o = out[arch]
        print(f"  {arch:22s} fail@{long_h} {o['failure_rate_final']:.0%}  "
              f"median-survival {o['median_survival']:.0f}/{long_h}  "
              f"perturb-amp {o['perturb_t0_amplification']:.2f}x  "
              f"out-of-range {o['bounded_violation_frac']:.2%}")
    return out


def _fold(recs, long_h):
    def m(f):
        vals = [f(r) for r in recs]
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("inf")

    curve_keys = sorted(recs[0]["failure_curve"])
    return {
        "failure_curve": {str(k): float(np.mean([r["failure_curve"][k] for r in recs]))
                          for k in curve_keys},
        "failure_rate_final": m(lambda r: r["failure_rate_final"]),
        "median_survival": m(lambda r: r["median_survival"]),
        "survival_summary": stats.summarize(
            [v for r in recs for v in r["first_failure_step"]]).as_dict(),
        "bounded_violation_frac": m(lambda r: r["bounded_violation_frac"]),
        "perturb_t0_amplification": m(lambda r: r["perturb_t0"]["amplification"]),
        "perturb_mid_amplification": m(lambda r: r["perturb_mid"]["amplification"]),
        "perturb_t0_diverged": float(np.mean([not r["perturb_t0"]["finite"] for r in recs])),
        "dt_sensitivity": {k: float(np.mean([r["dt_sensitivity"].get(k, np.nan) for r in recs]))
                           for k in recs[0]["dt_sensitivity"]},
        "params": int(recs[0]["params"]), "n_seeds": len(recs), "horizon": long_h,
    }


def to_markdown(pde, out, note=""):
    lines = [f"### Stability under stress - {pde} {note}", "",
             "Divergence guard **disabled** (`safety_factor=0`, no `output_clip`): a "
             "rollout that blows up is counted as a failure rather than clamped into a "
             "plausible-looking number. A run is failed once it is non-finite or its "
             f"amplitude exceeds {BLOWUP_FACTOR:g}x the teacher's physical range.", "",
             "| architecture | params | fail rate @ horizon | median survival (steps) | "
             "out-of-range cells | perturb amp (t=0) | perturb amp (mid) | "
             "rel-L2 @ dt/2 | rel-L2 @ 2dt |",
             "|---|---|---|---|---|---|---|---|---|"]
    for a, r in out.items():
        d = r["dt_sensitivity"]
        lines.append(
            f"| {a} | {r['params']} | {r['failure_rate_final']:.0%} | "
            f"{r['median_survival']:.0f}/{r['horizon']} | "
            f"{r['bounded_violation_frac']:.2%} | "
            f"{r['perturb_t0_amplification']:.3g}x | {r['perturb_mid_amplification']:.3g}x | "
            f"{d.get('0.5', float('nan')):.3g} | {d.get('2.0', float('nan')):.3g} |")
    lines += ["", "**Failure rate vs horizon** (fraction of initial conditions failed).", "",
              "| architecture | " + " | ".join(sorted(out[list(out)[0]]["failure_curve"],
                                                      key=int)) + " |",
              "|" + "---|" * (len(out[list(out)[0]]["failure_curve"]) + 1)]
    for a, r in out.items():
        cells = [f"{r['failure_curve'][k]:.0%}" for k in sorted(r["failure_curve"], key=int)]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", "`rel-L2 @ dt/2` and `@ 2dt` re-score the emulator against a teacher run "
                  "at a different timestep to the same physical time. The emulator has no dt "
                  "input, so a large change here means the learned map encodes the training "
                  "timestep rather than the operator."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--archs",
                    default="plain_nca,pi_nca,multiscale_flux_nca,bounded_multiscale_nca,fno,resnet,unet")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval", type=int, default=48, help="training-time eval horizon")
    ap.add_argument("--horizon-mult", type=int, default=8, help="stress horizon = eval x this")
    ap.add_argument("--n-ic", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("stability", allow_cpu=args.allow_cpu)
    seeds = (42,) if args.seeds <= 1 else tuple(range(args.seeds))
    archs = [a for a in args.archs.split(",") if a in registry.REGISTRY]
    print(f"[stability] {args.pde}: {len(archs)} archs, horizon "
          f"{args.eval * args.horizon_mult} steps, guard OFF")
    out = study(args.pde, archs, grid=args.grid, epochs=args.epochs,
                train_eval=args.eval, horizon_mult=args.horizon_mult,
                n_ic=args.n_ic, seeds=seeds)
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, f"stability_{args.pde}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"pde": args.pde, "seeds": list(seeds), "grid": args.grid,
                   "horizon": args.eval * args.horizon_mult, "blowup_factor": BLOWUP_FACTOR,
                   "results": out, "device": env.provenance("stability")}, f, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(args.pde, out,
                            f"(grid={args.grid}, seeds={list(seeds)}, n_ic={args.n_ic})"))
    print(f"[stability] wrote {base}.json / .md")


if __name__ == "__main__":
    main()
