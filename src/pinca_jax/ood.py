"""Out-of-distribution evaluation: held-out ICs, PDE coefficients, and horizons.

Random draws from the *training* initial-condition generator are not a test set. They
are more training data with a different seed, and an emulator that memorises the narrow
blob family they come from will score well on them while being useless on anything else.
This module defines the held-out axes explicitly, BEFORE any model is trained, and
evaluates a trained model on each of them:

| axis                | shift                                        | why it matters              |
|---------------------|----------------------------------------------|-----------------------------|
| amplitude           | x0.5, x2, x4 the training amplitude range     | tests scale equivariance    |
| structure count     | fewer / more blobs than training              | tests feature-count transfer|
| length scale        | narrower / wider features                     | tests spatial-frequency gap |
| spectrum            | added broadband high-frequency content        | probes the smoothing bias   |
| phase               | random spatial translation                    | must be free (periodic conv)|
| PDE coefficient     | diffusivity / viscosity scaled up and down    | tests operator, not one PDE |
| horizon             | 2x and 4x the training rollout                | tests stability, not fit    |

`phase` is included deliberately as a *negative control*: every architecture here is
translation-equivariant by construction, so a non-trivial phase gap would indicate a
bug in the evaluation rather than a property of a model.

Resolution transfer is the eighth axis and lives in `res_study.py`, which already
trains at one grid and evaluates at others.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import jax
import jax.numpy as jnp

from . import bench, env, ic, metrics, stats
from .equations import pdes
from .harness import EmuConfig, effective_clip, field_bounds, train_emulator, _emu_traj
from .models import registry

RES = bench.RESULTS_DIR

# Which PDE coefficient is the physically meaningful one to perturb per phenomenon.
COEFF = {"heat": "alpha", "adv_diff": "D", "wave": "c", "allen_cahn": "eps2",
         "cahn_hilliard": "eps2", "nagumo": "D", "navier_stokes": "nu",
         "gray_scott": "Du", "shallow_water": "g", "fitzhugh_nagumo": "Du"}


# --------------------------------------------------------------- IC shifts ---
def _rescale(x, factor, ref):
    """Scale a field's deviation from its reference level, keeping the level fixed.

    For a field whose physics lives in the departure from a background state (h=1 for
    shallow water, u=1 for Gray-Scott), scaling the raw values would move the background
    and change the problem rather than the amplitude. Scaling the deviation does not.
    """
    return ref + (x - ref) * factor


def shifted_state(key, pde, batch, size, shift: str):
    """An initial-condition batch drawn from a held-out distribution.

    Returns None when a shift is not defined for a phenomenon, so callers report
    "not applicable" rather than silently substituting the in-distribution draw.
    """
    x = ic.make_state(key, pde, batch, size)
    if shift == "in_dist":
        return x

    if shift == "phase":                       # random torus translation (control)
        k1, k2 = jax.random.split(key)
        dy = int(jax.random.randint(k1, (), 1, size))
        dx = int(jax.random.randint(k2, (), 1, size))
        return jnp.roll(jnp.roll(x, dy, axis=1), dx, axis=2)

    if shift in ("amp_half", "amp_2x", "amp_4x"):
        f = {"amp_half": 0.5, "amp_2x": 2.0, "amp_4x": 4.0}[shift]
        ref = _background(pde, x)
        y = _rescale(x, f, ref)
        return _reclip(pde, y)

    if shift in ("blobs_few", "blobs_many"):
        if pde not in ("heat", "adv_diff", "wave", "navier_stokes", "shallow_water"):
            return None                         # structure count is not a knob here
        lo, hi = (1, 2) if shift == "blobs_few" else (8, 12)
        return _blob_variant(key, pde, batch, size, n_min=lo, n_max=hi)

    if shift in ("scale_narrow", "scale_wide"):
        if pde not in ("heat", "adv_diff", "wave", "navier_stokes", "shallow_water"):
            return None
        frac = 0.04 if shift == "scale_narrow" else 0.16
        return _blob_variant(key, pde, batch, size, sigma_frac=frac)

    if shift == "spectrum_rough":               # broadband high-frequency perturbation
        k = jax.random.split(key)[1]
        amp = 0.10 * float(jnp.std(x) + 1e-6)
        return _reclip(pde, x + amp * jax.random.normal(k, x.shape))

    raise KeyError(f"unknown IC shift {shift!r}")


def _background(pde, x):
    """The field's background level, so amplitude scaling perturbs the signal only."""
    if pde in ("shallow_water", "gray_scott"):
        return jnp.mean(x, axis=(1, 2), keepdims=True)
    return 0.0


def _reclip(pde, x):
    """Keep a shifted IC inside the physical range its solver assumes."""
    if pde == "cahn_hilliard":
        return jnp.clip(x, -0.99, 0.99)
    if pde == "shallow_water":                   # positive depth
        return x.at[..., 0:1].set(jnp.clip(x[..., 0:1], min=1e-2))
    if pde == "nagumo":
        return jnp.clip(x, 0.0, 1.0)
    return x


def _blob_variant(key, pde, batch, size, **kw):
    """Rebuild a blob-based IC with the generator's knobs changed.

    Mirrors the per-PDE channel assembly in `ic.make_state`; the assembly is duplicated
    rather than parameterised so `ic.py` -- which the correctness gate pins against the
    original notebook -- stays untouched.
    """
    if pde in ("heat", "adv_diff"):
        return ic.gaussian_blobs(key, batch, size, **kw)
    if pde == "wave":
        u = ic.gaussian_blobs(key, batch, size, **{"amp_lo": 2.0, "amp_hi": 5.0,
                                                   "sigma_frac": 0.10, **kw})
        return jnp.concatenate([u, jnp.zeros_like(u)], axis=-1)
    if pde == "shallow_water":
        base = {"n_min": 2, "n_max": 4, "amp_lo": 0.15, "amp_hi": 0.5, "sigma_frac": 0.10}
        h = 1.0 + ic.gaussian_blobs(key, batch, size, **{**base, **kw})
        z = jnp.zeros_like(h)
        return jnp.concatenate([h, z, z], axis=-1)
    if pde == "navier_stokes":
        base = {"n_min": 3, "n_max": 6, "amp_lo": -3.0, "amp_hi": 3.0, "sigma_frac": 0.10}
        w = ic.gaussian_blobs(key, batch, size, **{**base, **kw})
        return w - w.mean(axis=(1, 2, 3), keepdims=True)
    return None


IC_SHIFTS = ["in_dist", "phase", "amp_half", "amp_2x", "amp_4x", "blobs_few",
             "blobs_many", "scale_narrow", "scale_wide", "spectrum_rough"]
COEFF_SHIFTS = {"coeff_half": 0.5, "coeff_2x": 2.0}


# ------------------------------------------------------------- evaluation ---
def _spec_for(pde, coeff_factor=None):
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    if coeff_factor is None:
        return spec
    name = COEFF.get(pde)
    if name is None or name not in spec.params:
        return None
    scaled = {**spec.params, name: spec.params[name] * coeff_factor}
    return pdes.PDESpec(spec.name, spec.channels, spec.step, scaled, spec.conserves_mass)


def eval_on(model, params, cfg: EmuConfig, x0, spec, steps: int):
    """rel-L2 per IC at the end of a `steps` rollout, plus divergence bookkeeping."""
    tgt = pdes.rollout_trajectory(spec, x0, steps)[-1]
    pred = _emu_traj(model, params, x0, steps, effective_clip(cfg))[-1]
    per_ic = metrics.rel_l2_per_sample(pred, tgt)
    finite = bool(jnp.all(jnp.isfinite(pred)))
    return {"per_ic_rel_l2": per_ic, "rel_l2": metrics.rel_l2(pred, tgt),
            "psnr": metrics.psnr(pred, tgt), "finite": finite,
            "diverged_frac": float(sum(v > 1.0 for v in per_ic) / max(1, len(per_ic)))}


def study(pde, archs, grid=24, epochs=150, rollout=12, eval_steps=48, batch=16,
          n_eval=16, seeds=(42,), horizons=(1, 2, 4)):
    """Train each arch in-distribution, then evaluate on every held-out axis.

    Training NEVER sees a shifted distribution: the shifts exist only at evaluation
    time, so the numbers are generalisation, not fit.
    """
    C = pdes.REGISTRY[pde].channels
    bounds = field_bounds(pde, grid)
    out = {}
    for arch in archs:
        cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=rollout,
                        eval_steps=eval_steps, epochs=epochs, batch=batch,
                        n_eval=n_eval, warmup_epochs=30,
                        preseed_steps=0 if pde == "cahn_hilliard" else 10)
        rows = {}
        for seed in seeds:
            c = EmuConfig(**{**cfg.__dict__, "seed": seed})
            tr = train_emulator(registry.REGISTRY[arch].make(C, bounds=bounds), c)
            model, prm = tr["model"], tr["params"]
            base_spec = _spec_for(pde)
            for name in IC_SHIFTS:
                key = jax.random.PRNGKey(seed + 10_000)   # same eval key across archs
                x0 = shifted_state(key, pde, n_eval, grid, name)
                if x0 is None:
                    continue
                r = eval_on(model, prm, c, x0, base_spec, eval_steps)
                rows.setdefault(name, []).append(r)
            for name, f in COEFF_SHIFTS.items():
                spec = _spec_for(pde, f)
                if spec is None:
                    continue
                x0 = ic.make_state(jax.random.PRNGKey(seed + 10_000), pde, n_eval, grid)
                r = eval_on(model, prm, c, x0, spec, eval_steps)
                rows.setdefault(name, []).append(r)
            for h in horizons:
                if h == 1:
                    continue
                x0 = ic.make_state(jax.random.PRNGKey(seed + 10_000), pde, n_eval, grid)
                r = eval_on(model, prm, c, x0, base_spec, eval_steps * h)
                rows.setdefault(f"horizon_{h}x", []).append(r)
        out[arch] = {k: _fold(v) for k, v in rows.items()}
        base = out[arch].get("in_dist", {}).get("rel_l2_mean", float("nan"))
        print(f"  {arch:22s} in-dist {base:.3e} | " +
              " ".join(f"{k}:{v['rel_l2_mean']:.2e}" for k, v in list(out[arch].items())[1:5]))
    return out


def _fold(run_list):
    """Pool per-IC values across seeds and summarise."""
    per_ic = [v for r in run_list for v in r["per_ic_rel_l2"]]
    s = stats.summarize(per_ic)
    return {"per_ic_rel_l2": per_ic, "rel_l2_mean": s.mean, "rel_l2_median": s.median,
            "ci_lo": s.ci_lo, "ci_hi": s.ci_hi, "n": s.n,
            "psnr_mean": float(sum(r["psnr"] for r in run_list) / len(run_list)),
            "diverged_frac": float(sum(r["diverged_frac"] for r in run_list) / len(run_list)),
            "all_finite": all(r["finite"] for r in run_list)}


def to_markdown(pde, out, cfg_note=""):
    axes = []
    for arch in out:
        for k in out[arch]:
            if k not in axes:
                axes.append(k)
    lines = [f"### OOD generalisation - {pde} {cfg_note}", "",
             "rel-L2 at the end of the rollout, pooled over held-out ICs "
             "(mean [95% bootstrap CI]); `div` = fraction of ICs whose rel-L2 exceeds 1 "
             "(worse than predicting nothing).", "",
             "| axis | " + " | ".join(out) + " |", "|" + "---|" * (len(out) + 1)]
    for ax in axes:
        cells = []
        for arch in out:
            r = out[arch].get(ax)
            if r is None:
                cells.append("n/a")
            else:
                d = f" div={r['diverged_frac']:.0%}" if r["diverged_frac"] > 0 else ""
                cells.append(f"{r['rel_l2_mean']:.3g} [{r['ci_lo']:.2g},{r['ci_hi']:.2g}]{d}")
        lines.append(f"| {ax} | " + " | ".join(cells) + " |")
    lines += ["", "**Degradation ratio** (axis rel-L2 / in-dist rel-L2); 1.0 = perfect transfer.", "",
              "| axis | " + " | ".join(out) + " |", "|" + "---|" * (len(out) + 1)]
    for ax in axes:
        if ax == "in_dist":
            continue
        cells = []
        for arch in out:
            r, base = out[arch].get(ax), out[arch].get("in_dist")
            cells.append("n/a" if (r is None or not base)
                         else f"{r['rel_l2_mean'] / (base['rel_l2_mean'] + 1e-12):.2f}x")
        lines.append(f"| {ax} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--archs", default="plain_nca,pi_nca,multiscale_flux_nca,fno,resnet,unet")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("ood", allow_cpu=args.allow_cpu)
    seeds = (42,) if args.seeds <= 1 else tuple(range(args.seeds))
    archs = [a for a in args.archs.split(",") if a in registry.REGISTRY]
    print(f"[ood] {args.pde}: {len(archs)} archs x {len(IC_SHIFTS) + len(COEFF_SHIFTS) + 2} axes")
    out = study(args.pde, archs, grid=args.grid, epochs=args.epochs,
                eval_steps=args.eval, n_eval=args.n_eval, seeds=seeds)
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, f"ood_{args.pde}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"pde": args.pde, "seeds": list(seeds), "grid": args.grid,
                   "epochs": args.epochs, "eval_steps": args.eval,
                   "results": out, "device": env.provenance("ood")}, f, indent=1)
    note = f"(grid={args.grid}, eval_steps={args.eval}, seeds={list(seeds)})"
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(args.pde, out, note))
    print(f"[ood] wrote {base}.json / .md")


if __name__ == "__main__":
    main()
