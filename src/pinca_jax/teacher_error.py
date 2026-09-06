"""How wrong is the teacher? Distillation error in units of solver error.

The emulators here are trained to imitate a numerical solver and then scored against
that same solver, which invites the obvious objection: the target is not ground truth,
so a small rel-L2 could mean "accurate physics" or merely "faithfully copied a bad
approximation". This module answers it with numbers instead of prose.

Three measurements:

1. **Absolute solver error, against an exact reference.** For the linear
   constant-coefficient members of the suite (heat, advection-diffusion, wave) the
   periodic problem is diagonal in Fourier space, so the continuum solution is available
   in closed form. `analytic_solution` evaluates it and `solver_vs_analytic` reports what
   the teacher's own rel-L2 is after the same horizon the emulators are scored over.
   The error is reported split: `rel_l2_spatial` is the 5-point stencil's own truncation
   error (a floor no timestep can cross), `rel_l2_temporal` is the forward-Euler error
   against the exactly-integrated semi-discrete system, and `rel_l2` is the total.

2. **Self-convergence, for the rest.** Gray-Scott, Cahn-Hilliard, shallow water,
   FitzHugh-Nagumo and Navier-Stokes have no closed form here, so the teacher is refined
   in time (dt, dt/2, dt/4 over the same physical interval) and the Cauchy difference
   between successive refinements bounds its own error. The observed order of accuracy
   is reported alongside; a scheme that does not show its design order is itself the
   finding.

3. **Cost.** Wall-clock per step for the solver and for an emulator at matched grid and
   batch, so "the surrogate is faster" is a measured ratio rather than an assumption.
   Reported next to the accuracy so the reader can see the actual trade.

The decision rule this supports: an emulator whose error is far ABOVE the teacher's own
error is limited by learning, and the distillation target is not the binding constraint.
An emulator whose error approaches the teacher's error has saturated the target, and any
further comparison between architectures at that point is measuring noise in the teacher
rather than a difference between models. Every headline accuracy number in the paper is
reported with the corresponding teacher error so the reader can tell which case applies.
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import bench, env, ic, metrics
from .equations import pdes

RES = bench.RESULTS_DIR

# Phenomena that are linear and constant-coefficient on the periodic grid, hence exactly
# solvable in Fourier space. dx = 1 throughout the package, so the domain is [0,N)^2 and
# the physical wavenumber of index k is 2*pi*k/N.
ANALYTIC = ("heat", "adv_diff", "wave")


def _wavenumbers(n):
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n)          # fftfreq gives k/N, so this is 2*pi*k/N
    return k


def analytic_solution(pde: str, u0: jax.Array, t: float, discrete: bool = False):
    """Exact solution at time `t` on the periodic domain, via the PDE symbol.

    `discrete=False` gives the exact solution of the **continuum PDE**: the gap to the
    solver is the total discretisation error, space and time together.

    `discrete=True` gives the exact solution of the **spatially discretised** ODE system
    (the 5-point Laplacian's own eigenvalues, and the central difference's own symbol):
    the gap to the solver is then the *temporal* error alone.

    Both are needed. Refining dt converges the solver to the discrete reference, not to
    the continuum one -- the spatial truncation error is a floor that no timestep can
    cross -- and reporting only the continuum gap makes that floor look like a failure
    to converge. `run` reports the split.
    """
    p = pdes.REGISTRY[pde].params
    n = u0.shape[1]
    ky, kx = jnp.meshgrid(_wavenumbers(n), _wavenumbers(n), indexing="ij")
    if discrete:
        # symbols of the actual stencils in equations/operators.py
        lap = 2.0 * jnp.cos(ky) + 2.0 * jnp.cos(kx) - 4.0     # 5-point Laplacian
        dx_sym, dy_sym = 1j * jnp.sin(kx), 1j * jnp.sin(ky)   # central differences
    else:
        lap = -(kx ** 2 + ky ** 2)                            # continuum Laplacian
        dx_sym, dy_sym = 1j * kx, 1j * ky

    if pde == "heat":
        return _apply_symbol(u0, jnp.exp(p["alpha"] * lap * t))
    if pde == "adv_diff":
        return _apply_symbol(u0, jnp.exp((p["D"] * lap - p["vx"] * dx_sym - p["vy"] * dy_sym) * t))
    if pde == "wave":
        # u_tt = c^2 L u started from (u0, v0=0)  ->  u = cos(c sqrt(-L) t) u0
        w = p["c"] * jnp.sqrt(jnp.maximum(-lap, 0.0))
        u = _apply_symbol(u0[..., 0:1], jnp.cos(w * t) + 0j)
        v = _apply_symbol(u0[..., 0:1], -w * jnp.sin(w * t) + 0j)
        return jnp.concatenate([u, v], axis=-1)
    raise KeyError(f"no closed form for {pde!r}")


def _apply_symbol(u, mult):
    """Multiply each Fourier mode of an NHWC field by `mult` (broadcast over B and C)."""
    f = jnp.fft.fft2(u, axes=(1, 2))
    return jnp.real(jnp.fft.ifft2(f * mult[None, :, :, None], axes=(1, 2)))


def solver_vs_analytic(pde: str, grid: int, steps: int, batch: int = 8, seed: int = 0):
    """Teacher error against closed-form references, split into spatial and temporal parts."""
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    u0 = ic.make_state(jax.random.PRNGKey(seed), pde, batch, grid)
    t = spec.params["dt"] * steps
    num = pdes.rollout(spec, u0, steps)
    exact = analytic_solution(pde, u0, t, discrete=False)
    semi = analytic_solution(pde, u0, t, discrete=True)
    return {"rel_l2": metrics.rel_l2(num, exact),          # total discretisation error
            "rel_l2_temporal": metrics.rel_l2(num, semi),  # vs the semi-discrete exact ODE
            "rel_l2_spatial": metrics.rel_l2(semi, exact), # stencil error alone
            "per_ic_rel_l2": metrics.rel_l2_per_sample(num, exact),
            "max_abs_err": metrics.max_abs_error(num, exact),
            "t_final": float(t), "steps": steps, "grid": grid, "kind": "vs_analytic"}


def self_convergence(pde: str, grid: int, steps: int, batch: int = 8, seed: int = 0,
                     refinements=(1, 2, 4, 8)):
    """Refine dt at fixed physical time; Cauchy differences bound the teacher's error.

    Returns the successive differences and the observed order of accuracy estimated from
    the last three refinements, p = log2(|u_1 - u_2| / |u_2 - u_4|).
    """
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    u0 = ic.make_state(jax.random.PRNGKey(seed), pde, batch, grid)
    sols = {}
    for r in refinements:
        fine = pdes.PDESpec(spec.name, spec.channels, spec.step,
                            {**spec.params, "dt": spec.params["dt"] / r},
                            spec.conserves_mass)
        sols[r] = pdes.rollout(fine, u0, steps * r)
    ref = sols[refinements[-1]]
    diffs = {r: metrics.rel_l2(sols[r], ref) for r in refinements[:-1]}
    order = float("nan")
    if len(refinements) >= 3:
        a, b, c = refinements[0], refinements[1], refinements[2]
        e1 = float(jnp.linalg.norm(sols[a] - sols[b]))
        e2 = float(jnp.linalg.norm(sols[b] - sols[c]))
        if e2 > 0 and e1 > 0:
            order = float(np.log2(e1 / e2))
    finite = bool(jnp.all(jnp.isfinite(ref)))
    return {"rel_l2": diffs[refinements[0]], "cauchy_diffs": diffs,
            "observed_order": order, "reference_refinement": refinements[-1],
            "t_final": float(spec.params["dt"] * steps), "steps": steps, "grid": grid,
            "reference_finite": finite, "kind": "self_convergence"}


def reliability(rec: dict) -> str:
    """Is this teacher trustworthy enough to rank architectures against?

    A distillation target that is not converging is not a target. Three verdicts:

    * ``converging`` -- the closed form is available, or the observed order is within a
      plausible band of the scheme's design order. Rankings against it are meaningful.
    * ``round-off limited`` -- the Cauchy differences have fallen to machine precision, so
      the observed order is numerically undefined. The teacher is fine; the *order
      estimate* is not, and reporting it as a failure would be wrong.
    * ``NOT CONVERGING`` -- refining the timestep does not reduce the difference. Any
      architecture comparison on this equation is measuring the solver's own instability,
      and this module says so rather than letting a ranking be read off it.
    """
    if rec.get("kind") == "vs_analytic":
        return "converging"
    err = rec.get("rel_l2", float("nan"))
    order = rec.get("observed_order", float("nan"))
    if err < 1e-6:
        return "round-off limited"
    if order != order or not (0.5 <= order <= 2.5):
        return "NOT CONVERGING"
    return "converging"


def teacher_error(pde: str, grid: int, steps: int, batch: int = 8, seed: int = 0):
    """Best available estimate of the teacher's own error over the evaluation horizon."""
    rec = (solver_vs_analytic(pde, grid, steps, batch, seed) if pde in ANALYTIC
           else self_convergence(pde, grid, steps, batch, seed))
    rec["reliability"] = reliability(rec)
    return rec


# ------------------------------------------------------------------- cost ---
def solver_cost(pde: str, grid: int, batch: int = 8, seed: int = 0):
    """Wall-clock seconds per solver step at this grid and batch (jitted, warmed up)."""
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    u0 = ic.make_state(jax.random.PRNGKey(seed), pde, batch, grid)
    step = jax.jit(lambda s: spec.step(s, spec.params))
    return metrics.time_callable(step, u0)


def emulator_cost(pde: str, grid: int, arch: str = "plain_nca", batch: int = 8, seed: int = 0):
    """Wall-clock seconds per emulator step, matched grid and batch (untrained weights).

    Inference cost does not depend on the weights' values, so this avoids paying for a
    training run just to time a forward pass.
    """
    from .models import registry
    C = pdes.REGISTRY[pde].channels
    model = registry.REGISTRY[arch].make(C)()
    u0 = ic.make_state(jax.random.PRNGKey(seed), pde, batch, grid)
    params = model.init(jax.random.PRNGKey(seed), u0)
    fn = jax.jit(lambda x: model.apply(params, x))
    return metrics.time_callable(fn, u0)


def run(pdes_list, grid=24, steps=48, batch=8, archs=("plain_nca", "fno")):
    out = {}
    for pde in pdes_list:
        rec = teacher_error(pde, grid, steps, batch)
        rec["solver_s_per_step"] = solver_cost(pde, grid, batch)
        rec["emulator_s_per_step"] = {a: emulator_cost(pde, grid, a, batch) for a in archs}
        rec["speedup_vs_solver"] = {
            a: rec["solver_s_per_step"] / (v + 1e-12)
            for a, v in rec["emulator_s_per_step"].items()}
        out[pde] = rec
        best = max(rec["speedup_vs_solver"].values())
        print(f"  {pde:16s} teacher rel-L2 {rec['rel_l2']:.3e} ({rec['kind']})  "
              f"solver {rec['solver_s_per_step'] * 1e3:.3f} ms/step  "
              f"best emulator speedup {best:.2f}x")
    return out


def to_markdown(out, grid, steps):
    lines = [f"### Teacher (solver) error and cost - grid {grid}, horizon {steps} steps", "",
             "`vs_analytic` = rel-L2 against the closed-form periodic solution (linear "
             "constant-coefficient PDEs). `self_convergence` = rel-L2 between the "
             "production dt and a dt/8 reference over the same physical time; the "
             "observed order should match the scheme's design order.", "",
             "| PDE | estimate | teacher rel-L2 | spatial | temporal | observed order | "
             "verdict | solver ms/step | fastest emulator ms/step | emulator vs solver |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    unreliable, slower = [], []
    for pde, r in out.items():
        order = r.get("observed_order", float("nan"))
        order_s = "-" if order != order else f"{order:.2f}"
        fastest = min(r["emulator_s_per_step"], key=r["emulator_s_per_step"].get)
        sp = r.get("rel_l2_spatial"); tp = r.get("rel_l2_temporal")
        verdict = r.get("reliability", reliability(r))
        if verdict == "NOT CONVERGING":
            unreliable.append(pde)
            verdict = "**NOT CONVERGING**"
        ratio = r["speedup_vs_solver"][fastest]
        if ratio < 1.0:
            slower.append(pde)
        # The ratio is solver-time / emulator-time, so <1 means the emulator is SLOWER.
        # Printing "0.02x speedup" invites the opposite reading, so say which it is.
        cost = (f"{ratio:.2f}x faster" if ratio >= 1.0 else f"**{1 / ratio:.0f}x slower**")
        lines.append(
            f"| {pde} | {r['kind']} | {r['rel_l2']:.3e} | "
            f"{'-' if sp is None else f'{sp:.3e}'} | {'-' if tp is None else f'{tp:.3e}'} | "
            f"{order_s} | {verdict} | {r['solver_s_per_step'] * 1e3:.3f} | "
            f"{r['emulator_s_per_step'][fastest] * 1e3:.3f} ({fastest}) | {cost} |")
    lines += ["",
              "**How to read an emulator's rel-L2 against this table.** An emulator error "
              "far above the teacher error is limited by learning: the target is not the "
              "binding constraint and architecture comparisons are meaningful. An "
              "emulator error approaching the teacher error has saturated the target, and "
              "further ranking there measures the teacher's own discretisation error "
              "rather than model quality.", "",
              "`round-off limited` means the Cauchy differences reached machine precision, "
              "so the order estimate is undefined -- the teacher is fine, the estimator is "
              "not. `NOT CONVERGING` means refining the timestep did not reduce the "
              "difference at all."]
    if unreliable:
        lines += ["", "> **Teacher does not converge on: " + ", ".join(unreliable) +
                  ".** Refining the timestep does not reduce the solver's own change, so "
                  "the distillation target on these equations is not a converged solution. "
                  "Any architecture ranking on them is measuring the solver's instability "
                  "rather than model quality, and should not be reported as a result. This "
                  "is the most likely explanation wherever no architecture beats the "
                  "do-nothing identity floor."]
    if slower:
        lines += ["", "> **The reference solver is faster than every emulator on: " +
                  ", ".join(slower) + ".** At this grid size a learned surrogate is not a "
                  "deployment case on these equations; it is a measurement harness. "
                  "Surrogates pay off where the solver does not fit -- much larger grids, "
                  "much longer horizons, stiff timestep restrictions, or differentiability "
                  "through an entire rollout."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdes", default="all")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("teacher_error", allow_cpu=args.allow_cpu)
    names = list(pdes.REGISTRY) if args.pdes == "all" else args.pdes.split(",")
    out = run(names, grid=args.grid, steps=args.steps, batch=args.batch)
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, "teacher_error")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"grid": args.grid, "steps": args.steps, "batch": args.batch,
                   "results": out, "device": env.provenance("teacher_error")}, f, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(out, args.grid, args.steps))
    print(f"[teacher_error] wrote {base}.json / .md")


def demo():
    """The two closed forms must separate the two truncation errors, and dt-refinement
    must converge to the SEMI-DISCRETE reference (not the continuum one, which the
    fixed stencil can never reach)."""
    u0 = ic.make_state(jax.random.PRNGKey(0), "heat", 2, 16)
    spec = pdes.REGISTRY["heat"]
    t = spec.params["dt"] * 8
    fine_spec = pdes.PDESpec("heat", 1, spec.step, {**spec.params, "dt": spec.params["dt"] / 16}, True)
    semi = analytic_solution("heat", u0, t, discrete=True)
    coarse_t = metrics.rel_l2(pdes.rollout(spec, u0, 8), semi)
    fine_t = metrics.rel_l2(pdes.rollout(fine_spec, u0, 128), semi)
    assert fine_t < coarse_t / 5, (fine_t, coarse_t)         # temporal error must collapse
    spatial = metrics.rel_l2(semi, analytic_solution("heat", u0, t, discrete=False))
    assert spatial > fine_t, (spatial, fine_t)               # spatial error is the floor
    assert metrics.rel_l2(analytic_solution("heat", u0, 0.0), u0) < 1e-5   # t=0 identity
    for pde in ANALYTIC:                                     # every closed form is real+finite
        x = ic.make_state(jax.random.PRNGKey(1), pde, 2, 16)
        y = analytic_solution(pde, x, 0.3)
        assert y.shape == x.shape and bool(jnp.all(jnp.isfinite(y))), pde
    sc = self_convergence("cahn_hilliard", 16, 8, batch=2)
    assert sc["rel_l2"] > 0 and sc["reference_finite"]
    print(f"teacher_error.demo OK (heat temporal {coarse_t:.2e} -> {fine_t:.2e}; "
          f"spatial floor {spatial:.2e})")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
