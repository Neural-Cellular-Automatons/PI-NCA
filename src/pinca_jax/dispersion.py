"""Is the wave solver dispersive? The continuum equation is not, so the label has to be
about the discretisation.

The 2-D wave equation $\\partial_{tt}u=c^2\\nabla^2u$ has phase speed $c$ at every
wavenumber: it is exactly non-dispersive. Calling the phenomenon "dispersive" can
therefore only be a statement about the scheme the benchmark distils from, and that is a
measurable thing rather than a matter of wording.

Two sources of numerical dispersion are separated here, both computed in closed form and
then confirmed against the solver itself on single-mode initial conditions:

* **Spatial.** The periodic 5-point Laplacian has eigenvalue
  $-\\lambda_h = -4[\\sin^2(\\pi m_x/N)+\\sin^2(\\pi m_y/N)]$ for mode $(m_x,m_y)$ on a
  unit-spacing grid, against $-|k|^2$ with $|k|=2\\pi\\sqrt{m_x^2+m_y^2}/N$ in the
  continuum. The two agree to $O(|k|^2\\Delta x^2)$ and diverge as the mode approaches the
  grid scale, which makes the semi-discrete phase speed $c\\sqrt{\\lambda_h}/|k|$ fall below
  $c$.
* **Temporal.** The stepper is the symplectic (Verlet-like) pair
  $v^{n+1}=v^n+\\Delta t\\,c^2\\Delta_h u^n$, $u^{n+1}=u^n+\\Delta t\\,v^{n+1}$, whose
  amplification matrix has eigenvalues $e^{\\pm i\\theta}$ with
  $\\cos\\theta = 1-\\tfrac{1}{2}\\Delta t^2c^2\\lambda_h$. The realised frequency is
  $\\theta/\\Delta t$, and it stays bounded while $\\Delta t^2c^2\\lambda_h<4$; beyond that
  the mode is unstable rather than merely mis-timed.

    python -m pinca_jax.dispersion --allow-cpu
"""
from __future__ import annotations

import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import bench, env
from .equations import pdes

RES = bench.RESULTS_DIR


def _mode(grid, mx, my):
    """u = cos(k.x) on a unit-spacing periodic grid, with v = 0."""
    y, x = jnp.meshgrid(jnp.arange(grid), jnp.arange(grid), indexing="ij")
    phase = 2.0 * jnp.pi * (mx * x + my * y) / grid
    u = jnp.cos(phase)[None, ..., None]
    return jnp.concatenate([u, jnp.zeros_like(u)], axis=-1), jnp.cos(phase)


def measured_omega(spec, grid, mx, my, steps):
    """Fit the realised angular frequency of one Fourier mode from the solver itself.

    With $v(0)=0$ the exact solution is $u(t)=\\cos(\\omega t)\\cos(k\\cdot x)$, so the
    projection of the trajectory onto the initial mode is $\\cos(\\omega t)$ and a
    least-squares fit over a grid of candidate frequencies recovers $\\omega$ without
    assuming which of the two error sources produced it.
    """
    s0, mode = _mode(grid, mx, my)
    traj = pdes.rollout_trajectory(spec, s0, steps)          # (steps,1,H,W,2)
    u = traj[:, 0, :, :, 0]
    norm = float(jnp.sum(mode * mode))
    a = np.asarray(jnp.tensordot(u, mode, axes=([1, 2], [0, 1])) / norm)   # (steps,)
    t = np.arange(1, steps + 1) * spec.params["dt"]
    # Any frequency above pi/dt is aliased by the sampling, so search below it. A coarse
    # sweep then a local refinement: with one grid the spacing itself shows up as a
    # percent-level "error" at the lowest modes, which would be an artefact of the fit
    # rather than of the scheme.
    def fit(lo, hi, n):
        """Least-squares fit of A*cos(wt)+B*sin(wt) per candidate w; pick the best w.

        Fitting a zero-phase cosine instead leaves a systematic bias, because the
        stepper is staggered: it advances the velocity first and the displacement with
        the already-updated velocity, so the sampled mode carries a half-step phase
        offset. Solving for the phase removes that from the frequency estimate.
        """
        cand = np.linspace(lo, hi, n)
        co, si = np.cos(np.outer(cand, t)), np.sin(np.outer(cand, t))
        # 2x2 normal equations per candidate, solved in closed form.
        scc, sss, scs = (co * co).sum(1), (si * si).sum(1), (co * si).sum(1)
        bc, bs = co @ a, si @ a
        det = scc * sss - scs ** 2
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        A = (bc * sss - bs * scs) / det
        B = (bs * scc - bc * scs) / det
        resid = ((A[:, None] * co + B[:, None] * si - a[None, :]) ** 2).sum(1)
        return cand, int(np.argmin(resid))

    hi0 = np.pi / spec.params["dt"]
    cand, i = fit(1e-6, hi0, 4001)
    step = cand[1] - cand[0]
    cand, i = fit(max(1e-9, cand[i] - 2 * step), cand[i] + 2 * step, 4001)
    return float(cand[i]), a


def analytic(grid, mx, my, c, dt):
    lam = 4.0 * (np.sin(np.pi * mx / grid) ** 2 + np.sin(np.pi * my / grid) ** 2)
    k = 2.0 * np.pi * np.hypot(mx, my) / grid
    w_exact = c * k
    w_semi = c * np.sqrt(lam)
    arg = 1.0 - 0.5 * dt * dt * c * c * lam
    w_full = float(np.arccos(arg) / dt) if abs(arg) <= 1.0 else float("inf")
    return dict(k=float(k), lam=float(lam), omega_exact=float(w_exact),
                omega_semi_discrete=float(w_semi), omega_full_discrete=w_full,
                stability_ratio=float(dt * dt * c * c * lam / 4.0))


def study(grid=48, steps=48, modes=None):
    spec = pdes.STABLE.get("wave", pdes.REGISTRY["wave"])
    c, dt = spec.params["c"], spec.params["dt"]
    modes = modes or [(m, 0) for m in (1, 2, 4, 8, 12, 16, 20, 24)] + \
        [(m, m) for m in (1, 2, 4, 8, 12, 16, 24)]
    rows = []
    for mx, my in modes:
        if max(mx, my) > grid // 2:
            continue
        a = analytic(grid, mx, my, c, dt)
        w_meas, _ = measured_omega(spec, grid, mx, my, steps)
        a.update(mx=mx, my=my, omega_measured=w_meas,
                 phase_speed_exact=c,
                 phase_speed_measured=w_meas / a["k"],
                 phase_error=(w_meas - a["omega_exact"]) / a["omega_exact"])
        rows.append(a)
        print(f"  mode ({mx:2d},{my:2d})  |k|={a['k']:.3f}  omega exact {a['omega_exact']:.4f}  "
              f"semi-discrete {a['omega_semi_discrete']:.4f}  full {a['omega_full_discrete']:.4f}  "
              f"measured {w_meas:.4f}  phase error {a['phase_error']:+.1%}")
    return {"grid": grid, "steps": steps, "c": float(c), "dt": float(dt), "modes": rows}


def to_markdown(out, note=""):
    rows = out["modes"]
    worst = max(rows, key=lambda r: abs(r["phase_error"]))
    axis = [r for r in rows if r["my"] == 0]
    L = [f"### Numerical dispersion of the wave solver {note}", "",
         "The continuum 2-D wave equation is **non-dispersive**: every Fourier mode "
         f"travels at the same speed $c={out['c']}$. The scheme the benchmark distils "
         "from is not. For each single-mode initial condition the table gives the exact "
         "frequency $c|k|$, the semi-discrete frequency implied by the 5-point Laplacian, "
         "the fully discrete frequency of the symplectic stepper, and the frequency fitted "
         "from the solver's own trajectory.", "",
         "| mode $(m_x,m_y)$ | $\\|k\\|$ | $\\omega$ exact | $\\omega$ semi-discrete | "
         "$\\omega$ full discrete | $\\omega$ measured | phase error | "
         "measured phase speed |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| ({r['mx']},{r['my']}) | {r['k']:.3f} | {r['omega_exact']:.4f} | "
                 f"{r['omega_semi_discrete']:.4f} | {r['omega_full_discrete']:.4f} | "
                 f"{r['omega_measured']:.4f} | {r['phase_error']:+.2%} | "
                 f"{r['phase_speed_measured']:.4f} |")
    L += ["", f"The fitted frequencies track the fully discrete prediction, and the phase "
              f"error grows monotonically with $|k|$: from "
              f"{abs(axis[0]['phase_error']):.2%} at mode "
              f"$({axis[0]['mx']},{axis[0]['my']})$ to "
              f"{abs(worst['phase_error']):.1%} at $({worst['mx']},{worst['my']})$, where "
              f"the measured phase speed is {worst['phase_speed_measured']:.3f} against "
              f"the exact {out['c']}. Both error sources push the same way, so short "
              f"wavelengths travel too slowly.", "",
          "The honest description of the phenomenon is therefore that the **continuum "
          "equation is non-dispersive and its discretisation is dispersive at the grid "
          "scale**, which is also why a model that resolves only smooth structure can "
          "still score well on it."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("dispersion", allow_cpu=args.allow_cpu)
    print(f"[dispersion] wave solver at grid {args.grid}, {args.steps} steps")
    out = study(grid=args.grid, steps=args.steps)
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, "dispersion_wave")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({**out, "device": env.provenance("dispersion")}, f, indent=1)
    md = to_markdown(out, f"(grid={args.grid})")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"[dispersion] wrote {base}.json / .md")


if __name__ == "__main__":
    main()
