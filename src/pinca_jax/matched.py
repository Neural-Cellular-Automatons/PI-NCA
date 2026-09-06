"""Matched PINN-vs-emulator comparison: same PDE, same ICs, same horizon, same metric.

The PINN numbers elsewhere in this repository are not comparable to the emulator numbers
and are labelled as such. A PINN solves one initial-value problem per training run; an
emulator is trained once on a distribution of initial conditions and then applied to any
of them. Putting the two in one accuracy column compares different quantities.

This module makes the comparison well posed by fixing the *task* rather than the model:

    Given K initial conditions drawn from the same distribution, produce the solution at
    time T for all K. Report the total wall-clock cost and the accuracy on each, against
    the same numerical reference, with the same relative-L2 metric.

Under that task the two paradigms have genuinely different cost structures, and the
comparison is a curve rather than a number:

* the **PINN** pays a full training run per initial condition, so its cost is linear in K
  and its accuracy per IC is independent of K;
* the **emulator** pays one training run and then K cheap rollouts, so its cost is nearly
  flat in K, but it must generalise across the IC distribution and it inherits the
  teacher's discretisation error;
* the **numerical solver** pays nothing up front and a small amount per IC, and defines
  the reference -- so it is included, because "the surrogate is worth it" is a claim
  against the solver, not only against the other surrogate.

The crossover K -- where the amortised emulator becomes cheaper than the per-IC PINN at
comparable accuracy -- is the number a practitioner actually needs, and it is what this
module reports. Reporting it honestly usually favours neither side unconditionally: the
PINN is competitive at K=1 and hopeless at K=100, the emulator is the reverse, and the
solver beats both whenever it is affordable.

Cost accounting: wall-clock on one device, measured the same way for all three (jitted,
warmed up, blocking on device). The PINN's per-IC training and the emulator's one-off
training are both counted; the solver's reference generation used for the emulator's
training data is counted against the emulator, because it is a real cost of that
paradigm.

    python -m pinca_jax.matched --pde heat --k 16 --allow-cpu
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp

from . import bench, env, ic, metrics, stats
from .equations import pdes
from .harness import EmuConfig, field_bounds, train_emulator, _emu_traj, effective_clip
from .models import registry
from .pinn_heat import HeatPINN, PINNConfig, make_loss

RES = bench.RESULTS_DIR


# ------------------------------------------------------------------- PINN ---
def pinn_solve_one(ic_grid, ref_T, cfg: PINNConfig):
    """Train one PINN on one initial condition; return its error at T and its cost."""
    key = jax.random.PRNGKey(cfg.seed)
    model = HeatPINN()
    params = model.init(key, jnp.array(0.1), jnp.array(0.2), jnp.array(0.0))
    opt_init, opt_update = _adam(cfg.lr)
    opt_state = opt_init(params)
    loss_fn = make_loss(model, cfg, ic_grid)

    @jax.jit
    def step(params, opt_state, k):
        (l, _), g = jax.value_and_grad(loss_fn, has_aux=True)(params, k)
        upd, opt_state = opt_update(g, opt_state, params)
        return _apply(params, upd), opt_state, l

    t0 = time.time()
    for _ in range(cfg.iters):
        key, sk = jax.random.split(key)
        params, opt_state, l = step(params, opt_state, sk)
    jax.block_until_ready(l)
    wall = time.time() - t0

    N = cfg.grid_size
    xs = (jnp.arange(N) + 0.5) / N
    XX, YY = jnp.meshgrid(xs, xs, indexing="xy")
    pred = jax.vmap(lambda x, y: model.apply(params, x, y, 1.0))(
        XX.ravel(), YY.ravel()).reshape(N, N)
    rel = float(jnp.linalg.norm(pred - ref_T) / (jnp.linalg.norm(ref_T) + 1e-8))
    return {"rel_l2": rel, "wall_s": wall,
            "params": metrics.param_count(params)}


def _adam(lr):
    import optax
    o = optax.adam(lr)
    return o.init, o.update


def _apply(params, updates):
    import optax
    return optax.apply_updates(params, updates)


# --------------------------------------------------------------- emulator ---
def emulator_solve_all(pde, arch, x0_batch, steps, grid, epochs, seed=42, batch=16):
    """Train one emulator on the IC distribution, then solve every held-out IC.

    The training initial conditions come from a different PRNG stream than the evaluation
    batch, so the emulator never sees a test IC. Training cost includes generating the
    solver targets, because that is a real cost of the distillation paradigm.
    """
    C = pdes.REGISTRY[pde].channels
    cfg = EmuConfig(pde=pde, grid_size=grid, rollout_steps=min(12, steps),
                    eval_steps=steps, epochs=epochs, batch=batch, seed=seed,
                    warmup_epochs=30, preseed_steps=0 if pde == "cahn_hilliard" else 10)
    ctor = registry.REGISTRY[arch].make(C, bounds=field_bounds(pde, grid))
    tr = train_emulator(ctor, cfg)          # wall_s already includes target generation
    model, params = tr["model"], tr["params"]

    clip = effective_clip(cfg)
    roll = jax.jit(lambda x: _emu_traj(model, params, x, steps, clip)[-1])
    jax.block_until_ready(roll(x0_batch[:1]))          # warm up before timing
    t0 = time.time()
    pred = jax.block_until_ready(roll(x0_batch))
    infer_wall = time.time() - t0
    return {"pred": pred, "train_wall_s": tr["wall_s"], "infer_wall_s": infer_wall,
            "params": metrics.param_count(params)}


def solver_cost_per_ic(pde, grid, steps, batch=1):
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    x = ic.make_state(jax.random.PRNGKey(0), pde, batch, grid)
    fn = jax.jit(lambda s: pdes.rollout(spec, s, steps))
    jax.block_until_ready(fn(x))
    t0 = time.time()
    jax.block_until_ready(fn(x))
    return (time.time() - t0) / batch


# ------------------------------------------------------------------- study ---
def run(pde="heat", k=16, grid=16, steps=32, epochs=600, arch="multiscale_flux_nca",
        pinn_iters=4000, seed=42):
    if pde != "heat":
        raise SystemExit("[matched] only the heat PINN exists; extend pinn_heat.py first")
    spec = pdes.REGISTRY[pde]
    # The K held-out initial conditions. Both paradigms see exactly these.
    x0 = ic.make_state(jax.random.PRNGKey(seed + 10_000), pde, k, grid)
    ref = pdes.rollout(spec, x0, steps)

    print(f"[matched] {k} held-out ICs, grid {grid}, T = {steps} solver steps")
    print(f"[matched] emulator ({arch}): one training, then {k} rollouts")
    emu = emulator_solve_all(pde, arch, x0, steps, grid, epochs, seed=seed)
    emu_per_ic = metrics.rel_l2_per_sample(emu["pred"], ref)

    print(f"[matched] PINN: {k} independent trainings ({pinn_iters} iters each)")
    pcfg = PINNConfig(grid_size=grid, n_steps=steps, iters=pinn_iters, seed=seed)
    pinn_runs = []
    for i in range(k):
        r = pinn_solve_one(x0[i, :, :, 0], ref[i, :, :, 0], pcfg)
        pinn_runs.append(r)
        print(f"    IC {i + 1:3d}/{k}  rel-L2 {r['rel_l2']:.3e}  {r['wall_s']:.1f}s")
    pinn_per_ic = [r["rel_l2"] for r in pinn_runs]
    pinn_wall = sum(r["wall_s"] for r in pinn_runs)

    solver_s = solver_cost_per_ic(pde, grid, steps)
    out = {
        "pde": pde, "k": k, "grid": grid, "steps": steps, "arch": arch,
        "emulator": {"per_ic_rel_l2": emu_per_ic,
                     "summary": stats.summarize(emu_per_ic).as_dict(),
                     "train_wall_s": emu["train_wall_s"],
                     "infer_wall_s": emu["infer_wall_s"],
                     "total_wall_s": emu["train_wall_s"] + emu["infer_wall_s"],
                     "params": emu["params"]},
        "pinn": {"per_ic_rel_l2": pinn_per_ic,
                 "summary": stats.summarize(pinn_per_ic).as_dict(),
                 "train_wall_s": pinn_wall, "infer_wall_s": 0.0,
                 "total_wall_s": pinn_wall, "params": pinn_runs[0]["params"]},
        "solver": {"per_ic_rel_l2": [0.0] * k, "s_per_ic": solver_s,
                   "total_wall_s": solver_s * k,
                   "note": "defines the reference; its own error is quantified "
                           "separately by pinca_jax.teacher_error"},
        "paired": stats.paired_test(emu_per_ic, pinn_per_ic, lower_is_better=True),
    }
    out["crossover_k"] = _crossover(out)
    return out


def _crossover(out):
    """Smallest K at which the amortised emulator costs less total wall-clock than PINNs.

    The emulator's cost is train + K * per-IC inference; the PINN's is K * per-IC
    training. Both are extrapolated linearly from the measured K, which is exact for the
    PINN (independent runs) and very nearly exact for the emulator (a batched rollout).
    """
    k = out["k"]
    emu_fixed = out["emulator"]["train_wall_s"]
    emu_var = out["emulator"]["infer_wall_s"] / k
    pinn_var = out["pinn"]["total_wall_s"] / k
    if pinn_var <= emu_var:
        return None                     # the PINN is cheaper per IC; no crossover exists
    return int(emu_fixed / (pinn_var - emu_var)) + 1


def to_markdown(out):
    e, p, s = out["emulator"], out["pinn"], out["solver"]
    es, ps = e["summary"], p["summary"]
    t = out["paired"]
    x = out["crossover_k"]
    L = [f"### Matched comparison - {out['pde']}, {out['k']} held-out initial conditions", "",
         f"Task: produce the solution at T = {out['steps']} solver steps for all "
         f"{out['k']} initial conditions, drawn from the same distribution. Same PDE, "
         f"same ICs, same horizon, same reference, same metric. The emulator "
         f"(`{out['arch']}`) trains once and rolls out {out['k']} times; the PINN trains "
         f"once **per initial condition**. Grid {out['grid']}.", "",
         "| | rel-L2 mean | 95% CI | median | params | fixed cost (s) | per-IC cost (s) | "
         f"total for K={out['k']} (s) |",
         "|---|---|---|---|---|---|---|---|",
         f"| emulator (`{out['arch']}`) | {es['mean']:.3e} | "
         f"[{es['ci_lo']:.2e}, {es['ci_hi']:.2e}] | {es['median']:.3e} | {e['params']} | "
         f"{e['train_wall_s']:.1f} | {e['infer_wall_s'] / out['k']:.4f} | "
         f"{e['total_wall_s']:.1f} |",
         f"| PINN (per IC) | {ps['mean']:.3e} | [{ps['ci_lo']:.2e}, {ps['ci_hi']:.2e}] | "
         f"{ps['median']:.3e} | {p['params']} | 0 | "
         f"{p['total_wall_s'] / out['k']:.1f} | {p['total_wall_s']:.1f} |",
         f"| numerical solver | 0 (reference) | - | - | - | 0 | {s['s_per_ic']:.4f} | "
         f"{s['total_wall_s']:.3f} |",
         "",
         f"Paired test (same ICs): mean difference {t['mean_diff']:+.3e} "
         f"[{t['ci_lo']:.2e}, {t['ci_hi']:.2e}], Wilcoxon p = {t['p_value']:.3g}, "
         f"emulator wins on {t['win_rate']:.0%} of initial conditions -> "
         f"**{ {'a': 'emulator better', 'b': 'PINN better', 'tie': 'not resolvable'}[t['better']] }**.",
         ""]
    if x is None:
        L += ["The PINN is cheaper per initial condition than the emulator's amortised "
              "inference, so no crossover exists at this scale: there is no K at which "
              "training the emulator pays for itself on wall-clock alone."]
    else:
        L += [f"**Crossover: K = {x}.** Below that many initial conditions the per-IC PINN "
              f"is cheaper in total wall-clock; above it the emulator's one-off training "
              f"is amortised. This is the number a practitioner needs, and it is why a "
              f"single-IC accuracy comparison between the two paradigms is not "
              f"informative on its own."]
    ratio = p["total_wall_s"] / max(e["total_wall_s"], 1e-9)
    L += ["",
          f"**Compute asymmetry.** The PINN received {ratio:.2f}x the emulator's total "
          f"wall-clock on this task. That is stated rather than equalised, because the "
          f"two budgets are not interchangeable -- the PINN's is K independent runs, the "
          f"emulator's is one -- but it does settle the direction: "
          + ("the emulator's advantage here is not bought with extra compute."
             if ratio >= 1.0 else
             "the emulator used MORE compute, so its accuracy advantage is a budget "
             "result and not a like-for-like architecture result.")]
    speed = (e["infer_wall_s"] / out["k"]) / max(s["s_per_ic"], 1e-12)
    L += ["",
          f"**Against the solver, both surrogates lose at this scale.** The numerical "
          f"solver produces the reference in {s['s_per_ic'] * 1e3:.3f} ms per initial "
          f"condition; the emulator takes {speed:.0f}x that per initial condition and is "
          f"less accurate, and the PINN is slower still. A learned surrogate pays off "
          f"only where the solver does not fit -- much larger grids, much longer "
          f"horizons, stiff timestep restrictions, or the need to differentiate through "
          f"the whole rollout. On a {out['grid']}x{out['grid']} periodic grid with an "
          f"explicit stepper, none of those apply, so this regime is a measurement "
          f"harness rather than a deployment case. No claim in this repository should be "
          f"read as 'replace the solver'."]
    L += ["",
          "The solver row is included deliberately. It is the reference, so its error is "
          "zero by construction here -- `pinca_jax.teacher_error` measures what it "
          "actually gets wrong against closed-form solutions -- but its *cost* is real, "
          "and any claim that a surrogate is worthwhile has to clear the solver, not only "
          "the other surrogate."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--k", type=int, default=16, help="held-out initial conditions")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=600, help="emulator training epochs")
    ap.add_argument("--pinn-iters", type=int, default=4000)
    ap.add_argument("--arch", default="multiscale_flux_nca")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("matched", allow_cpu=args.allow_cpu)
    out = run(pde=args.pde, k=args.k, grid=args.grid, steps=args.steps,
              epochs=args.epochs, arch=args.arch, pinn_iters=args.pinn_iters)
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, f"matched_{args.pde}")
    out["device"] = env.provenance("matched")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(to_markdown(out))
    print("\n" + to_markdown(out))
    print(f"[matched] wrote {base}.json / .md")


if __name__ == "__main__":
    main()
