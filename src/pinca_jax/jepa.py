"""JEPA-style latent pretraining for the latent-FNO emulator, and whether it pays.

The proposal this implements: encode the field, evolve the *latent* with an FNO, and
train by matching a stop-gradient EMA encoding of the future field, with no field-space
reconstruction in the loss at all.

Three things about that have to be handled honestly before it can be compared with
anything else in this benchmark, and they are the reason this module exists rather than
a registry entry alone.

**1. A latent loss is not comparable to a field error.** Every accuracy number in this
study is a relative $L^2$ in field space against the reference solver. A latent MSE is a
different quantity in a space the model chose for itself; a model can drive it to zero
by choosing a degenerate space. So a JEPA-pretrained encoder/predictor is never scored by
its own objective here. It is scored by freezing it and fitting *only* the decoder
(`fit_decoder`), then evaluating that model in field space through the same harness as
everything else. That is a linear-probe protocol, and it is what makes the comparison
legitimate.

**2. EMA does not prevent representation collapse; it is only known to help.** A
collapsed encoder emits a near-constant latent, the target matches it trivially, and the
loss looks excellent. `collapse_metrics` therefore reports the latent's per-dimension
standard deviation and its effective rank (the exponentiated entropy of the normalised
covariance spectrum, in [1/d, 1] after normalising by d). A constant-encoder control is
included, which is the JEPA analogue of this project's identity floor: it bounds what a
degenerate solution scores, so "the loss went down" cannot be mistaken for "the
representation is useful".

**3. Distillation already supplies dense, correct supervision.** JEPA's usual motivation
is that pixel reconstruction wastes capacity on detail that is unpredictable in
principle. Here the target is a deterministic solver trajectory, so that motivation is
much weaker: the field-space target is neither noisy nor unpredictable. The honest
hypothesis is therefore *not* that latent training is more accurate, but that it might
buy (a) cheaper rollout, since the predictor runs on a p-fold downsampled grid, and (b)
a representation that transfers better off-distribution. This module measures (a)
directly and leaves (b) to the out-of-distribution harness.

The study compares three regimes for the *same architecture and parameter count*:

    distill     end-to-end field-space distillation (the control, and the protocol
                every other architecture in the benchmark uses)
    jepa_probe  JEPA pretraining, encoder and predictor then frozen, decoder only fitted
    jepa_ft     JEPA pretraining, then the same end-to-end distillation as `distill`

`distill` is the one to beat. If `jepa_ft` does not beat it, the pretraining bought
nothing on this task, and that is a reportable result.

    python -m pinca_jax.jepa --pde heat --allow-cpu
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import bench, env, ic, metrics, stats
from .equations import pdes
from .harness import (EmuConfig, effective_clip, evaluate_emulator, field_bounds,
                      train_emulator, _emu_traj)
from .models.latent_fno import LatentFNOEmulator, largest_patch, latent_rollout

RES = bench.RESULTS_DIR


# --------------------------------------------------------------- diagnostics ---
def collapse_metrics(z: jax.Array) -> dict:
    """Is this representation degenerate?

    `z` is (B, H', W', d). Positions are treated as samples, so the covariance is over
    the latent dimension. Two numbers:

    * `latent_std` -- mean per-dimension standard deviation. Near zero is collapse.
    * `eff_rank` -- exp(entropy of the normalised covariance eigenvalues), divided by d.
      1.0 means variance spread evenly over all directions; 1/d means every sample lies
      on one line. This is the quantity a low loss cannot fake.
    """
    Z = z.reshape(-1, z.shape[-1])
    Z = Z - Z.mean(axis=0, keepdims=True)
    n, d = Z.shape
    if n < 2:
        return {"latent_std": float("nan"), "eff_rank": float("nan")}
    # eigenvalues of the covariance, via singular values (numerically the safer route)
    s = jnp.linalg.svd(Z / jnp.sqrt(n - 1), compute_uv=False)
    lam = s ** 2
    tot = jnp.sum(lam) + 1e-12
    p = lam / tot
    ent = -jnp.sum(jnp.where(p > 0, p * jnp.log(p + 1e-12), 0.0))
    return {"latent_std": float(jnp.mean(jnp.std(Z, axis=0))),
            "eff_rank": float(jnp.exp(ent) / d),
            "top1_var_frac": float(p[0])}


def _normalise(z, eps=1e-6):
    """Standardise each latent channel over the batch and spatial positions.

    Comparing raw latents lets the loss be reduced by shrinking the representation,
    which is the collapse direction. Standardising removes that degree of freedom from
    the objective without removing it from the diagnostics.
    """
    mu = z.mean(axis=(0, 1, 2), keepdims=True)
    sd = z.std(axis=(0, 1, 2), keepdims=True)
    return (z - mu) / (sd + eps)


def latent_loss(pred, target, kind="mse", var_weight=0.0):
    """Latent-space objective. `target` must already be stop-gradiented by the caller."""
    p, t = _normalise(pred), _normalise(target)
    if kind == "cosine":
        pn = p / (jnp.linalg.norm(p, axis=-1, keepdims=True) + 1e-8)
        tn = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)
        core = jnp.mean(1.0 - jnp.sum(pn * tn, axis=-1))
    else:
        core = jnp.mean((p - t) ** 2)
    if var_weight > 0.0:
        # VICReg-style hinge on the per-dimension standard deviation of the *prediction*:
        # an explicit push away from collapse, so its necessity can be measured rather
        # than assumed away.
        sd = jnp.sqrt(pred.var(axis=(0, 1, 2)) + 1e-6)
        core = core + var_weight * jnp.mean(jax.nn.relu(1.0 - sd))
    return core


# ------------------------------------------------------------------ training ---
def _make_model(pde, grid, latent_dim, patch, width, modes, depth):
    C = pdes.REGISTRY[pde].channels
    return LatentFNOEmulator(out_channels=C, latent_dim=latent_dim,
                             patch=largest_patch(grid, patch), width=width,
                             modes=modes, depth=depth)


def jepa_pretrain(pde, grid, *, epochs=800, batch=32, k_steps=12, lr=1e-3, seed=42,
                  ema=0.99, loss_kind="mse", var_weight=0.0, latent_dim=32, patch=4,
                  width=32, modes=6, depth=4, preseed=10, verbose=False):
    """Train encoder + predictor by latent matching against an EMA target encoder.

    The decoder receives no gradient here at all: `_normalise` and the loss touch only
    latents, so the decoder's zero-initialised weights are still zero afterwards. That
    is deliberate -- it is what makes `fit_decoder` a probe of the representation rather
    than a continuation of training.
    """
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    key = jax.random.PRNGKey(seed)
    key, ik = jax.random.split(key)
    model = _make_model(pde, grid, latent_dim, patch, width, modes, depth)
    dummy = ic.make_state(ik, pde, 1, grid)
    params = model.init(ik, dummy)
    target = jax.tree_util.tree_map(jnp.copy, params)      # EMA copy, encoder is what matters

    opt = optax.adamw(lr, weight_decay=1e-5)
    opt_state = opt.init(params)

    def loss_fn(p, tgt_params, k):
        x0 = ic.make_state(k, pde, batch, grid)
        if preseed > 0:
            x0 = pdes.rollout(spec, x0, preseed)
        xk = pdes.rollout(spec, x0, k_steps)
        z_ctx = model.apply(p, x0, method=model.encode)
        s = model.apply(p, z_ctx, method=model.predict)
        z_tgt = jax.lax.stop_gradient(
            model.apply(tgt_params, xk, method=model.encode))
        return latent_loss(s, z_tgt, kind=loss_kind, var_weight=var_weight)

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2))
    def step(params, target, opt_state, k):
        l, g = jax.value_and_grad(loss_fn)(params, target, k)
        upd, opt_state = opt.update(g, opt_state, params)
        params = optax.apply_updates(params, upd)
        # EMA on every leaf; only the encoder's leaves are read by the target branch,
        # but tracking the whole tree keeps the update a single tree_map.
        target = jax.tree_util.tree_map(lambda t, s: ema * t + (1.0 - ema) * s,
                                        target, params)
        return params, target, opt_state, l

    t0 = time.time()
    keys = jax.random.split(key, epochs)
    losses = []
    for i in range(epochs):
        params, target, opt_state, l = step(params, target, opt_state, keys[i])
        losses.append(l)
        if verbose and i % max(1, epochs // 8) == 0:
            print(f"    jepa epoch {i:5d} | latent loss {float(l):.4e}")
    losses = [float(v) for v in losses]

    # Collapse diagnostics on held-out states, not on training states.
    probe = ic.make_state(jax.random.PRNGKey(seed + 10_000), pde, 16, grid)
    z = model.apply(params, probe, method=model.encode)
    cm = collapse_metrics(z)
    return {"model": model, "params": params, "target": target, "losses": losses,
            "wall_s": time.time() - t0, "collapse": cm,
            "decoder_untouched": _decoder_is_zero(params)}


def _decoder_is_zero(params) -> bool:
    """Confirm the latent objective really left the decoder alone."""
    leaves = _decoder_leaves(params)
    return bool(leaves) and all(float(jnp.max(jnp.abs(v))) == 0.0 for v in leaves)


def _decoder_leaves(params):
    out = []
    for path, v in jax.tree_util.tree_flatten_with_path(params)[0]:
        if any("decoder" in str(k) for k in path):
            out.append(v)
    return out


def _decoder_labels(params):
    """Pytree labelling each leaf "train" (decoder) or "freeze" (everything else).

    Used with `optax.multi_transform`, not `optax.masked`. `masked` applies the inner
    transform to the selected leaves and passes the *remaining updates through
    untouched* -- which means the unselected leaves receive their raw gradient and move
    by gradient ascent. That silently turns a frozen probe into fine-tuning, and it is
    what the encoder fingerprint assertion in `fit_decoder` caught. `multi_transform`
    routes the rest to `set_to_zero`, which is the behaviour intended here.
    """
    def label(path, _):
        return "train" if any("decoder" in str(k) for k in path) else "freeze"
    return jax.tree_util.tree_map_with_path(label, params)


def fit_decoder(pre, pde, grid, *, epochs=400, batch=32, rollout_steps=12, lr=1e-3,
                seed=42, preseed=10):
    """Fit ONLY the decoder on frozen encoder+predictor, by field-space rollout MSE.

    A linear-probe protocol: the representation is judged by how well a fixed-capacity
    readout can recover the solver's trajectory from it. The mask is asserted rather
    than trusted, because a silent leak of gradient into the encoder would turn this
    into ordinary fine-tuning and invalidate the comparison.
    """
    spec = pdes.STABLE.get(pde, pdes.REGISTRY[pde])
    model, params = pre["model"], jax.tree_util.tree_map(jnp.copy, pre["params"])
    labels = _decoder_labels(params)
    n_train = sum(int(np.prod(v.shape)) for v, lab in
                  zip(jax.tree_util.tree_leaves(params),
                      jax.tree_util.tree_leaves(labels)) if lab == "train")
    opt = optax.multi_transform(
        {"train": optax.adamw(lr, weight_decay=0.0), "freeze": optax.set_to_zero()},
        labels)
    opt_state = opt.init(params)
    enc_before = _encoder_fingerprint(params)

    def loss_fn(p, k):
        x0 = ic.make_state(k, pde, batch, grid)
        if preseed > 0:
            x0 = pdes.rollout(spec, x0, preseed)
        tgt = pdes.rollout(spec, x0, rollout_steps)
        pred = _emu_traj(model, p, x0, rollout_steps, None)[-1]
        return jnp.mean((pred - tgt) ** 2)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, k):
        l, g = jax.value_and_grad(loss_fn)(params, k)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, l

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), epochs)
    losses = []
    for i in range(epochs):
        params, opt_state, l = step(params, opt_state, keys[i])
        losses.append(float(l))
    assert _encoder_fingerprint(params) == enc_before, (
        "the decoder-only mask leaked gradient into the encoder; the probe would then be "
        "fine-tuning and could not be compared with the frozen protocol")
    return {"model": model, "params": params, "losses": losses,
            "wall_s": time.time() - t0, "trainable_params": n_train}


def _frozen_leaves(params):
    """The leaves a decoder-only probe must not touch, keyed by path."""
    return {tuple(str(k) for k in path): v
            for path, v in jax.tree_util.tree_flatten_with_path(params)[0]
            if any(("encoder" in str(k) or "predictor" in str(k)) for k in path)}


def _encoder_fingerprint(params):
    """Exact identity of the frozen subtree.

    Compared with `jnp.array_equal` rather than by a float summary: a summary statistic
    can coincide after a real change, and rounding one introduces a tolerance where an
    exact answer is available. This is an assertion about whether an optimiser touched
    a tensor, so it should be exact.
    """
    return tuple(sorted((k, v.shape, bytes(np.asarray(v).view(np.uint8).tobytes()))
                        for k, v in _frozen_leaves(params).items()))


# ------------------------------------------------------------------- rollout ---
def rollout_cost(model, params, x0, steps, n=5):
    """Wall-clock per rollout for per-step decoding versus latent-only iteration."""
    per_step = jax.jit(lambda x: _emu_traj(model, params, x, steps, None)[-1])
    latent = jax.jit(lambda x: latent_rollout(model, params, x, steps))
    out = {}
    for name, fn in (("per_step", per_step), ("latent", latent)):
        jax.block_until_ready(fn(x0))
        ts = []
        for _ in range(n):
            t0 = time.time()
            jax.block_until_ready(fn(x0))
            ts.append(time.time() - t0)
        out[name] = float(np.median(ts))
    out["speedup"] = out["per_step"] / (out["latent"] + 1e-12)
    return out


def evaluate(model, params, cfg: EmuConfig):
    """Field-space evaluation through the shared harness, plus the latent-rollout mode."""
    ev = evaluate_emulator(model, params, cfg)
    spec = cfg.spec()
    x0 = ic.make_state(jax.random.PRNGKey(cfg.seed + 10_000), cfg.pde, cfg.n_eval,
                       cfg.grid_size)
    tgt = pdes.rollout(spec, x0, cfg.eval_steps)
    lat = latent_rollout(model, params, x0, cfg.eval_steps)
    clip = effective_clip(cfg)
    if clip is not None:
        lat = jnp.clip(lat, clip[0], clip[1])
    ev["latent_rollout_rel_l2"] = metrics.rel_l2(lat, tgt)
    ev["latent_rollout_per_ic"] = metrics.rel_l2_per_sample(lat, tgt)
    ev["cost"] = rollout_cost(model, params, x0, cfg.eval_steps)
    z = model.apply(params, x0, method=model.encode)
    ev["collapse"] = collapse_metrics(z)
    return ev


# --------------------------------------------------------------------- study ---
def _cfg(pde, grid, epochs, rollout, eval_steps, batch, n_eval, seed):
    return EmuConfig(pde=pde, grid_size=grid, rollout_steps=rollout,
                     eval_steps=eval_steps, epochs=epochs, batch=batch, n_eval=n_eval,
                     seed=seed, warmup_epochs=min(30, epochs // 4),
                     preseed_steps=0 if pde == "cahn_hilliard" else 10)


def study(pde, *, grid=48, epochs=1200, jepa_epochs=800, probe_epochs=400, batch=32,
          rollout=12, eval_steps=48, n_eval=8, seeds=(0, 1, 2), latent_dim=32, patch=4,
          width=32, modes=6, depth=4, var_weight=0.0, loss_kind="mse", verbose=False):
    """distill vs jepa_probe vs jepa_ft, same architecture, same parameter count."""
    out = {k: [] for k in ("distill", "jepa_probe", "jepa_ft", "constant_encoder")}
    for seed in seeds:
        cfg = _cfg(pde, grid, epochs, rollout, eval_steps, batch, n_eval, seed)
        bounds = field_bounds(pde, grid)
        ctor = lambda: _make_model(pde, grid, latent_dim, patch, width, modes, depth)

        # 1. control: the protocol every other architecture in the benchmark uses.
        tr = train_emulator(ctor, cfg)
        ev = evaluate(tr["model"], tr["params"], cfg)
        ev["train_wall_s"] = tr["wall_s"]
        out["distill"].append(ev)

        # 2. JEPA pretraining, then a decoder-only probe.
        pre = jepa_pretrain(pde, grid, epochs=jepa_epochs, batch=batch,
                            k_steps=rollout, seed=seed, loss_kind=loss_kind,
                            var_weight=var_weight, latent_dim=latent_dim, patch=patch,
                            width=width, modes=modes, depth=depth,
                            preseed=cfg.preseed_steps, verbose=verbose)
        probe = fit_decoder(pre, pde, grid, epochs=probe_epochs, batch=batch,
                            rollout_steps=rollout, seed=seed,
                            preseed=cfg.preseed_steps)
        ev = evaluate(probe["model"], probe["params"], cfg)
        ev["train_wall_s"] = pre["wall_s"] + probe["wall_s"]
        ev["jepa"] = {"latent_loss_final": pre["losses"][-1],
                      "collapse": pre["collapse"],
                      "decoder_untouched": pre["decoder_untouched"],
                      "probe_trainable_params": probe["trainable_params"]}
        out["jepa_probe"].append(ev)

        # 3. JEPA pretraining, then the same end-to-end distillation as the control.
        ft = _finetune(pre, cfg)
        ev = evaluate(ft["model"], ft["params"], cfg)
        ev["train_wall_s"] = pre["wall_s"] + ft["wall_s"]
        out["jepa_ft"].append(ev)

        # 4. collapse floor: a deliberately degenerate encoder, decoder-only probe.
        deg = _degenerate(pre)
        probe0 = fit_decoder(deg, pde, grid, epochs=probe_epochs, batch=batch,
                             rollout_steps=rollout, seed=seed,
                             preseed=cfg.preseed_steps)
        ev = evaluate(probe0["model"], probe0["params"], cfg)
        ev["train_wall_s"] = probe0["wall_s"]
        out["constant_encoder"].append(ev)

        if verbose:
            for k in out:
                print(f"  seed {seed} {k:18s} rel-L2 {out[k][-1]['rel_l2']:.4e} "
                      f"eff_rank {out[k][-1]['collapse']['eff_rank']:.3f}")
    return out


def _finetune(pre, cfg: EmuConfig):
    """End-to-end distillation starting from the JEPA-pretrained weights."""
    spec = cfg.spec()
    model, params = pre["model"], jax.tree_util.tree_map(jnp.copy, pre["params"])
    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)

    def loss_fn(p, k):
        x0 = ic.make_state(k, cfg.pde, cfg.batch, cfg.grid_size)
        if cfg.preseed_steps > 0:
            x0 = pdes.rollout(spec, x0, cfg.preseed_steps)
        tgt = pdes.rollout(spec, x0, cfg.rollout_steps)
        pred = _emu_traj(model, p, x0, cfg.rollout_steps, None)[-1]
        return jnp.mean((pred - tgt) ** 2)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, opt_state, k):
        l, g = jax.value_and_grad(loss_fn)(params, k)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, l

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 3), cfg.epochs)
    for i in range(cfg.epochs):
        params, opt_state, l = step(params, opt_state, keys[i])
    return {"model": model, "params": params, "wall_s": time.time() - t0}


def _degenerate(pre):
    """Zero the encoder so its output is constant: the collapse floor.

    This is the JEPA analogue of the identity emulator in the accuracy tables. It bounds
    what a decoder alone can achieve with no information from the encoder, so a probe
    score can be read as evidence about the representation rather than about the decoder.
    """
    def z(path, v):
        return jnp.zeros_like(v) if any("encoder" in str(k) for k in path) else v
    return {**pre, "params": jax.tree_util.tree_map_with_path(z, pre["params"])}


def _pool(runs, key="per_ic_rel_l2"):
    return [v for r in runs for v in r.get(key, [])]


def _agg(runs):
    s = stats.summarize(_pool(runs))
    sl = stats.summarize(_pool(runs, "latent_rollout_per_ic"))
    return {
        "rel_l2": s.as_dict(), "latent_rollout_rel_l2": sl.as_dict(),
        "eff_rank": float(np.mean([r["collapse"]["eff_rank"] for r in runs])),
        "latent_std": float(np.mean([r["collapse"]["latent_std"] for r in runs])),
        "params": int(runs[0]["params"]),
        "train_wall_s": float(np.mean([r["train_wall_s"] for r in runs])),
        "cost": {k: float(np.mean([r["cost"][k] for r in runs]))
                 for k in runs[0]["cost"]},
        "per_ic_rel_l2": _pool(runs),
        "latent_per_ic": _pool(runs, "latent_rollout_per_ic"),
        "jepa": runs[0].get("jepa"),
        "n_seeds": len(runs),
    }


def to_markdown(pde, agg, note=""):
    order = ["distill", "jepa_probe", "jepa_ft", "constant_encoder"]
    order = [k for k in order if k in agg]
    L = [f"### JEPA latent pretraining vs distillation - {pde} {note}", "",
         "Same architecture (`latent_fno`) and parameter count in every row; only the "
         "training regime differs. `distill` is the control and uses the identical "
         "protocol as every other architecture in the benchmark. `jepa_probe` freezes the "
         "JEPA-pretrained encoder and predictor and fits only the decoder. "
         "`constant_encoder` zeroes the encoder and fits the decoder alone: it is the "
         "collapse floor, and a probe that does not clearly beat it has learned nothing "
         "the decoder could not infer without it.", "",
         "| regime | rel-L2 (per-step decode) | 95% CI | rel-L2 (latent rollout) | "
         "eff. rank | latent std | train (s) |",
         "|---|---|---|---|---|---|---|"]
    for k in order:
        a = agg[k]
        r, lr = a["rel_l2"], a["latent_rollout_rel_l2"]
        L.append(f"| {k} | {r['mean']:.4e} | [{r['ci_lo']:.3e}, {r['ci_hi']:.3e}] | "
                 f"{lr['mean']:.4e} | {a['eff_rank']:.3f} | {a['latent_std']:.3g} | "
                 f"{a['train_wall_s']:.0f} |")
    # paired verdicts against the control
    if "distill" in agg:
        L += ["", "Paired against `distill` on the same initial conditions "
                  "(Wilcoxon, Holm-corrected):", ""]
        samples = {k: agg[k]["per_ic_rel_l2"] for k in order}
        cmp = stats.compare_archs(samples, reference="distill", lower_is_better=True)
        L += ["| regime | mean diff vs distill | p (Holm) | verdict |", "|---|---|---|---|"]
        for k in order:
            if k == "distill":
                continue
            t = cmp["comparisons"][k]
            verdict = ("worse" if t.get("holm_significant") and t["better"] == "b"
                       else "better" if t.get("holm_significant") else "tie")
            L.append(f"| {k} | {t['mean_diff']:+.3e} | {t['p_value']:.2g} | {verdict} |")
    a = agg.get("distill") or agg[order[0]]
    L += ["", f"**Latent rollout cost.** Encoding once and iterating the predictor on "
              f"the downsampled latent grid takes {a['cost']['latent'] * 1e3:.2f} ms "
              f"against {a['cost']['per_step'] * 1e3:.2f} ms for per-step decoding, a "
              f"{a['cost']['speedup']:.2f}x ratio over {len(order)} regimes' worth of "
              f"identical architecture. The two are different maps -- latent rollout "
              f"never returns to field space in between, so it is not constrained to "
              f"match the teacher at intermediate steps -- which is why both accuracy "
              f"columns are reported and never pooled."]
    if "jepa_probe" in agg and agg["jepa_probe"].get("jepa"):
        j = agg["jepa_probe"]["jepa"]
        floor = agg.get("constant_encoder", {}).get("eff_rank")
        ctrl = agg.get("distill", {}).get("eff_rank")
        L += ["", f"**Collapse check.** Final latent loss "
                  f"{j['latent_loss_final']:.3e}; effective rank at the end of "
                  f"pretraining {j['collapse']['eff_rank']:.3f}, with "
                  f"{j['collapse']['top1_var_frac']:.0%} of the latent variance in a "
                  f"single direction. The decoder was verified untouched by the latent "
                  f"objective: {j['decoder_untouched']}."]
        if floor is not None and ctrl is not None:
            L += ["", f"Effective rank must be read against two references, not against "
                      f"1.0. The degenerate floor is {floor:.3f} (a constant encoder, "
                      f"i.e.\\ 1/d), and the distillation-trained encoder --- which has "
                      f"no anti-collapse pressure on it at all --- reaches {ctrl:.3f}. "
                      f"A smooth field genuinely has low-dimensional structure, so a "
                      f"rank well below 1 is expected rather than pathological. The "
                      f"question the diagnostic answers is whether the latent objective "
                      f"drove the representation *below* what plain distillation "
                      f"produces, which would be collapse attributable to the objective "
                      f"and not to the data."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="heat")
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--jepa-epochs", type=int, default=800)
    ap.add_argument("--probe-epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rollout", type=int, default=12)
    ap.add_argument("--eval", type=int, default=48)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--var-weight", type=float, default=0.0,
                    help="VICReg-style variance hinge; 0 tests whether EMA alone holds")
    ap.add_argument("--loss", default="mse", choices=["mse", "cosine"])
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    env.require_gpu("jepa", allow_cpu=args.allow_cpu)
    seeds = tuple(range(args.seeds))
    print(f"[jepa] {args.pde}: grid {args.grid}, patch {args.patch}, "
          f"{len(seeds)} seeds, loss={args.loss}, var_weight={args.var_weight}")
    raw = study(args.pde, grid=args.grid, epochs=args.epochs,
                jepa_epochs=args.jepa_epochs, probe_epochs=args.probe_epochs,
                batch=args.batch, rollout=args.rollout, eval_steps=args.eval,
                n_eval=args.n_eval, seeds=seeds, patch=args.patch,
                latent_dim=args.latent_dim, var_weight=args.var_weight,
                loss_kind=args.loss, verbose=True)
    agg = {k: _agg(v) for k, v in raw.items() if v}
    os.makedirs(RES, exist_ok=True)
    base = os.path.join(RES, f"jepa_{args.pde}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"pde": args.pde, "grid": args.grid, "patch": args.patch,
                   "seeds": list(seeds), "loss_kind": args.loss,
                   "var_weight": args.var_weight, "results": agg,
                   "device": env.provenance("jepa")}, f, indent=1)
    note = (f"(grid={args.grid}, patch={args.patch}, seeds={list(seeds)}, "
            f"loss={args.loss}, var_weight={args.var_weight})")
    md = to_markdown(args.pde, agg, note)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print(f"[jepa] wrote {base}.json / .md")


if __name__ == "__main__":
    main()
