"""Generate every table and every quoted number in the paper, from results/.

The failure mode this prevents is specific and was already present in this project: the
abstract said eight 2-D phenomena while the body said ten, because both were typed by
hand. Any number a reader can check must therefore be produced here and pulled into the
LaTeX by macro, never retyped.

Two outputs, both written to `paper/generated/`:

* `facts.tex` -- a `\\newcommand` per quoted quantity (`\\NpdesTwoD`, `\\bestHeat`, ...).
  Prose in `main.tex` uses the macro. A stale number in the paper is then impossible: it
  either regenerates or the build fails on an undefined macro.
* `tab_*.tex` -- the tables, as `tabular` bodies with booktabs rules.

Every table degrades to an explicit "not yet measured" note when the underlying results
file is absent, rather than silently omitting a row. A missing experiment should be
visible in the compiled PDF, not invisible.

    python -m pinca_jax.paper
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from . import claims, stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "generated")

# Human-readable names, so the LaTeX does not carry snake_case identifiers.
PDE_LABEL = {
    "heat": "Heat", "adv_diff": "Advection--diffusion", "wave": "Wave",
    "allen_cahn": "Allen--Cahn", "cahn_hilliard": "Cahn--Hilliard",
    "gray_scott": "Gray--Scott", "shallow_water": "Shallow water",
    "fitzhugh_nagumo": "FitzHugh--Nagumo", "nagumo": "Nagumo",
    "navier_stokes": "Navier--Stokes",
}
ARCH_LABEL = {
    "plain_nca": "NCA", "pi_nca": "PI-NCA (flux)", "fno": "FNO",
    "fno_small": "FNO (iso-param)", "mc_flux_nca": "Multi-channel PI-NCA",
    "bounded_cons_nca": "Bounded PI-NCA", "spectral_flux_nca": "Spectral PI-NCA",
    "multiscale_flux_nca": "Multi-scale PI-NCA",
    "bounded_multiscale_nca": "Bounded multi-scale PI-NCA",
    "resnet": "ResNet", "resnet_iso": "ResNet (iso-param)", "unet": "U-Net",
    "unet_iso": "U-Net (iso-param)", "identity": "Identity (floor)",
    "abl_flux": "flux head", "abl_residual": "residual head",
    "abl_k3": "$3\\times3$", "abl_k5": "$5\\times5$",
    "abl_multiscale": "dilated $(1,2,4)$",
    "abl_proj_uniform": "uniform projection", "abl_proj_headroom": "headroom projection",
    "abl_proj_none": "clip only",
}


def tex(s) -> str:
    """Escape the characters LaTeX would otherwise interpret."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def label(name, table=None):
    return (table or {}).get(name) or PDE_LABEL.get(name) or ARCH_LABEL.get(name) \
        or tex(name)


def load(name):
    path = os.path.join(RES, f"{name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _missing(what, how):
    return ("\\multicolumn{1}{l}{\\emph{%s: not yet measured. Run \\texttt{%s}.}}\n"
            % (tex(what), tex(how)))


def _rows(d):
    """Architecture rows of a bench file, failed cells excluded but remembered."""
    res = (d or {}).get("results", {})
    ok = {a: r for a, r in res.items() if isinstance(r, dict) and "error" not in r}
    bad = {a: r for a, r in res.items() if isinstance(r, dict) and "error" in r}
    return ok, bad


def num(v, fmt=".3g", dash="--"):
    """Format a number, or `dash` when it is not available.

    Older result files predate the per-IC payloads, so their bootstrap intervals do not
    exist. Printing `nan` in a paper table would look like a computation that failed
    rather than a measurement that has not been rerun; an em-dash says which it is.
    """
    return dash if v != v else format(v, fmt)


def ci(s, dash="--"):
    return dash if s.ci_lo != s.ci_lo else f"[{s.ci_lo:.2g}, {s.ci_hi:.2g}]"


def _mean(rec, key, default=float("nan")):
    v = rec.get(key)
    return float(v["mean"]) if isinstance(v, dict) and "mean" in v else default


def _summary(rec):
    """Bootstrap summary of the per-IC errors when present, else the seed aggregate."""
    per = rec.get("_per_ic_rel_l2")
    if per:
        return stats.summarize(per)
    v = rec.get("rel_l2") or {}
    m, s, n = v.get("mean", float("nan")), v.get("std", 0.0), v.get("n", 0)
    return stats.Summary(n, m, s, 0.0, m, 0.0, float("nan"), float("nan"), m, m)


# --------------------------------------------------------------- the tables ---
def table_regime_map(tag="full"):
    """One row per phenomenon: the best architecture, the floor, and the teacher error."""
    te = (load("teacher_error") or {}).get("results", {})
    lines = []
    for pde in sorted(PDE_LABEL):
        d = load(f"bench_{pde}_{tag}")
        ok, _ = _rows(d)
        if not ok:
            continue
        ranked = sorted(ok, key=lambda a: _summary(ok[a]).mean)
        best = ranked[0]
        bs = _summary(ok[best])
        floor = _summary(ok["identity"]).mean if "identity" in ok else float("nan")
        beats = "yes" if bs.mean < floor else ("\\textbf{no}" if floor == floor else "--")
        trec = te.get(pde, {})
        tm = trec.get("rel_l2", float("nan"))
        rel = trec.get("reliability", "")
        # An equation whose teacher does not converge cannot support a ranking at all,
        # so the row says so instead of printing a winner that means nothing.
        note = "" if rel != "NOT CONVERGING" else r" \textbf{(teacher not converging)}"
        lines.append(
            f"{label(pde)} & {label(best)} & {num(bs.mean)} & {ci(bs)} & "
            f"{num(floor)} & {beats} & {num(tm, '.2g')}{note} \\\\")
    if not lines:
        return _missing("2-D regime map",
                        "python -m pinca_jax.bench_all --group all --seeds 5")
    return "\n".join(lines) + "\n"


def table_paired(pde, tag="headline"):
    """Ranked architectures with CIs and Holm-corrected paired verdicts, for one PDE."""
    d = load(f"bench_{pde}_{tag}") or load(f"bench_{pde}_full")
    ok, _ = _rows(d)
    samples = {a: r["_per_ic_rel_l2"] for a, r in ok.items() if r.get("_per_ic_rel_l2")}
    if not samples:
        return _missing(f"paired comparison on {pde}",
                        f"python -m pinca_jax.bench_all --pdes {pde} --seeds 10 "
                        f"--tag headline")
    cmp = stats.compare_archs(samples, lower_is_better=True)
    ref = cmp["reference"]
    lines = []
    for a in sorted(samples, key=lambda k: stats.summarize(samples[k]).mean):
        s = stats.summarize(samples[a])
        pcount = int(_mean(ok[a], "params", 0))
        if a == ref:
            verdict, pv = "\\textbf{reference}", "--"
        else:
            t = cmp["comparisons"][a]
            pv = f"{t['p_value']:.2g}"
            verdict = ("worse" if t.get("holm_significant") and t["better"] == "b"
                       else "better" if t.get("holm_significant") else "tie")
        lines.append(f"{label(a)} & {pcount} & {num(s.mean)} & {ci(s)} & "
                     f"{num(s.median)} & {pv} & {verdict} \\\\")
    return "\n".join(lines) + "\n"


def table_ablation(tag, pde, archs):
    d = load(f"bench_{pde}_{tag}")
    ok, _ = _rows(d)
    rows = [a for a in archs if a in ok]
    if not rows:
        return _missing(f"ablation {tag} on {pde}",
                        "python -m pinca_jax.bench_all --group ablation --seeds 5")
    lines = []
    for a in rows:
        s = _summary(ok[a])
        lines.append(f"{label(a)} & {int(_mean(ok[a], 'params', 0))} & {num(s.mean)} & "
                     f"{ci(s)} & {num(_mean(ok[a], 'conservation_err'), '.2g')} \\\\")
    return "\n".join(lines) + "\n"


def table_ood(pde="heat"):
    """The whole tabular, not just its rows.

    The OOD table's width is the number of architectures measured, which is data. Emitting
    only the body and fixing the column spec by hand in the LaTeX means the two silently
    disagree the moment an architecture is added or dropped -- so the generator emits the
    complete environment and the paper just includes it.
    """
    d = load(f"ood_{pde}")
    if not d:
        return ("\\begin{tabular}{l}\n\\toprule\nHeld-out axis \\\\\n\\midrule\n"
                + _missing(f"OOD study on {pde}",
                           f"python -m pinca_jax.ood --pde {pde} --seeds 3")
                + "\\bottomrule\n\\end{tabular}\n")
    res = d["results"]
    archs = list(res)
    axes = []
    for a in archs:
        for k in res[a]:
            if k not in axes:
                axes.append(k)
    L = ["\\begin{tabular}{l" + "c" * len(archs) + "}", "\\toprule",
         "Held-out axis & " + " & ".join(label(a) for a in archs) + " \\\\",
         "\\midrule"]
    for ax in axes:
        cells = []
        for a in archs:
            r = res[a].get(ax)
            base = res[a].get("in_dist")
            if r is None or not base:
                cells.append("--")
            else:
                ratio = r["rel_l2_mean"] / (base["rel_l2_mean"] + 1e-12)
                cell = f"{ratio:.2f}$\\times$"
                cells.append(f"\\textbf{{{cell}}}" if r["diverged_frac"] > 0 else cell)
        L.append(tex(ax) + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def table_stability(pde="cahn_hilliard"):
    d = load(f"stability_{pde}")
    if not d:
        return _missing(f"stability study on {pde}",
                        f"python -m pinca_jax.stability --pde {pde} --seeds 3")
    lines = []
    for a, r in d["results"].items():
        bv = r.get("bounded_violation_frac", float("nan"))
        lines.append(
            f"{label(a)} & {r['params']} & {r['failure_rate_final'] * 100:.0f}\\% & "
            f"{r['median_survival']:.0f}/{r['horizon']} & "
            + ("--" if bv != bv else f"{bv * 100:.2f}\\%") +
            f" & {r['perturb_t0_amplification']:.3g}$\\times$ \\\\")
    return "\n".join(lines) + "\n"


def table_teacher():
    d = load("teacher_error")
    if not d:
        return _missing("teacher error", "python -m pinca_jax.teacher_error")
    lines = []
    for pde, r in d["results"].items():
        sp, tp = r.get("rel_l2_spatial"), r.get("rel_l2_temporal")
        order = r.get("observed_order", float("nan"))
        fastest = min(r["emulator_s_per_step"], key=r["emulator_s_per_step"].get)
        lines.append(
            f"{label(pde)} & {tex(r['kind'].replace('_', ' '))} & {r['rel_l2']:.2g} & "
            + ("--" if sp is None else f"{sp:.2g}") + " & "
            + ("--" if tp is None else f"{tp:.2g}") + " & "
            + ("--" if order != order else f"{order:.2f}") +
            f" & {r['solver_s_per_step'] * 1e3:.3f} & "
            f"{r['speedup_vs_solver'][fastest]:.2f}$\\times$ \\\\")
    return "\n".join(lines) + "\n"


def table_matched(pde="heat"):
    d = load(f"matched_{pde}")
    if not d:
        return _missing("matched PINN comparison",
                        "python -m pinca_jax.matched --pde heat --k 32")
    e, p, s = d["emulator"], d["pinn"], d["solver"]
    k = d["k"]
    rows = [
        (f"Emulator ({label(d['arch'])})", e["summary"], e["params"],
         e["train_wall_s"], e["infer_wall_s"] / k, e["total_wall_s"]),
        ("PINN (one run per IC)", p["summary"], p["params"], 0.0,
         p["total_wall_s"] / k, p["total_wall_s"]),
    ]
    lines = [f"{n} & {su['mean']:.3g} & [{su['ci_lo']:.2g}, {su['ci_hi']:.2g}] & "
             f"{pc} & {fx:.1f} & {vr:.4f} & {tot:.1f} \\\\"
             for n, su, pc, fx, vr, tot in rows]
    lines.append(f"Numerical solver (reference) & 0 & -- & -- & 0 & "
                 f"{s['s_per_ic']:.4f} & {s['total_wall_s']:.3f} \\\\")
    return "\n".join(lines) + "\n"


def table_3d(tag=None):
    lines = []
    for pde in sorted(PDE_LABEL):
        d = load(f"bench3d_{pde}")
        ok, _ = _rows(d)
        if not ok:
            continue
        ranked = sorted(ok, key=lambda a: _summary(ok[a]).mean)
        best = ranked[0]
        s = _summary(ok[best])
        floor = _summary(ok["identity"]).mean if "identity" in ok else float("nan")
        lines.append(f"{label(pde)} & {label(best)} & {num(s.mean)} & "
                     f"{num(floor)} \\\\")
    if not lines:
        return _missing("3-D matrix", "python -m pinca_jax.bench3d")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the facts ---
def facts() -> dict:
    inv = claims.inventory()
    f = {
        "NpdesTwoD": inv["n_pdes_2d"], "NpdesThreeD": inv["n_pdes_3d"],
        "Narchs": inv["n_archs_2d"], "Nresultfiles": inv["files"],
        "MinSeeds": inv["min_seeds"], "MaxSeeds": inv["max_seeds"],
        "Backends": ", ".join(inv["backends"]) or "unrecorded",
        "Grids": ", ".join(str(g) for g in inv["grids"]) or "unrecorded",
        "Nfailedcells": len(inv["failed_cells"]),
        "Nablations": len(inv["ablations"]),
    }
    try:
        bibp = json.load(open(os.path.join(ROOT, "docs", "bibliography.json"),
                              encoding="utf-8"))
        f["Ncitations"] = bibp["n_cited"]
        f["Ncitationsresolved"] = bibp["n_resolved"]
    except Exception:
        f["Ncitations"] = f["Ncitationsresolved"] = 0
    f["Ntests"] = _count_tests()

    # Per-PDE winner and floor, so the prose can name them without retyping.
    for pde in PDE_LABEL:
        d = load(f"bench_{pde}_headline") or load(f"bench_{pde}_full")
        ok, _ = _rows(d)
        if not ok:
            continue
        ranked = sorted(ok, key=lambda a: _summary(ok[a]).mean)
        s = _summary(ok[ranked[0]])
        key = pde.title().replace("_", "")
        f[f"best{key}"] = label(ranked[0])
        f[f"best{key}Err"] = num(s.mean)
        f[f"best{key}CI"] = ci(s)
        if "identity" in ok:
            fl = _summary(ok["identity"]).mean
            f[f"floor{key}"] = f"{fl:.3g}"
            f[f"beatsFloor{key}"] = "yes" if s.mean < fl else "no"

    m = load("matched_heat")
    if m:
        f["matchedK"] = m["k"]
        f["matchedCrossover"] = m["crossover_k"] if m["crossover_k"] else "none"
        f["matchedEmuErr"] = f"{m['emulator']['summary']['mean']:.3g}"
        f["matchedPinnErr"] = f"{m['pinn']['summary']['mean']:.3g}"
        sp = (m["emulator"]["infer_wall_s"] / m["k"]) / max(m["solver"]["s_per_ic"], 1e-12)
        f["matchedSolverSpeedup"] = f"{sp:.0f}"
    st = load("stability_cahn_hilliard")
    if st:
        rs = st["results"]
        f["stabHorizon"] = st["horizon"]
        if "plain_nca" in rs:
            f["stabNCAFail"] = f"{rs['plain_nca']['failure_rate_final'] * 100:.0f}"
        for k in ("bounded_multiscale_nca", "bounded_cons_nca"):
            if k in rs:
                f["stabBoundedFail"] = f"{rs[k]['failure_rate_final'] * 100:.0f}"
                break
    te = load("teacher_error")
    if te and "heat" in te["results"]:
        r = te["results"]["heat"]
        f["teacherHeatTotal"] = f"{r['rel_l2']:.2g}"
        if r.get("rel_l2_spatial") is not None:
            f["teacherHeatSpatial"] = f"{r['rel_l2_spatial']:.2g}"
            f["teacherHeatTemporal"] = f"{r['rel_l2_temporal']:.2g}"
    return f


def _count_tests():
    n = 0
    for path in glob.glob(os.path.join(ROOT, "tests", "test_*.py")):
        try:
            n += sum(1 for line in open(path, encoding="utf-8")
                     if line.startswith("def test_"))
        except Exception:
            pass
    return n


def _macro(name, value):
    return "\\newcommand{\\%s}{%s}\n" % (name, value)


def write_all(out=OUT):
    os.makedirs(out, exist_ok=True)
    f = facts()
    with open(os.path.join(out, "facts.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Generated by `python -m pinca_jax.paper`. Do not edit.\n"
                 "% Every number quoted in the paper prose is defined here, from "
                 "results/.\n")
        for k, v in sorted(f.items()):
            fh.write(_macro(k, tex(v) if not isinstance(v, str) else v))

    ood_table = table_ood("heat")
    tables = {
        "tab_regime_map": table_regime_map(),
        "tab_paired_heat": table_paired("heat"),
        "tab_paired_ch": table_paired("cahn_hilliard"),
        "tab_paired_ns": table_paired("navier_stokes"),
        "tab_ablation_a4": table_ablation("A4", "heat", ["abl_flux", "abl_residual"]),
        "tab_ablation_a5": table_ablation("A5", "heat",
                                          ["abl_k3", "abl_k5", "abl_multiscale"]),
        "tab_ablation_a7": table_ablation("A7", "cahn_hilliard",
                                          ["abl_proj_none", "abl_proj_uniform",
                                           "abl_proj_headroom"]),
        "tab_ood": ood_table,
        "tab_stability": table_stability(),
        "tab_teacher": table_teacher(),
        "tab_matched": table_matched(),
        "tab_3d": table_3d(),
    }
    for name, body in tables.items():
        with open(os.path.join(out, name + ".tex"), "w", encoding="utf-8") as fh:
            fh.write("%% Generated by `python -m pinca_jax.paper`. Do not edit.\n")
            fh.write(body)
    print(f"[paper] wrote {len(tables) + 1} files to {out}")
    print(f"[paper] {len(f)} macros; {f['NpdesTwoD']} 2-D / {f['NpdesThreeD']} 3-D "
          f"phenomena, {f['Narchs']} architectures, backend {f['Backends']}")
    return f, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    write_all(ap.parse_args().out)


if __name__ == "__main__":
    main()
