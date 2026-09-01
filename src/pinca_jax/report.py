"""Generate the report's results sections from results/*.json.

The architecture prose lives in `docs/report_template.md`; every results table is
rendered here from whatever the benchmarks actually produced. That is deliberate: the
tables used to be hand-written, so when the benchmark matrix changed the document
silently kept quoting an older, smaller set of architectures. Now the document cannot
disagree with the data.

Run:  python -m pinca_jax.report
Then: python -m pinca_jax.md2pdf docs/PI-NCA_Architectures_and_Results.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from .equations import pdes
from .models import registry

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
TEMPLATE = os.path.join(ROOT, "docs", "report_template.md")
OUT = os.path.join(ROOT, "docs", "PI-NCA_Architectures_and_Results.md")

PDE_ORDER = ["heat", "adv_diff", "allen_cahn", "nagumo", "wave", "cahn_hilliard",
             "gray_scott", "shallow_water", "fitzhugh_nagumo", "navier_stokes"]
PDE_LABEL = {"heat": "Heat", "adv_diff": "Advection-diffusion", "allen_cahn": "Allen-Cahn",
             "nagumo": "Nagumo", "wave": "Wave", "cahn_hilliard": "Cahn-Hilliard",
             "gray_scott": "Gray-Scott", "shallow_water": "Shallow-water",
             "fitzhugh_nagumo": "FitzHugh-Nagumo", "navier_stokes": "Navier-Stokes"}
CHARACTER = {
    "heat": "smooth, local, conservative", "adv_diff": "linear transport, conservative",
    "allen_cahn": "non-conservative phase separation", "nagumo": "non-conservative bistable",
    "wave": "2nd-order hyperbolic", "cahn_hilliard": "stiff 4th-order, bounded",
    "gray_scott": "reaction-diffusion patterns", "shallow_water": "conservative, multi-field",
    "fitzhugh_nagumo": "non-conservative reaction", "navier_stokes": "globally coupled",
}


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _ok(results):
    """Drop cells that recorded a failure."""
    return {a: r for a, r in (results or {}).items()
            if isinstance(r, dict) and "error" not in r and "rel_l2" in r}


def _m(rec, key, default=None):
    cell = rec.get(key)
    return default if cell is None else cell["mean"]


def _fmt(v, kind="sci"):
    if v is None:
        return "—"
    if kind == "int":
        return f"{int(v):,}".replace(",", " ")
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "f3":
        return f"{v:.3f}"
    return f"{v:.3e}"


def _pm(rec, key, kind="sci"):
    """mean ± std, only showing the ± when more than one seed ran."""
    cell = rec.get(key)
    if cell is None:
        return "—"
    m, s, n = cell["mean"], cell.get("std", 0.0), cell.get("n", 1)
    base = _fmt(m, kind)
    if n and n > 1 and s:
        return f"{base} ± {_fmt(s, 'sci' if kind == 'sci' else kind)}"
    return base


def _best(results, key, lower=True):
    vals = {a: _m(r, key) for a, r in results.items() if _m(r, key) is not None}
    vals = {a: v for a, v in vals.items() if v == v}          # drop NaN
    if not vals:
        return None
    return (min if lower else max)(vals, key=vals.get)


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def arch_table():
    rows = []
    heat = _ok((_load("bench_heat_full.json") or {}).get("results"))
    for name in registry.BENCH_ARCHS:
        spec = registry.REGISTRY[name]
        p = _m(heat.get(name, {}), "params") if heat else None
        rows.append([f"`{name}`", spec.note, _fmt(p, "int") if p else "—"])
    return _table(["Name in code", "What it is", "Params (heat)"], rows)


def run_info():
    man = _load("run_manifest.json")
    dev = None
    for f in sorted(glob.glob(os.path.join(RES, "bench_*_full.json"))):
        d = _load(os.path.basename(f))
        if d and d.get("device"):
            dev = d["device"]
            cfg = d.get("config", {})
            seeds = d.get("seeds", [])
            break
    if not dev:
        return "*No benchmark results found yet — run `python -m pinca_jax.runner`.*"
    lines = [f"Measured on **{dev.get('backend', '?')}** "
             f"(`{', '.join(dev.get('devices', []) or ['?'])}`), JAX {dev.get('jax', '?')}, "
             f"grid {cfg.get('grid_size', '?')}, batch {cfg.get('batch', '?')}, "
             f"{cfg.get('epochs', '?')} epochs, train horizon {cfg.get('rollout_steps', '?')} / "
             f"eval horizon {cfg.get('eval_steps', '?')}, "
             f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''}."]
    if man and man.get("total_seconds"):
        lines.append(f"Total run time **{man['total_seconds'] / 3600:.2f} h**. "
                     f"Peak device memory {dev.get('peak_mem_mb', 0):.0f} MB.")
    if len(seeds) == 1:
        lines.append("Single fixed seed (42), so no ± is shown; the protocol is the "
                     "originals' He-init + zero-init heads + LR warm-up + pre-seeding.")
    return "\n\n".join(lines)


def regime_map():
    rows = []
    for pde in PDE_ORDER:
        d = _load(f"bench_{pde}_full.json")
        res = _ok((d or {}).get("results"))
        if not res:
            continue
        win = _best(res, "rel_l2")
        others = sorted(((_m(r, "rel_l2"), a) for a, r in res.items() if a != win),
                        key=lambda t: (t[0] is None, t[0]))
        runner = f"{others[0][1]} {_fmt(others[0][0], 'f3')}" if others else "—"
        rows.append([PDE_LABEL.get(pde, pde), CHARACTER.get(pde, ""),
                     f"**{win}**", _fmt(_m(res[win], "rel_l2"), "f3"), runner,
                     str(len(res))])
    if not rows:
        return "*Not yet run.*"
    return _table(["PDE", "Character", "Winner", "rel-L2", "Runner-up", "Models compared"],
                  rows)


def matrix_2d():
    blocks = []
    for pde in PDE_ORDER:
        d = _load(f"bench_{pde}_full.json")
        res = _ok((d or {}).get("results"))
        if not res:
            continue
        cfg = d.get("config", {})
        C = pdes.REGISTRY[pde].channels
        best_l2 = _best(res, "rel_l2")
        best_cons = _best(res, "conservation_err")
        best_p = _best(res, "params")
        rows = []
        order = sorted(res, key=lambda a: (_m(res[a], "rel_l2") is None,
                                           _m(res[a], "rel_l2")))
        for a in order:
            r = res[a]
            l2 = _pm(r, "rel_l2")
            cons = _fmt(_m(r, "conservation_err"))
            par = _fmt(_m(r, "params"), "int")
            rows.append([
                f"**{a}**" if a == best_l2 else a,
                f"**{l2}**" if a == best_l2 else l2,
                _fmt(_m(r, "psnr"), "f2"),
                f"**{cons}**" if a == best_cons else cons,
                f"**{par}**" if a == best_p else par,
                _fmt(_m(r, "infer_s_per_step")),
            ])
        blocks.append(
            f"**{PDE_LABEL.get(pde, pde)}** — {CHARACTER.get(pde, '')}, C={C}, "
            f"grid {cfg.get('grid_size', '?')}, eval {cfg.get('eval_steps', '?')} steps\n\n"
            + _table(["Model", "rel-L2 ↓", "PSNR ↑", "Mass drift ↓", "Params ↓",
                      "Infer s/step ↓"], rows))
    return "\n\n".join(blocks) if blocks else "*Not yet run.*"


def hybrid_results():
    """The hybrids, on every phenomenon, next to the baselines they were meant to beat."""
    hybrids = ["multiscale_flux_nca", "bounded_cons_nca", "bounded_multiscale_nca",
               "spectral_flux_nca"]
    base = ["plain_nca", "pi_nca", "fno"]
    rows, wins = [], {h: [] for h in hybrids}
    for pde in PDE_ORDER:
        res = _ok((_load(f"bench_{pde}_full.json") or {}).get("results"))
        if not res:
            continue
        win = _best(res, "rel_l2")
        best_base = _best({a: res[a] for a in base if a in res}, "rel_l2")
        row = [PDE_LABEL.get(pde, pde)]
        for h in hybrids:
            v = _m(res.get(h, {}), "rel_l2")
            cell = _fmt(v, "f3")
            if h == win:
                cell = f"**{cell}**"
                wins[h].append(pde)
            row.append(cell)
        row.append(f"{best_base} {_fmt(_m(res.get(best_base, {}), 'rel_l2'), 'f3')}"
                   if best_base else "—")
        rows.append(row)
    if not rows:
        return "*Not yet run.*"
    table = _table(["PDE"] + [f"`{h}`" for h in hybrids] + ["Best baseline"], rows)

    scorecard = []
    intent = {
        "multiscale_flux_nca": "widen the receptive field without an FFT",
        "bounded_cons_nca": "be bounded AND mass-conserving at once",
        "bounded_multiscale_nca": "combine multi-scale reach with bounding",
        "spectral_flux_nca": "add global spectral reach to a local conservative NCA",
    }
    for h in hybrids:
        w = wins[h]
        verdict = ("wins " + ", ".join(PDE_LABEL.get(p, p) for p in w)) if w \
            else "does not win any phenomenon outright"
        scorecard.append([f"`{h}`", intent[h], verdict])
    return (table + "\n\nrel-L2, lower is better. **Bold** marks the overall winner for "
            "that phenomenon across all architectures.\n\n"
            + _table(["Hybrid", "What it targets", "Outcome"], scorecard))


def ablations():
    blocks = []
    for tag, title, question in [
        ("A4", "A4 — conservation on/off at matched backbone width",
         "Same backbone, same widths; only the head differs (flux vs residual). "
         "The cleanest test of whether the conservation prior helps."),
        ("A5", "A5 — perception / receptive-field size",
         "Same head, same widths; only the perception differs (3x3, 5x5, dilated 1/2/4)."),
    ]:
        rows = []
        for path in sorted(glob.glob(os.path.join(RES, f"bench_*_{tag}.json"))):
            pde = os.path.basename(path)[len("bench_"):-len(f"_{tag}.json")]
            res = _ok((_load(os.path.basename(path)) or {}).get("results"))
            if not res:
                continue
            best = _best(res, "rel_l2")
            cells = [PDE_LABEL.get(pde, pde)]
            for a in sorted(res):
                v = _fmt(_m(res[a], "rel_l2"), "f3")
                cells.append(f"**{v}** ({a})" if a == best else f"{v} ({a})")
            rows.append(cells)
        if rows:
            width = max(len(r) for r in rows)
            rows = [r + ["—"] * (width - len(r)) for r in rows]
            head = ["PDE"] + [f"variant {i + 1}" for i in range(width - 1)]
            blocks.append(f"**{title}**\n\n{question}\n\n" + _table(head, rows))
    return "\n\n".join(blocks) if blocks else "*Not yet run.*"


def efficiency():
    res = _ok((_load("bench_heat_full.json") or {}).get("results"))
    if not res:
        return "*Not yet run.*"
    scored = []
    for a, r in res.items():
        l2, p = _m(r, "rel_l2"), _m(r, "params")
        if l2 and p:
            scored.append((l2 * p, a, l2, p))
    if not scored:
        return "*Not yet run.*"
    scored.sort()
    best = scored[0][0]
    rows = [[f"**{a}**" if i == 0 else a, _fmt(l2, "f3"), _fmt(p, "int"),
             _fmt(s, "sci"), f"{s / best:.0f}x"]
            for i, (s, a, l2, p) in enumerate(scored)]
    return ("Cost of accuracy on heat, as rel-L2 x parameters (lower is better). This is "
            "where the conservative NCAs' small size shows up as more than a footnote.\n\n"
            + _table(["Model", "rel-L2", "Params", "rel-L2 x params", "vs best"], rows))


def matrix_3d():
    blocks = []
    for path in sorted(glob.glob(os.path.join(RES, "bench3d_*.json"))):
        pde = os.path.basename(path)[len("bench3d_"):-len(".json")]
        d = _load(os.path.basename(path))
        res = _ok((d or {}).get("results"))
        if not res:
            continue
        grid = (d.get("config") or {}).get("grid_size", "?")
        best = _best(res, "rel_l2")
        best_cons = _best(res, "conservation_err")
        rows = []
        for a in sorted(res, key=lambda a: (_m(res[a], "rel_l2") is None,
                                            _m(res[a], "rel_l2"))):
            r = res[a]
            l2, cons = _fmt(_m(r, "rel_l2"), "f3"), _fmt(_m(r, "conservation_err"))
            rows.append([f"**{a}**" if a == best else a,
                         f"**{l2}**" if a == best else l2,
                         _fmt(_m(r, "psnr"), "f2"),
                         f"**{cons}**" if a == best_cons else cons,
                         _fmt(_m(r, "params"), "int")])
        blocks.append(f"**{PDE_LABEL.get(pde, pde)}** — {grid}³\n\n"
                      + _table(["Model", "rel-L2 ↓", "PSNR ↑", "Mass drift ↓", "Params ↓"],
                               rows))
    return "\n\n".join(blocks) if blocks else "*Not yet run.*"


def resolution():
    blocks = []
    for path in sorted(glob.glob(os.path.join(RES, "bench_resolution_*.json"))):
        pde = os.path.basename(path)[len("bench_resolution_"):-len(".json")]
        d = _load(os.path.basename(path))
        if not d:
            continue
        grids, res = d.get("grids", []), d.get("results", {})
        for arch, cell in res.items():
            rows = []
            for gt in grids:
                row = [f"train {gt}²"]
                for ge in grids:
                    v = cell.get(f"{gt}->{ge}")
                    row.append(f"**{v:.3f}**" if (v is not None and gt == ge)
                               else (f"{v:.3f}" if v is not None else "—"))
                rows.append(row)
            blocks.append(f"**{PDE_LABEL.get(pde, pde)} / `{arch}`** (rel-L2)\n\n"
                          + _table([""] + [f"eval {g}²" for g in grids], rows))
    return "\n\n".join(blocks) if blocks else "*Not yet run.*"


SECTIONS = {
    "ARCH_TABLE": arch_table, "RUN_INFO": run_info, "REGIME_MAP": regime_map,
    "MATRIX_2D": matrix_2d, "HYBRID_RESULTS": hybrid_results, "ABLATIONS": ablations,
    "EFFICIENCY": efficiency, "MATRIX_3D": matrix_3d, "RESOLUTION": resolution,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    with open(args.template, encoding="utf-8") as f:
        doc = f.read()
    for key, fn in SECTIONS.items():
        marker = "{{" + key + "}}"
        if marker in doc:
            try:
                doc = doc.replace(marker, fn())
            except Exception as exc:                       # noqa: BLE001
                doc = doc.replace(marker, f"*Section failed to generate: {exc}*")
                print(f"  [warn] {key}: {type(exc).__name__}: {exc}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[report] wrote {os.path.relpath(args.out, ROOT)} "
          f"({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
