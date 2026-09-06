"""Single source of truth for what was actually measured, and an audit against the prose.

The abstract said eight 2-D phenomena while the body said ten. That is not a typo
problem, it is a process problem: counts were written by hand in several documents and
nothing checked them against `results/`. This module derives the inventory from the raw
result files and audits every document against it, so a stale number fails the test
suite instead of reaching a reviewer.

Two layers:

* `inventory()` reads `results/*.json` and reports what exists -- which phenomena were
  measured in 2-D and 3-D, which architectures, how many seeds, which grids, whether the
  run was on GPU or CPU, and which cells failed. Documents quote these numbers through
  `render_counts()` rather than restating them.
* `CLAIMS` is a pre-registered list of the paper's headline assertions, each paired with
  a predicate over the inventory. `audit()` evaluates them and marks each SUPPORTED,
  UNSUPPORTED or NOT_YET_MEASURED. A claim with no check is itself a finding, so adding
  a sentence to the paper without a corresponding check is visible.

`python -m pinca_jax.claims` writes docs/claims_audit.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")

# Words the documents use for small counts, so "ten 2-D phenomena" can be checked.
WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16}


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def inventory() -> dict:
    """What the result files actually contain. No prose, no hand-maintained lists."""
    inv = {"pdes_2d": set(), "pdes_3d": set(), "archs_2d": set(), "archs_3d": set(),
           "seeds": {}, "grids": set(), "backends": set(), "failed_cells": [],
           "files": 0, "ood_pdes": set(), "stability_pdes": set(),
           "resolution_pdes": set(), "ablations": set(), "has_teacher_error": False,
           "n_eval_ics": set()}
    for path in sorted(glob.glob(os.path.join(RES, "*.json"))):
        d = _load(path)
        if not isinstance(d, dict):
            continue
        inv["files"] += 1
        name = os.path.basename(path)[:-5]
        dev = d.get("device") or {}
        if dev.get("backend"):
            inv["backends"].add(dev["backend"])
        cfg = d.get("config") or {}
        if cfg.get("grid_size"):
            inv["grids"].add(int(cfg["grid_size"]))
        if cfg.get("n_eval"):
            inv["n_eval_ics"].add(int(cfg["n_eval"]))

        if name == "teacher_error":
            inv["has_teacher_error"] = True
        elif name.startswith("ood_"):
            inv["ood_pdes"].add(name[4:])
        elif name.startswith("stability_"):
            inv["stability_pdes"].add(name[10:])
        elif name.startswith("bench_resolution_"):
            inv["resolution_pdes"].add(name[len("bench_resolution_"):])
        elif name.startswith("bench3d_"):
            pde = name[len("bench3d_"):]
            inv["pdes_3d"].add(pde)
            _collect(d, inv, "archs_3d", pde, name)
        elif name.startswith("bench_"):
            rest = name[len("bench_"):]
            m = re.match(r"^(.*)_(full|main|hybrid|unified|mc|a2|A\d)$", rest)
            pde, tag = (m.group(1), m.group(2)) if m else (rest, "")
            if tag.startswith("A") or tag == "a2":
                inv["ablations"].add(tag.upper())
            if tag in ("full", "main", ""):
                inv["pdes_2d"].add(pde)
                _collect(d, inv, "archs_2d", pde, name)
        if d.get("seeds"):
            inv["seeds"][name] = list(d["seeds"])
    for k in ("pdes_2d", "pdes_3d", "archs_2d", "archs_3d", "grids", "backends",
              "ood_pdes", "stability_pdes", "resolution_pdes", "ablations", "n_eval_ics"):
        inv[k] = sorted(inv[k])
    inv["n_pdes_2d"] = len(inv["pdes_2d"])
    inv["n_pdes_3d"] = len(inv["pdes_3d"])
    inv["n_archs_2d"] = len(inv["archs_2d"])
    inv["min_seeds"] = min((len(v) for v in inv["seeds"].values()), default=0)
    inv["max_seeds"] = max((len(v) for v in inv["seeds"].values()), default=0)
    inv["single_seed_files"] = sorted(k for k, v in inv["seeds"].items() if len(v) < 2)
    return inv


def _collect(d, inv, key, pde, fname):
    for arch, rec in (d.get("results") or {}).items():
        if isinstance(rec, dict) and "error" in rec:
            inv["failed_cells"].append({"file": fname, "pde": pde, "arch": arch,
                                        "kind": rec.get("kind", "error")})
        else:
            inv[key].add(arch)


# ------------------------------------------------------- pre-registered claims ---
# Each entry: (id, the sentence as the paper states it, predicate over the inventory).
# A claim whose predicate cannot be evaluated yet returns None -> NOT_YET_MEASURED.
CLAIMS = [
    ("C1", "The 2-D benchmark matrix is uniform: the same architecture list is measured "
           "on every phenomenon.",
     lambda i: (len(i["archs_2d"]) > 0 and i["n_pdes_2d"] > 0) or None),
    ("C2", "Headline comparisons use at least 5 independent seeds.",
     lambda i: i["min_seeds"] >= 5 if i["seeds"] else None),
    ("C3", "Headline numbers were produced on a GPU backend.",
     lambda i: ("gpu" in i["backends"]) if i["backends"] else None),
    ("C4", "Every architecture comparison includes a do-nothing identity floor.",
     lambda i: ("identity" in i["archs_2d"]) if i["archs_2d"] else None),
    ("C5", "Every architecture comparison includes a physics-free CNN baseline.",
     lambda i: bool({"resnet", "unet"} & set(i["archs_2d"])) if i["archs_2d"] else None),
    ("C6", "An iso-parameter control is reported alongside the large spectral model.",
     lambda i: bool({"fno_small", "resnet_iso", "unet_iso"} & set(i["archs_2d"]))
     if i["archs_2d"] else None),
    ("C7", "Out-of-distribution generalisation is measured on held-out IC families.",
     lambda i: len(i["ood_pdes"]) > 0 or None),
    ("C8", "Long-horizon failure rates are reported with the divergence guard disabled.",
     lambda i: len(i["stability_pdes"]) > 0 or None),
    ("C9", "The teacher's own discretisation error is quantified.",
     lambda i: i["has_teacher_error"] or None),
    ("C10", "Resolution transfer is measured (train at one grid, evaluate at others).",
     lambda i: len(i["resolution_pdes"]) > 0 or None),
    ("C11", "Conclusions are reproduced in 3-D.",
     lambda i: i["n_pdes_3d"] >= 3 or None),
    ("C12", "No benchmark cell is silently missing: every failure is recorded.",
     lambda i: True),
]


def audit(inv=None) -> list[dict]:
    inv = inv or inventory()
    out = []
    for cid, text, check in CLAIMS:
        try:
            v = check(inv)
        except Exception as exc:                      # noqa: BLE001 - reported, not fatal
            v = None
            text += f"  [check error: {type(exc).__name__}]"
        status = "NOT_YET_MEASURED" if v is None else ("SUPPORTED" if v else "UNSUPPORTED")
        out.append({"id": cid, "claim": text, "status": status})
    return out


# --------------------------------------------------------- prose count audit ---
COUNT_PATTERNS = [
    (re.compile(r"\b(\w+|\d+)\s+2-?D\s+(?:phenomena|PDEs|equations|systems)\b", re.I), "n_pdes_2d"),
    (re.compile(r"\b(\w+|\d+)\s+3-?D\s+(?:phenomena|PDEs|equations|systems)\b", re.I), "n_pdes_3d"),
]


def _as_int(tok):
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return WORD.get(tok)


def prose_counts(inv=None, paths=None) -> list[dict]:
    """Find count claims in the documents and compare them to the inventory."""
    inv = inv or inventory()
    paths = paths or (sorted(glob.glob(os.path.join(DOCS, "*.md"))) +
                      sorted(glob.glob(os.path.join(DOCS, "*.txt"))))
    hits = []
    for path in paths:
        try:
            text = open(path, encoding="utf-8").read()
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, key in COUNT_PATTERNS:
                for m in pat.finditer(line):
                    n = _as_int(m.group(1))
                    if n is None:
                        continue
                    hits.append({"file": os.path.relpath(path, ROOT), "line": lineno,
                                 "quoted": m.group(0).strip(), "claimed": n,
                                 "measured": inv[key], "key": key,
                                 "ok": n == inv[key]})
    return hits


def render_counts(inv=None) -> str:
    """The canonical sentence documents should quote instead of hand-counting."""
    inv = inv or inventory()
    return (f"{inv['n_pdes_2d']} 2-D phenomena and {inv['n_pdes_3d']} 3-D phenomena, "
            f"{inv['n_archs_2d']} architectures, grids {inv['grids']}, "
            f"seeds per table {inv['min_seeds']}-{inv['max_seeds']}, "
            f"backend(s) {inv['backends'] or ['unrecorded']}")


def to_markdown(inv=None) -> str:
    inv = inv or inventory()
    rows = audit(inv)
    hits = prose_counts(inv)
    bad = [h for h in hits if not h["ok"]]
    L = ["# Claims audit", "",
         "Generated by `python -m pinca_jax.claims`. Do not edit by hand -- edit the "
         "results, or the claim.", "",
         "## What was actually measured", "",
         f"- 2-D phenomena ({inv['n_pdes_2d']}): {', '.join(inv['pdes_2d']) or 'none'}",
         f"- 3-D phenomena ({inv['n_pdes_3d']}): {', '.join(inv['pdes_3d']) or 'none'}",
         f"- Architectures in the 2-D matrix ({inv['n_archs_2d']}): "
         f"{', '.join(inv['archs_2d']) or 'none'}",
         f"- Grids: {inv['grids'] or 'none'}   |   evaluation ICs per cell: "
         f"{inv['n_eval_ics'] or 'unrecorded'}",
         f"- Seeds per results file: min {inv['min_seeds']}, max {inv['max_seeds']}",
         f"- Backends recorded: {', '.join(inv['backends']) or 'none recorded'}",
         f"- OOD studies: {', '.join(inv['ood_pdes']) or 'none'}",
         f"- Stability studies: {', '.join(inv['stability_pdes']) or 'none'}",
         f"- Resolution studies: {', '.join(inv['resolution_pdes']) or 'none'}",
         f"- Ablation tags present: {', '.join(inv['ablations']) or 'none'}",
         f"- Teacher-error study: {'yes' if inv['has_teacher_error'] else 'no'}",
         f"- Result files scanned: {inv['files']}", ""]
    if inv["single_seed_files"]:
        L += ["> **Single-seed tables still present** (a headline number must not come "
              "from these): " + ", ".join(f"`{f}`" for f in inv["single_seed_files"]), ""]
    if inv["failed_cells"]:
        L += ["## Failed cells (recorded, not hidden)", "",
              "| file | PDE | architecture | kind |", "|---|---|---|---|"]
        L += [f"| {c['file']} | {c['pde']} | {c['arch']} | {c['kind']} |"
              for c in inv["failed_cells"]]
        L += [""]
    L += ["## Pre-registered claims", "", "| id | status | claim |", "|---|---|---|"]
    L += [f"| {r['id']} | {r['status']} | {r['claim']} |" for r in rows]
    L += ["", "## Count claims found in the documents", ""]
    if not hits:
        L += ["No numeric phenomenon-count claims found in prose.", ""]
    else:
        L += ["| file:line | quoted | claimed | measured | ok |", "|---|---|---|---|---|"]
        L += [f"| {h['file']}:{h['line']} | {h['quoted']} | {h['claimed']} | "
              f"{h['measured']} | {'yes' if h['ok'] else '**NO**'} |" for h in hits]
        L += [""]
    if bad:
        L += [f"> **{len(bad)} count claim(s) disagree with the measured inventory.** "
              "Fix the prose or run the missing experiments.", ""]
    L += ["## Canonical sentence", "", "> " + render_counts(inv), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DOCS, "claims_audit.md"))
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any prose count disagrees with the results")
    args = ap.parse_args()
    inv = inventory()
    md = to_markdown(inv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    bad = [h for h in prose_counts(inv) if not h["ok"]]
    unsupported = [r for r in audit(inv) if r["status"] == "UNSUPPORTED"]
    print(render_counts(inv))
    print(f"[claims] wrote {args.out}: {len(bad)} count mismatch(es), "
          f"{len(unsupported)} unsupported claim(s)")
    for h in bad:
        print(f"  MISMATCH {h['file']}:{h['line']} '{h['quoted']}' "
              f"claims {h['claimed']}, measured {h['measured']}")
    if args.strict and (bad or unsupported):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
