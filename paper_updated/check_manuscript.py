"""Structural checks for the manuscript, standing in for a LaTeX build.

There is no TeX toolchain in the environment this manuscript was assembled in, so these
checks do what a compile would catch (missing \\input, dangling \\ref, unbalanced
environments, undefined citation keys) plus two things a compile would not: that no table
body disagrees with its column specification, and that the specific over-claiming phrases
the paper is careful to avoid have not crept back in.

    python check_manuscript.py        # exits non-zero on any failure
"""
from __future__ import annotations

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["main.tex", "appendix.tex", "intervention-figure.tex",
           "architecture-diagrams.tex"] + sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(HERE, "table-*.tex")))

# Phrasing the paper deliberately does not use, because the measurements do not support
# it. Each is allowed only inside a sentence that explicitly refuses the claim.
BANNED = {
    "dominates": "no architecture dominates; name the significant comparisons",
    "dimension-independent": "two dimensionalities are not independent samples",
    "state-of-the-art": "not measured against a shared public benchmark",
    "universally superior": "the whole point is that no prior is",
    "proves that": "a benchmark gives evidence, not proof",
}
DISCLAIMERS = ("deliberately do not", "not describe", "we avoid", "rather than",
               "unsupport", "would be", "neither a universal")


def read(name: str) -> str:
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def all_tex() -> str:
    return "\n".join(read(n) for n in SOURCES if os.path.exists(os.path.join(HERE, n)))


def check_inputs(tex, fail):
    for m in re.finditer(r"\\input\{([^}]+)\}", tex):
        name = m.group(1)
        path = os.path.join(HERE, name if name.endswith(".tex") else name + ".tex")
        if not os.path.exists(path):
            fail(f"\\input{{{name}}} does not exist")


def check_refs(tex, fail):
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for r in sorted(set(re.findall(r"\\ref\{([^}]+)\}", tex))):
        if r not in labels:
            fail(f"dangling \\ref{{{r}}}")


def check_citations(tex, fail):
    refs = read("references.tex")
    keys = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", refs))
    # Strip comments first: a citation list broken across lines with a trailing `%`
    # otherwise yields a key with the comment marker glued to it.
    clean = re.sub(r"(?<!\\)%.*", "", tex)
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", clean):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    for k in sorted(cited - keys):
        fail(f"citation key '{k}' has no \\bibitem in references.tex")
    unused = sorted(keys - cited)
    if unused:
        print(f"  note: {len(unused)} bibliography entries are not cited: "
              f"{', '.join(unused[:8])}{' ...' if len(unused) > 8 else ''}")


def check_environments(tex, fail):
    for env in ("table", "tabular", "tabularx", "abstract", "itemize", "enumerate",
                "equation", "document", "figure", "tikzpicture"):
        b = len(re.findall(r"\\begin\{" + env + r"\}", tex))
        e = len(re.findall(r"\\end\{" + env + r"\}", tex))
        if b != e:
            fail(f"unbalanced {env}: {b} begin vs {e} end")


def _brace_group(text: str, start: int):
    """Return the balanced {...} group beginning at `start`, and the index after it.

    A column specification contains braces of its own (`@{}`, `p{2cm}`), so a
    non-greedy regex captures the wrong thing and silently reports zero columns.
    """
    if start >= len(text) or text[start] != "{":
        return None, start
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None, start


def _ncols(spec: str) -> int:
    spec = re.sub(r"@\{[^{}]*\}", "", spec)          # inter-column material
    spec = re.sub(r"[><]\{[^{}]*\}", "", spec)       # array package modifiers
    spec = re.sub(r"[pmb]\{[^{}]*\}", "c", spec)     # fixed-width columns
    spec = re.sub(r"\*\{(\d+)\}\{([^{}]*)\}",
                  lambda m: m.group(2) * int(m.group(1)), spec)
    return len(re.sub(r"[^lcrX]", "", spec))


def check_table_widths(fail):
    """Every body row must have exactly (ncols - 1) alignment characters."""
    for name in SOURCES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path) or not name.startswith("table-"):
            continue
        body = read(name)
        # tabularx takes a width argument before the column spec; tabular does not.
        m = re.search(r"\\begin\{(tabularx|tabular)\}", body)
        if not m:
            continue
        pos = m.end()
        if m.group(1) == "tabularx":
            _, pos = _brace_group(body, pos)         # skip the width argument
        spec, _ = _brace_group(body, pos)
        if spec is None:
            continue
        n = _ncols(spec)
        for line in body.splitlines():
            s = line.strip()
            if (not s or s.startswith("%") or s.startswith("\\")
                    or "multicolumn" in s or "multirow" in s):
                continue
            if not s.endswith(r"\\"):
                continue
            got = s.count("&") + 1
            if got != n:
                fail(f"{name}: row has {got} cells, spec declares {n}: {s[:70]}")


def check_language(tex, fail):
    low = tex.lower()
    for word, why in BANNED.items():
        for m in re.finditer(re.escape(word), low):
            ctx = low[max(0, m.start() - 240):m.start() + 140]
            if any(d in ctx for d in DISCLAIMERS):
                continue
            fail(f"over-claiming phrase '{word}' ({why}): "
                 f"...{low[max(0, m.start()-50):m.start()+50]}...")


def check_required_sections(fail):
    main = read("main.tex")
    for needed in ("Reproducibility Statement", "AI Use Statement", "Ethics Statement",
                   "Discussion and Limitations"):
        if needed not in main:
            fail(f"missing required section: {needed}")
    tail = main[main.index("Discussion and Limitations"):]
    if "not measured" not in tail:
        fail("limitations do not state which evaluation axes were not measured")


def main():
    failures = []

    def fail(msg):
        failures.append(msg)

    tex = all_tex()
    check_inputs(tex, fail)
    check_refs(tex, fail)
    check_citations(tex, fail)
    check_environments(tex, fail)
    check_table_widths(fail)
    check_language(tex, fail)
    check_required_sections(fail)

    if failures:
        print(f"FAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    words = len(re.findall(r"\b\w+\b", read("main.tex")))
    print(f"OK: {len(SOURCES)} source files, ~{words} words in main.tex, "
          f"no structural problems found.")


if __name__ == "__main__":
    main()
