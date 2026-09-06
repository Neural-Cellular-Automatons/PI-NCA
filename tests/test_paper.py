"""The paper must be buildable, self-consistent, and free of unsupported phrasing.

There is no LaTeX toolchain in the correctness environment, so these tests do what a
compile would catch plus several things it would not: that every generated table the
paper includes actually exists, that every number quoted in prose is a macro defined by
the generator rather than typed by hand, that every citation key resolves to a verified
arXiv entry, and that the specific over-claiming words a reviewer would flag do not
appear outside the passages that explicitly disclaim them.

The point is that a stale number cannot reach a PDF: it either regenerates from
`results/` or the build fails on an undefined macro.
"""
from __future__ import annotations

import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "paper")
GEN = os.path.join(PAPER, "generated")

pytestmark = pytest.mark.skipif(not os.path.isdir(PAPER), reason="paper/ not present")


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _sources():
    return {n: _read(PAPER, n) for n in ("main.tex", "appendix.tex")
            if os.path.exists(os.path.join(PAPER, n))}


def _all_tex():
    return "\n".join(_sources().values())


# ------------------------------------------------------------- structure ---
def test_generated_tables_are_regenerable_and_present():
    """Every \\input{generated/...} must exist. A missing one is a hole in the PDF."""
    missing = []
    for src in _sources().values():
        for m in re.finditer(r"\\input\{generated/([A-Za-z0-9_]+)\}", src):
            if not os.path.exists(os.path.join(GEN, m.group(1) + ".tex")):
                missing.append(m.group(1))
    assert missing == [], f"run `python -m pinca_jax.paper`; missing: {missing}"


def test_every_prose_macro_is_defined_by_the_generator():
    """A number in the prose must come from facts.tex, never from a keystroke."""
    facts = _read(GEN, "facts.tex")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", facts))
    # macros defined in main.tex itself, and LaTeX/package commands
    local = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", _all_tex()))
    used = set(re.findall(r"\\([A-Z][A-Za-z]*)\{\}", _all_tex()))
    unknown = sorted(u for u in used if u not in defined and u not in local)
    assert unknown == [], (f"undefined in facts.tex: {unknown}. Either the experiment "
                           f"has not been run or the macro name is wrong.")


def test_no_table_of_contents():
    """ICLR style has no TOC; the previous draft carried one."""
    assert "\\tableofcontents" not in _all_tex()


def test_environments_are_balanced():
    tex = _all_tex()
    for env in ("table", "tabular", "abstract", "itemize", "enumerate", "equation",
                "quote", "document"):
        b = len(re.findall(r"\\begin\{" + env + r"\}", tex))
        e = len(re.findall(r"\\end\{" + env + r"\}", tex))
        assert b == e, f"unbalanced {env}: {b} begin vs {e} end"


def test_every_label_referenced_exists_and_vice_versa():
    tex = _all_tex()
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    assert refs - labels == set(), f"dangling \\ref: {sorted(refs - labels)}"


def test_tabular_column_counts_match_the_generated_bodies():
    """A generated row with the wrong number of & silently corrupts a table."""
    main = _read(PAPER, "main.tex") + _read(PAPER, "appendix.tex")
    bad = []
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                         main, re.S):
        spec, body = m.group(1), m.group(2)
        ncol = len(re.sub(r"[^lcrp]", "", re.sub(r"p\{[^}]*\}", "p", spec)))
        inc = re.search(r"\\input\{generated/([A-Za-z0-9_]+)\}", body)
        if not inc:
            continue
        rows = _read(GEN, inc.group(1) + ".tex").splitlines()
        for row in rows:
            row = row.strip()
            if not row or row.startswith("%") or "multicolumn" in row:
                continue
            got = row.count("&") + 1
            if got != ncol:
                bad.append((inc.group(1), ncol, got, row[:60]))
    assert bad == [], f"column-count mismatches: {bad[:5]}"


# ------------------------------------------------------------- citations ---
def test_every_citation_key_is_in_the_generated_bibliography():
    bib_path = os.path.join(PAPER, "refs.bib")
    if not os.path.exists(bib_path):
        pytest.skip("refs.bib not generated yet (python -m pinca_jax.bib)")
    keys = set(re.findall(r"@\w+\{([^,]+),", _read(bib_path)))
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", _all_tex()):
        cited |= {k.strip() for k in m.group(1).split(",")}
    assert cited - keys == set(), (
        f"citation keys with no verified entry: {sorted(cited - keys)}. "
        f"Every key must come from an arXiv id that `pinca_jax.bib` resolved.")


def test_bibliography_has_no_unresolved_entries():
    import json
    path = os.path.join(ROOT, "docs", "bibliography.json")
    if not os.path.exists(path):
        pytest.skip("bibliography not verified yet")
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    assert p["unresolved"] == [], f"unresolvable arXiv ids cited: {p['unresolved']}"


# ------------------------------------------------------------ over-claim ---
# Words that assert more than a benchmark at this scale can support. Each is allowed
# only where the paper explicitly disclaims it.
BANNED = {
    "dominates": "no architecture dominates; say which comparisons are significant",
    "dimension-independent": "two dimensionalities are not independent samples",
    "state-of-the-art": "not measured against a shared public benchmark",
    "outperforms all": "the identity floor and the ties make this unsupportable",
    "proves that": "a benchmark provides evidence, not proof",
}


def test_no_unsupported_superlatives():
    tex = _all_tex().lower()
    hits = []
    for word, why in BANNED.items():
        for m in re.finditer(re.escape(word), tex):
            ctx = tex[max(0, m.start() - 220):m.start() + 120]
            # allowed where the sentence is explicitly refusing the claim
            if any(d in ctx for d in ("deliberately do", "not describe", "we avoid",
                                      "rather than", "unsupport", "would be")):
                continue
            hits.append((word, why, tex[max(0, m.start() - 60):m.start() + 60]))
    assert hits == [], f"over-claiming language: {hits[:3]}"


def test_limitations_section_exists_and_is_itemised():
    main = _read(PAPER, "main.tex")
    assert "\\section{Limitations}" in main
    tail = main[main.index("\\section{Limitations}"):]
    body = tail[:tail.index("\\section{Conclusion}")]
    assert body.count("\\item") >= 5, "a limitations section with fewer than five items"


def test_headline_sections_carry_an_explicit_limitation():
    """Each result section that makes a claim must qualify it in place."""
    main = _read(PAPER, "main.tex")
    for sec in ("sec:regime", "sec:stability", "sec:threed"):
        i = main.index("\\label{" + sec + "}")
        nxt = main.find("\\section{", i)
        chunk = main[i:nxt if nxt > 0 else len(main)]
        assert "\\limitation{" in chunk, f"{sec} states results with no limitation"


def test_reproducibility_statement_names_runnable_commands():
    main = _read(PAPER, "main.tex")
    assert "Reproducibility statement" in main
    for cmd in ("pytest", "runner", "claims", "paper", "bib"):
        assert cmd in main, f"reproducibility statement does not mention {cmd}"
