"""Recompute the paired tests behind table-pinca.tex and table-vsnca.tex.

The benchmark run tests every architecture against the per-phenomenon winner only. The
two tables that frame PI-NCA need different pairs -- each PI-NCA model against the
strongest baseline outside its family, and PI-NCA against the standard NCA on every
phenomenon -- so they are recomputed here from the per-initial-condition errors the run
saved, with the same paired test and Holm correction the run itself uses.

    PYTHONPATH=src python paper_updated/pinca_tests.py [results_dir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from pinca_jax import stats

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")

# (phenomenon, protocol file, PI-NCA model, baseline). Heat and CH use the ten-seed files.
LEADS = [
    ("shallow_water", "full", "pi_nca", "fno"),
    ("shallow_water", "full", "mc_flux_nca", "fno"),
    ("shallow_water", "full", "mc_flux_nca", "unet"),
    ("cahn_hilliard", "headline", "bounded_multiscale_nca", "resnet"),
    ("cahn_hilliard", "headline", "bounded_cons_nca", "fno"),
    ("allen_cahn", "full", "spectral_flux_nca", "fno"),
    ("heat", "headline", "multiscale_flux_nca", "unet"),
    ("heat", "headline", "multiscale_flux_nca", "resnet"),
    ("adv_diff", "full", "pi_nca", "unet"),
    ("adv_diff", "full", "pi_nca", "resnet"),
]
PHENOMENA = ["heat", "adv_diff", "wave", "allen_cahn", "cahn_hilliard", "gray_scott",
             "shallow_water", "fitzhugh_nagumo", "nagumo", "navier_stokes"]


def per_ic(pde, proto, arch):
    with open(os.path.join(RES, f"bench_{pde}_{proto}.json"), encoding="utf-8") as f:
        return np.asarray(json.load(f)["results"][arch]["_per_ic_rel_l2"], float)


def family(pairs):
    """Paired tests for a list of (pde, proto, a, b), Holm-corrected as one family."""
    tests = [stats.paired_test(per_ic(p, f, a), per_ic(p, f, b)) for p, f, a, b in pairs]
    rej = stats.holm_bonferroni([t["p_value"] for t in tests])
    for (p, f, a, b), t, r in zip(pairs, tests, rej):
        verdict = ("a better" if r and t["significant"] and t["better"] == "a"
                   else "b better" if r and t["significant"] else "tie")
        print(f"{p:16s} {a:24s} {per_ic(p, f, a).mean():.4g}  vs  {b:10s} "
              f"{per_ic(p, f, b).mean():.4g}  p={t['p_value']:.2g}  "
              f"win={t['win_rate']:.2f}  n={t['n']}  {verdict}")


if __name__ == "__main__":
    print("table-pinca: PI-NCA models against the strongest outside baseline")
    family(LEADS)
    print("\ntable-vsnca: PI-NCA against the standard NCA, five seeds")
    family([(p, "full", "pi_nca", "plain_nca") for p in PHENOMENA])
