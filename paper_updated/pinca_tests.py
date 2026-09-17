"""Every number in the manuscript that the benchmark run did not tabulate directly.

The paper introduces two architectures, PI-NCA and MC-PI-NCA, and evaluates them against
the baselines. The run itself trained further conservative variants (dilated-perception
and spectral hybrids); they are not part of this paper, so every ranking here is
recomputed over the models the paper reports, from the saved per-initial-condition
errors, with the same paired test and Holm correction the run uses.

PI-NCA is reported in two configurations of one architecture: plain, and with the
parameter-free bound projection enabled (code name `bounded_cons_nca`, identical
parameters).

    PYTHONPATH=src python paper_updated/pinca_tests.py [results_dir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau

from pinca_jax import stats

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")

NAME = {"pi_nca": "PI-NCA", "bounded_cons_nca": "PI-NCA (bounded)", "mc_flux_nca": "MC-PI-NCA",
        "plain_nca": "NCA", "fno": "FNO", "fno_small": "FNO-S", "resnet": "ResNet",
        "resnet_iso": "ResNet-S", "unet": "U-Net", "unet_iso": "U-Net-S", "identity": "Identity"}
PROPOSED = ("pi_nca", "bounded_cons_nca", "mc_flux_nca")
PHENOMENA = ["shallow_water", "cahn_hilliard", "heat", "adv_diff", "wave", "navier_stokes",
             "allen_cahn", "gray_scott", "nagumo", "fitzhugh_nagumo"]
TEN_SEED = {"heat", "cahn_hilliard", "navier_stokes"}


def load(pde, proto):
    with open(os.path.join(RES, f"bench_{pde}_{proto}.json"), encoding="utf-8") as f:
        return {k: v for k, v in json.load(f)["results"].items() if k in NAME}


def per_ic(r, a):
    return np.asarray(r[a]["_per_ic_rel_l2"], float)


def mean(v):
    return v["mean"] if isinstance(v, dict) else v


def family(title, rows):
    """rows: (label, a_samples, b_samples). Holm over the family."""
    tests = [stats.paired_test(a, b) for _, a, b in rows]
    rej = stats.holm_bonferroni([t["p_value"] for t in tests])
    print(f"\n== {title}")
    for (lab, a, b), t, r in zip(rows, tests, rej):
        v = ("a better" if r and t["significant"] and t["better"] == "a"
             else "b better" if r and t["significant"] else "tie")
        print(f"  {lab:48s} {a.mean():.4g} vs {b.mean():.4g}  p={t['p_value']:.2g} "
              f"win={t['win_rate']:.2f} n={t['n']}  {v}")


def rankings():
    print("== rankings over the reported models (five seeds; ten-seed where marked)")
    for pde in PHENOMENA:
        for proto in (["full", "headline"] if pde in TEN_SEED else ["full"]):
            r = load(pde, proto)
            order = sorted(r, key=lambda a: per_ic(r, a).mean())
            rank = {a: i + 1 for i, a in enumerate(order)}
            best_prop = min(PROPOSED, key=lambda a: per_ic(r, a).mean())
            print(f"  {pde:16s} {proto:8s} n_models={len(r)} best={NAME[order[0]]} "
                  f"{per_ic(r, order[0]).mean():.4g} | best proposed={NAME[best_prop]} "
                  f"{per_ic(r, best_prop).mean():.4g} (rank {rank[best_prop]}) | floor "
                  f"{per_ic(r, 'identity').mean():.4g}")
            print("     order:", ", ".join(f"{NAME[a]} {per_ic(r, a).mean():.3g}" for a in order))


def main():
    rankings()
    sw, ch, heat, adv = (load("shallow_water", "full"), load("cahn_hilliard", "headline"),
                         load("heat", "headline"), load("adv_diff", "full"))
    family("table-pinca: proposed models against the strongest baselines", [
        ("shallow water: PI-NCA vs FNO", per_ic(sw, "pi_nca"), per_ic(sw, "fno")),
        ("shallow water: MC-PI-NCA vs FNO", per_ic(sw, "mc_flux_nca"), per_ic(sw, "fno")),
        ("shallow water: MC-PI-NCA vs U-Net", per_ic(sw, "mc_flux_nca"), per_ic(sw, "unet")),
        ("CH: PI-NCA (bounded) vs ResNet", per_ic(ch, "bounded_cons_nca"), per_ic(ch, "resnet")),
        ("CH: PI-NCA (bounded) vs FNO", per_ic(ch, "bounded_cons_nca"), per_ic(ch, "fno")),
        ("heat: MC-PI-NCA vs ResNet", per_ic(heat, "mc_flux_nca"), per_ic(heat, "resnet")),
        ("heat: MC-PI-NCA vs ResNet-S", per_ic(heat, "mc_flux_nca"), per_ic(heat, "resnet_iso")),
        ("adv-diff: PI-NCA vs U-Net", per_ic(adv, "pi_nca"), per_ic(adv, "unet")),
        ("adv-diff: PI-NCA vs ResNet", per_ic(adv, "pi_nca"), per_ic(adv, "resnet")),
    ])
    family("table-vsnca: PI-NCA vs the standard NCA (five seeds)",
           [(p, per_ic(load(p, "full"), "pi_nca"), per_ic(load(p, "full"), "plain_nca"))
            for p in PHENOMENA])
    ac = load("allen_cahn", "full")
    family("A1 at the PI-NCA backbone: bound projection on vs off", [
        ("CH (ten seeds)", per_ic(ch, "bounded_cons_nca"), per_ic(ch, "pi_nca")),
        ("Allen-Cahn (five seeds)", per_ic(ac, "bounded_cons_nca"), per_ic(ac, "pi_nca")),
    ])
    print("\n== diagnostic panels (mean over seeds)")
    for label, r in (("heat ten-seed", heat), ("CH ten-seed", ch), ("allen_cahn", ac),
                     ("shallow_water", sw), ("adv_diff", adv), ("wave", load("wave", "full"))):
        for a in NAME:
            v = r[a]
            print(f"  {label:14s} {NAME[a]:17s} relL2={per_ic(r, a).mean():.4g} "
                  f"psnr={mean(v['psnr']):.1f} hf={mean(v['highfreq_err_frac']):.3f} "
                  f"mass={mean(v['conservation_err']):.2e} growth={mean(v['error_growth_ratio']):.2f} "
                  f"train={mean(v['train_wall_s']):.1f}s infer={mean(v['infer_s_per_step'])*1e3:.2f}ms "
                  f"params={mean(v['params']):.0f}")
    print("\n== rank stability over the reported models present in each sweep")
    for pde in ("heat", "cahn_hilliard", "navier_stokes"):
        with open(os.path.join(RES, f"scaling_{pde}.json"), encoding="utf-8") as f:
            sc = json.load(f)["results"]
        for axis, d in sc.items():
            vals = list(d["values"])
            ends = [{a: m["mean"] for a, m in d["values"][v].items() if a in NAME}
                    for v in (vals[0], vals[-1])]
            archs = sorted(set(ends[0]) & set(ends[1]))
            tau = kendalltau([ends[0][a] for a in archs], [ends[1][a] for a in archs])[0]
            winners = [NAME[min(e, key=e.get)] for e in ends]
            allw = [NAME[min((a for a in d["values"][v] if a in NAME),
                             key=lambda a: d["values"][v][a]["mean"])] for v in vals]
            print(f"  {pde:14s} {axis:8s} {vals[0]}->{vals[-1]} tau={tau:.2f} "
                  f"winner {winners[0]} -> {winners[1]}  all: {allw}  archs={len(archs)}")
    print("\n== 3-D (single seed)")
    for pde in ("heat", "adv_diff", "allen_cahn", "nagumo", "gray_scott", "fitzhugh_nagumo"):
        with open(os.path.join(RES, f"bench3d_{pde}.json"), encoding="utf-8") as f:
            r = json.load(f)["results"]
        row = {a: mean(r[a]["rel_l2"]) for a in r if a in NAME}
        print(f"  {pde:16s} " + ", ".join(f"{NAME[a]} {e:.3g}" for a, e in sorted(row.items(), key=lambda x: x[1])))


if __name__ == "__main__":
    main()
