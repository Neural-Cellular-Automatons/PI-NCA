"""Plot every benchmark in results/ - one command, all figures.

Reads the JSON the benchmark drivers already write (no re-training) and renders:

  bench_accuracy_2d.png      rel-L2 per architecture, grouped by phenomenon (log y)
  bench_psnr_2d.png          PSNR per architecture
  bench_conservation_2d.png  mass-conservation error
  bench_train_time.png       training wall-clock
  bench_throughput.png       inference throughput (cells/s)
  bench_error_growth.png     rel-L2 at T/4, T/2, 3T/4, T - error-growth profiles
  bench_accuracy_vs_cost.png params vs rel-L2 (Pareto view)
  bench_regime_map.png       per-phenomenon normalised rel-L2 heatmap (who wins where)
  bench_accuracy_3d.png      the 3-D suite
  bench_ablation_A4/A5.png   conservation on/off, perception size
  bench_resolution_*.png     train-grid x eval-grid transfer heatmaps

Run:  python -m pinca_jax.plots            # everything found in results/
      python -m pinca_jax.plots --out some/dir
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures", "bench")

# stable colour per architecture so every figure reads the same way
ARCH_COLORS = {
    "plain_nca": "#4C72B0", "pi_nca": "#DD8452", "multiscale_flux_nca": "#55A868",
    "spectral_flux_nca": "#C44E52", "mc_flux_nca": "#8172B3", "fno": "#937860",
    "fno_small": "#DA8BC3", "bounded_cons_nca": "#8C8C8C",
    "bounded_multiscale_nca": "#CCB974", "abl_flux": "#4C72B0",
    "abl_residual": "#C44E52", "abl_k3": "#4C72B0", "abl_k5": "#DD8452",
    "abl_multiscale": "#55A868",
}
GROWTH_KEYS = [("rel_l2_t_q1", "T/4"), ("rel_l2_t_half", "T/2"),
               ("rel_l2_t_q3", "3T/4"), ("rel_l2_t_final", "T")]


def _color(a):
    return ARCH_COLORS.get(a, "#666666")


def _load(pattern):
    """{label: payload} for every results/<pattern> file."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(path, encoding="utf-8") as f:
            out[os.path.basename(path)[:-5]] = json.load(f)
    return out


def _mean(res, arch, key):
    """Mean of one metric, or None when this arch did not report it."""
    cell = res.get(arch, {}).get(key)
    return None if cell is None else cell["mean"]


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _arch_order(a):
    return list(ARCH_COLORS).index(a) if a in ARCH_COLORS else 99


# ---------- grouped bar: one metric, phenomena x architectures ---------- #
def grouped_bar(runs, key, title, ylabel, name, log=True):
    """runs: {pde: results-dict}."""
    pdes = [p for p in sorted(runs)
            if any(_mean(runs[p], a, key) is not None for a in runs[p])]
    if not pdes:
        return
    archs = sorted({a for p in pdes for a in runs[p]}, key=_arch_order)
    fig, ax = plt.subplots(figsize=(1.9 * len(pdes) + 3, 4.4))
    w = 0.8 / len(archs)
    for j, a in enumerate(archs):
        xs, ys = [], []
        for i, p in enumerate(pdes):
            v = _mean(runs[p], a, key)
            if v is not None and (not log or v > 0):
                xs.append(i + j * w - 0.4 + w / 2)
                ys.append(v)
        if xs:
            ax.bar(xs, ys, width=w * 0.92, label=a, color=_color(a))
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(pdes)))
    ax.set_xticklabels(pdes, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, name)


# ---------- error-growth profiles ---------- #
def error_growth(runs, name="bench_error_growth.png"):
    pdes = [p for p in sorted(runs)
            if any(_mean(runs[p], a, "rel_l2_t_final") is not None for a in runs[p])]
    if not pdes:
        return
    ncol = min(4, len(pdes))
    nrow = int(np.ceil(len(pdes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
    xs = list(range(len(GROWTH_KEYS)))
    for i, p in enumerate(pdes):
        ax = axes[i // ncol][i % ncol]
        for a in sorted(runs[p], key=_arch_order):
            ys = [_mean(runs[p], a, k) for k, _ in GROWTH_KEYS]
            if any(y is None or y <= 0 for y in ys):
                continue
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=_color(a), label=a)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in GROWTH_KEYS])
        ax.set_title(p, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if i % ncol == 0:
            ax.set_ylabel("rel-L2")
    for k in range(len(pdes), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    seen, handles, labels = set(), [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.suptitle("Error growth over the rollout horizon (log rel-L2)")
    fig.tight_layout()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, -0.05))
    _save(fig, name)


# ---------- accuracy vs cost (Pareto) ---------- #
def accuracy_vs_cost(runs, name="bench_accuracy_vs_cost.png"):
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    plotted = False
    for i, p in enumerate(sorted(runs)):
        for a in runs[p]:
            x, y = _mean(runs[p], a, "params"), _mean(runs[p], a, "rel_l2")
            if x is None or y is None or x <= 0 or y <= 0:
                continue
            ax.scatter(x, y, color=_color(a), marker=markers[i % len(markers)],
                       s=52, edgecolor="white", linewidth=0.6, zorder=3)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("rel-L2 @T (lower better)")
    ax.set_title("Accuracy vs model size - colour = architecture, marker = phenomenon")
    ax.grid(alpha=0.3, which="both")
    arch_h = [plt.Line2D([], [], marker="o", ls="", color=_color(a), label=a)
              for a in sorted({a for p in runs for a in runs[p]}, key=_arch_order)]
    pde_h = [plt.Line2D([], [], marker=markers[i % len(markers)], ls="", color="#444",
                        label=p) for i, p in enumerate(sorted(runs))]
    ax.legend(handles=arch_h + pde_h, fontsize=7, ncol=2, loc="best")
    _save(fig, name)


# ---------- regime map: normalised rel-L2 per phenomenon ---------- #
def regime_map(runs, name="bench_regime_map.png"):
    pdes = [p for p in sorted(runs) if any(_mean(runs[p], a, "rel_l2") for a in runs[p])]
    archs = sorted({a for p in pdes for a in runs[p]}, key=_arch_order)
    if not pdes or not archs:
        return
    M = np.full((len(archs), len(pdes)), np.nan)
    for j, p in enumerate(pdes):
        vals = {a: _mean(runs[p], a, "rel_l2") for a in archs}
        finite = [v for v in vals.values() if v and v > 0]
        if not finite:
            continue
        best = min(finite)
        for i, a in enumerate(archs):
            if vals[a] and vals[a] > 0:
                M[i, j] = vals[a] / best          # 1.0x == winner for that phenomenon
    fig, ax = plt.subplots(figsize=(1.05 * len(pdes) + 3.6, 0.55 * len(archs) + 2.4))
    im = ax.imshow(np.log10(M), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pdes)))
    ax.set_xticklabels(pdes, rotation=30, ha="right")
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(archs)
    for i in range(len(archs)):
        for j in range(len(pdes)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}x", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="log10(rel-L2 / best for that PDE)")
    ax.set_title("Regime map - 1.0x is the winner for that phenomenon")
    _save(fig, name)


# ---------- resolution transfer heatmaps ---------- #
def resolution_maps():
    for label, payload in _load("bench_resolution_*.json").items():
        pde = label.replace("bench_resolution_", "")
        grids, res = payload["grids"], payload["results"]
        archs = list(res)
        if not archs:
            continue
        fig, axes = plt.subplots(1, len(archs), figsize=(3.6 * len(archs), 3.4),
                                 squeeze=False)
        for k, a in enumerate(archs):
            M = np.array([[res[a][f"{gt}->{ge}"] for ge in grids] for gt in grids])
            ax = axes[0][k]
            im = ax.imshow(np.log10(M), cmap="viridis_r", aspect="auto")
            ax.set_xticks(range(len(grids)))
            ax.set_xticklabels([str(g) for g in grids])
            ax.set_yticks(range(len(grids)))
            ax.set_yticklabels([str(g) for g in grids])
            ax.set_xlabel("eval grid")
            ax.set_title(a, fontsize=10)
            if k == 0:
                ax.set_ylabel("train grid")
            for i in range(len(grids)):
                for j in range(len(grids)):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                            fontsize=7, color="white")
            fig.colorbar(im, ax=ax, label="log10 rel-L2")
        fig.suptitle(f"{pde} - resolution transfer (train grid -> eval grid)")
        fig.tight_layout()
        _save(fig, f"bench_resolution_{pde}.png")


def _runs_from(pattern, prefix, suffix):
    """results/<pattern> -> {pde: results-dict}, stripping the filename affixes."""
    runs = {}
    for label, payload in _load(pattern).items():
        pde = label[len(prefix):]
        if suffix and pde.endswith(suffix):
            pde = pde[: -len(suffix)]
        if "results" in payload:
            runs[pde] = payload["results"]
    return runs


def main():
    global FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir (default docs/figures/bench)")
    args = ap.parse_args()
    if args.out:
        FIG = args.out
    print(f"[plots] reading {os.path.abspath(RES)}")
    print(f"[plots] writing {os.path.abspath(FIG)}")

    two_d = _runs_from("bench_*_full.json", "bench_", "_full")
    print(f"[plots] 2-D phenomena: {sorted(two_d) or 'none found'}")
    grouped_bar(two_d, "rel_l2", "Final-horizon accuracy (2-D suite)",
                "rel-L2 @T (lower better)", "bench_accuracy_2d.png")
    grouped_bar(two_d, "psnr", "Reconstruction quality (2-D suite)",
                "PSNR (dB, higher better)", "bench_psnr_2d.png", log=False)
    grouped_bar(two_d, "conservation_err", "Mass-conservation error (2-D suite)",
                "abs mass drift (lower better)", "bench_conservation_2d.png")
    grouped_bar(two_d, "train_wall_s", "Training cost (2-D suite)", "wall-clock s",
                "bench_train_time.png", log=False)
    grouped_bar(two_d, "throughput_cells_per_s", "Inference throughput (2-D suite)",
                "cells / s (higher better)", "bench_throughput.png")
    error_growth(two_d)
    accuracy_vs_cost(two_d)
    regime_map(two_d)

    three_d = _runs_from("bench3d_*.json", "bench3d_", "")
    print(f"[plots] 3-D phenomena: {sorted(three_d) or 'none found'}")
    grouped_bar(three_d, "rel_l2", "Final-horizon accuracy (3-D suite)",
                "rel-L2 @T (lower better)", "bench_accuracy_3d.png")

    for tag, title in (("A4", "A4 - conservation on/off (matched backbone)"),
                       ("A5", "A5 - perception / receptive-field size")):
        abl = _runs_from(f"bench_*_{tag}.json", "bench_", f"_{tag}")
        if abl:
            grouped_bar(abl, "rel_l2", title, "rel-L2 @T (lower better)",
                        f"bench_ablation_{tag}.png")

    resolution_maps()
    print("[plots] done.")


if __name__ == "__main__":
    main()
