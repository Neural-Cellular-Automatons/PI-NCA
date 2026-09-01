"""Architecture diagrams for the documentation (matplotlib, no external tools).

Draws one block diagram per architecture into docs/figures/arch/*.png. The Mermaid
versions in docs/architecture_diagrams.md render only on GitHub; these are images, so
they embed in the PDF and open anywhere.

Run:  python -m pinca_jax.arch_figs
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures", "arch")

# role -> (fill, edge). Consistent across every diagram so shapes read at a glance.
STYLE = {
    "io":       ("#E8E8EC", "#5A5A66"),   # state in / out
    "perceive": ("#CFE3F7", "#2C6EA8"),   # spatial convolution (the only non-local op)
    "mlp":      ("#E2D8F3", "#6B4FA8"),   # per-cell 1x1 MLP
    "head":     ("#FBE2C8", "#B5761F"),   # output head (zero-initialised)
    "physics":  ("#CFEBD5", "#2F7D45"),   # conservation / bounding operators
    "spectral": ("#F8D3D3", "#B03A3A"),   # FFT / spectral mixing
    "loss":     ("#F5EFC8", "#8A7A1E"),
}
FS = 8.2


def box(ax, x, y, w, h, text, role="mlp", fontsize=FS):
    fill, edge = STYLE[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.025",
                                linewidth=1.3, facecolor=fill, edgecolor=edge, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, linespacing=1.35)
    return (x, y, w, h)


def plus(ax, x, y, size=17):
    """Residual-add node. A marker is sized in POINTS so it stays circular whatever the
    axes aspect ratio is - a data-space Circle comes out an ellipse on a wide canvas."""
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor="white",
            markeredgecolor="#5A5A66", markeredgewidth=1.3, zorder=3, clip_on=False)
    ax.text(x, y, "+", ha="center", va="center", fontsize=10.5, zorder=4)


def arrow(ax, p0, p1, rad=0.0, color="#4A4A55"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def chain(ax, specs, y, h, x0=0.012, gap=0.032, connect=True):
    """Lay a row of boxes out left-to-right with a real gap, so the arrows are visible.

    specs: list of (width, text, role). Returns the box tuples.
    """
    boxes, x = [], x0
    for w, text, role in specs:
        boxes.append(box(ax, x, y, w, h, text, role))
        x += w + gap
    if connect:
        for p, q in zip(boxes, boxes[1:]):
            arrow(ax, right(p), left(q))
    return boxes


def skip(ax, src_box, dst_point, lane_y, color="#8A8A96"):
    """Residual skip connection routed BELOW the row - above it would hit the title."""
    sx, sy = bottom(src_box)
    dx, dy = dst_point
    ax.plot([sx, sx], [sy, lane_y], color=color, lw=1.2, zorder=1)
    ax.plot([sx, dx], [lane_y, lane_y], color=color, lw=1.2, zorder=1)
    ax.add_patch(FancyArrowPatch((dx, lane_y), (dx, dy - 0.030), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=color, zorder=1))


def _canvas(w=11.0, h=3.0):
    fig, ax = plt.subplots(figsize=(w, h))
    # Patches are clipped to the axes limits, and several rows end just past x=1.0,
    # so leave a little headroom rather than shaving box widths until text stops fitting.
    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, name, title):
    os.makedirs(OUT, exist_ok=True)
    fig.suptitle(title, fontsize=11, y=1.0)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote docs/figures/arch/{name}")


# --------------------------------------------------------------------------- #
def fig_plain_nca():
    fig, ax = _canvas(11.5, 2.8)
    y, h = 0.50, 0.32
    a, b, c, d = chain(ax, [(0.13, "state x\n(B,H,W,C)", "io"),
                            (0.18, "perceive\n3x3 circular conv\n48 ch, He-init", "perceive"),
                            (0.17, "per-cell MLP\n1x1 64 -> 1x1 48\nReLU", "mlp"),
                            (0.16, "update head\n1x1 -> C ch\n(zero-init)", "head")], y, h)
    px = 0.80
    plus(ax, px, y + h / 2)
    e = box(ax, 0.855, y, 0.135, h, "x_next\nno constraint", "io")
    arrow(ax, right(d), (px - 0.024, y + h / 2))
    arrow(ax, (px + 0.024, y + h / 2), left(e))
    skip(ax, a, (px, y + h / 2), 0.28)
    ax.text(0.5, 0.05, "x(t+1) = x(t) + MLP(ReLU(perceive(x)))  -  a residual state "
                       "increment.\nNothing constrains it, so total mass can drift.",
            ha="center", fontsize=9)
    _save(fig, "arch_plain_nca.png", "A. Plain NCA  -  models/nca.py")


def fig_pi_nca():
    fig, ax = _canvas(12.0, 2.9)
    y, h = 0.50, 0.32
    a, b, c, d, e = chain(ax, [(0.12, "state x\n(B,H,W,1)", "io"),
                               (0.15, "perceive\n3x3 circular\n32 ch", "perceive"),
                               (0.16, "per-cell MLP\n1x1 64 -> 1x1 32", "mlp"),
                               (0.14, "flux head\n1x1 -> 2 ch\n(fx, fy)", "head"),
                               (0.17, "divergence\nof the flux", "physics")],
                          y, h, gap=0.028)
    px = 0.875
    plus(ax, px, y + h / 2)
    f = box(ax, 0.925, y, 0.115, h, "x_next\nmass\nconserved", "io")
    arrow(ax, right(e), (px - 0.024, y + h / 2))
    arrow(ax, (px + 0.024, y + h / 2), left(f))
    skip(ax, a, (px, y + h / 2), 0.28)
    ax.text(0.5, 0.04, "dx = (roll(fx) - fx) + (roll(fy) - fy).  Summed over a periodic grid "
                       "this telescopes to exactly zero,\nso total mass is conserved by "
                       "construction - the finite-volume form, not a penalty term.",
            ha="center", fontsize=9)
    _save(fig, "arch_pi_nca.png",
          "B. Conservative PI-NCA (DeepFluxNCA)  -  models/flux_nca.py")


def fig_multiscale():
    fig, ax = _canvas(11.5, 3.3)
    a = box(ax, 0.012, 0.42, 0.11, 0.26, "state x", "io")
    d1 = box(ax, 0.175, 0.71, 0.165, 0.21, "perceive d=1\n3x3, 24 ch", "perceive")
    d2 = box(ax, 0.175, 0.435, 0.165, 0.21, "perceive d=2\n3x3 dilated", "perceive")
    d4 = box(ax, 0.175, 0.16, 0.165, 0.21, "perceive d=4\n3x3 dilated", "perceive")
    cc = box(ax, 0.395, 0.435, 0.10, 0.21, "concat\nReLU", "mlp")
    m = box(ax, 0.53, 0.435, 0.14, 0.21, "1x1 conv 64\nReLU", "mlp")
    fh = box(ax, 0.705, 0.435, 0.13, 0.21, "flux head\n(zero-init)", "head")
    dv = box(ax, 0.87, 0.435, 0.125, 0.21, "divergence\n+ mass\nre-project", "physics")
    for q, r in ((d1, 0.16), (d2, 0.0), (d4, -0.16)):
        arrow(ax, right(a), left(q), rad=r)
        arrow(ax, right(q), left(cc), rad=-r)
    for p, q in [(cc, m), (m, fh), (fh, dv)]:
        arrow(ax, right(p), left(q))
    ax.text(0.5, 0.03, "Dilations 1/2/4 widen the reach to +/-4 cells per step; a single 3x3 "
                       "reaches only +/-1.\nThat closes most of the locality gap at about 1% "
                       "of the FNO's parameter count.", ha="center", fontsize=9)
    _save(fig, "arch_multiscale_flux_nca.png",
          "C. MultiScaleFluxNCA  -  dilated multi-scale perception + conservative flux")


def fig_bounded():
    fig, ax = _canvas(11.5, 3.0)
    y, h = 0.50, 0.28
    a, b, d, e, cl, rp = chain(ax, [(0.115, "state x\nin [-1,1]", "io"),
                                    (0.15, "perceive 3x3\n+ per-cell MLP", "perceive"),
                                    (0.13, "flux head\n(zero-init)", "head"),
                                    (0.145, "divergence\n(conserves)", "physics"),
                                    (0.15, "clip to [-1,1]\nstabilises,\nbreaks mass",
                                     "physics"),
                                    (0.155, "re-project\nmass to m(t)", "physics")],
                               y, h, gap=0.026)
    tm = box(ax, 0.145, 0.10, 0.155, 0.20, "record total\nmass m(t)", "physics")
    arrow(ax, bottom(a), left(tm), rad=-0.25)
    arrow(ax, right(tm), bottom(rp), rad=-0.16)
    ax.text(0.5, 0.015, "Resolves the stability-vs-conservation tension from ablation A1: "
                        "clipping alone fixed the divergence but destroyed\nconservation "
                        "(3.3e-5 -> 7.6). Re-projecting the recorded mass restores it "
                        "(2.4e-4) and the field stays bounded.", ha="center", fontsize=9)
    _save(fig, "arch_bounded_cons_nca.png",
          "D. BoundedConsFluxNCA  -  bounded AND mass-conserving")


def fig_spectral():
    fig, ax = _canvas(11.5, 3.2)
    a = box(ax, 0.012, 0.42, 0.10, 0.24, "state x", "io")
    lo = box(ax, 0.16, 0.66, 0.25, 0.24,
             "LOCAL stream\nperceive 3x3 -> 1x1 64\n-> flux head", "perceive")
    ld = box(ax, 0.45, 0.66, 0.16, 0.24, "divergence\nconserves mass", "physics")
    gl = box(ax, 0.16, 0.14, 0.25, 0.24,
             "GLOBAL stream\nlift 1x1 ->\n[SpectralConv2d + 1x1] x2", "spectral")
    gp = box(ax, 0.45, 0.14, 0.16, 0.24, "project -> g\n(zero-init)", "head")
    ce = box(ax, 0.755, 0.40, 0.16, 0.24, "conserve mass\n(optional)", "physics")
    o = box(ax, 0.945, 0.40, 0.10, 0.24, "x_next", "io")
    arrow(ax, right(a), left(lo), rad=0.12)
    arrow(ax, right(a), left(gl), rad=-0.12)
    arrow(ax, right(lo), left(ld))
    arrow(ax, right(gl), left(gp))
    px = 0.678
    plus(ax, px, 0.52)
    arrow(ax, right(ld), (px - 0.02, 0.545), rad=0.08)
    arrow(ax, right(gp), (px - 0.02, 0.495), rad=-0.08)
    arrow(ax, (px + 0.022, 0.52), left(ce))
    arrow(ax, right(ce), left(o))
    ax.text(0.5, 0.015, "The central hybrid idea: the spectral stream supplies the global "
                        "reach a local NCA lacks,\nwhile the flux-divergence stream keeps the "
                        "combined update mass-conserving.", ha="center", fontsize=9)
    _save(fig, "arch_spectral_flux_nca.png",
          "E. SpectralFluxNCA  -  local conservation + global spectral correction")


def fig_mc_flux():
    fig, ax = _canvas(11.5, 2.8)
    y, h = 0.48, 0.30
    chain(ax, [(0.135, "state x\n(B,H,W,C)\nC = 2..3", "io"),
               (0.165, "perceive 3x3 48 ch\n-> 1x1 96", "perceive"),
               (0.165, "flux head\n1x1 -> 2C ch\n(fx, fy) per field", "head"),
               (0.135, "reshape\n(B,H,W,C,2)", "mlp"),
               (0.155, "per-channel\ndivergence", "physics"),
               (0.115, "x_next", "io")], y, h, gap=0.028)
    ax.text(0.5, 0.05, "Each field gets its own flux pair, so EVERY channel's total is "
                       "conserved separately.\nThe right prior for shallow-water "
                       "(mass + momentum); a deliberately wrong one for reaction systems, "
                       "which have source terms.", ha="center", fontsize=9)
    _save(fig, "arch_mc_flux_nca.png",
          "F. MultiChannelFluxNCA  -  per-field conservation for multi-field states")


def fig_fno():
    fig, ax = _canvas(11.5, 3.1)
    y, h = 0.48, 0.26
    a = box(ax, 0.012, y, 0.10, h, "state x", "io")
    li = box(ax, 0.15, y, 0.125, h, "lift 1x1\n-> width 24", "mlp")
    sp = box(ax, 0.32, 0.68, 0.215, 0.24,
             "SpectralConv2d\nFFT -> keep 8x8\nlowest modes -> iFFT", "spectral")
    lc = box(ax, 0.32, 0.16, 0.215, 0.24, "local 1x1 conv\nW v", "mlp")
    ge = box(ax, 0.575, y, 0.11, h, "GeLU\nx4 depth", "mlp")
    pr = box(ax, 0.72, y, 0.135, h, "project 1x1\n-> C\n(zero-init)", "head")
    o = box(ax, 0.945, y, 0.10, h, "x_next", "io")
    arrow(ax, right(a), left(li))
    arrow(ax, right(li), left(sp), rad=0.14)
    arrow(ax, right(li), left(lc), rad=-0.14)
    arrow(ax, right(sp), left(ge), rad=0.14)
    arrow(ax, right(lc), left(ge), rad=-0.14)
    arrow(ax, right(ge), left(pr))
    px = 0.893
    plus(ax, px, y + h / 2)
    arrow(ax, right(pr), (px - 0.024, y + h / 2))
    arrow(ax, (px + 0.024, y + h / 2), left(o))
    skip(ax, a, (px, y + h / 2), 0.075)
    ax.text(0.5, 0.015, "Truncating to the lowest Fourier modes gives a GLOBAL receptive "
                        "field in a constant number of layers - the opposite\ntrade-off to an "
                        "NCA. The cost is about 593k parameters here, roughly 100x the "
                        "conservative NCAs, and no built-in conservation.",
            ha="center", fontsize=9)
    _save(fig, "arch_fno.png", "G. Fourier Neural Operator (FNO2d)  -  models/fno.py")


def fig_pipeline():
    fig, ax = _canvas(10.5, 3.6)
    a = box(ax, 0.03, 0.76, 0.22, 0.20, "random initial condition\nseed 42", "io")
    b = box(ax, 0.03, 0.44, 0.22, 0.20,
            "pre-seed: solver-evolve\nN steps to a developed state", "physics")
    t = box(ax, 0.35, 0.66, 0.24, 0.20, "teacher =\nsolver.rollout(K)\ndifferentiable",
            "physics")
    p = box(ax, 0.35, 0.28, 0.24, 0.20, "model rollout K\n(lax.scan, BPTT)", "mlp")
    l = box(ax, 0.67, 0.47, 0.13, 0.20, "MSE loss", "loss")
    o = box(ax, 0.85, 0.47, 0.14, 0.20, "AdamW\n+ LR warmup", "loss")
    ev = box(ax, 0.35, 0.04, 0.24, 0.16, "evaluate: long horizon,\n20 metrics vs solver", "io")
    arrow(ax, bottom(a), top(b))
    arrow(ax, right(b), left(t), rad=0.18)
    arrow(ax, right(b), left(p), rad=-0.12)
    arrow(ax, right(t), left(l), rad=-0.14)
    arrow(ax, right(p), left(l), rad=0.14)
    arrow(ax, right(l), left(o))
    ax.plot([0.92, 0.92], [0.47, 0.20], color="#8A8A96", lw=1.2, zorder=1)
    ax.plot([0.92, 0.47], [0.20, 0.20], color="#8A8A96", lw=1.2, zorder=1)
    arrow(ax, (0.47, 0.20), (0.47, 0.272), color="#8A8A96")
    arrow(ax, bottom(p), top(ev))
    ax.text(0.5, -0.05, "Every architecture trains the same way - same teacher, same horizon, "
                        "same metrics - so the comparison is apples to apples.",
            ha="center", fontsize=9)
    _save(fig, "arch_training_pipeline.png",
          "H. Shared training pipeline  -  identical for every architecture")


def fig_family():
    fig, ax = _canvas(10.5, 3.8)
    base = box(ax, 0.26, 0.78, 0.53, 0.18,
               "shared backbone\nperceive (3x3 circular)  ->  per-cell MLP  ->  head",
               "perceive", fontsize=8.6)
    resid = box(ax, 0.04, 0.48, 0.24, 0.18, "residual head\nPLAIN NCA", "head", fontsize=8.6)
    flux = box(ax, 0.38, 0.48, 0.24, 0.18, "flux head + divergence\nPI-NCA", "physics",
               fontsize=8.6)
    fno = box(ax, 0.72, 0.48, 0.24, 0.18, "spectral mixing\nFNO", "spectral", fontsize=8.6)
    ms = box(ax, 0.16, 0.14, 0.22, 0.20, "+ dilated\nperception\nMultiScale", "physics",
             fontsize=8.4)
    bc = box(ax, 0.42, 0.14, 0.22, 0.20, "+ clip &\nre-project\nBoundedCons", "physics",
             fontsize=8.4)
    sf = box(ax, 0.68, 0.14, 0.22, 0.20, "+ spectral\nstream\nSpectralFlux", "spectral",
             fontsize=8.4)
    arrow(ax, bottom(base), top(resid), rad=0.25)
    arrow(ax, bottom(base), top(flux))
    arrow(ax, bottom(base), top(fno), rad=-0.25)
    for q, r in ((ms, 0.16), (bc, 0.0), (sf, -0.16)):
        arrow(ax, bottom(flux), top(q), rad=r)
    ax.text(0.5, 0.01, "All three hybrids are the conservative PI-NCA plus exactly ONE change. "
                       "That is what makes the ablations clean:\neach hybrid isolates a single "
                       "component - reach, boundedness, or global mixing.",
            ha="center", fontsize=9)
    _save(fig, "arch_family_tree.png",
          "I. How the architectures relate  -  one backbone, different heads")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[arch_figs] writing {os.path.abspath(OUT)}")
    for fn in (fig_family, fig_plain_nca, fig_pi_nca, fig_multiscale, fig_bounded,
               fig_spectral, fig_mc_flux, fig_fno, fig_pipeline):
        fn()
    print("[arch_figs] done.")


if __name__ == "__main__":
    main()
