"""Architecture registry - name -> constructor, for benchmark sweeps.

Each entry is a factory `(out_channels) -> (() -> Flax module)` so the harness can
instantiate the right output width per PDE.

Every architecture is now generic in the channel count, so `applicable()` returns the
whole registry for every PDE and the benchmark matrix is uniform: the same model list
is measured on every phenomenon. Previously the flux-form models were scalar-only,
which is why multi-field PDEs (wave, gray_scott, shallow_water, fitzhugh_nagumo) had
only three rows while scalar ones had five. `scalar_only` is kept on the ablation
entries alone, because A4/A5 are defined as scalar-field studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .nca import NCA
from .flux_nca import DeepFluxNCA, MultiChannelFluxNCA
from .fno import FNO2d
from .hybrids import BoundedConsFluxNCA, SpectralFluxNCA, MultiScaleFluxNCA
from .ablation_nca import AblationNCA
from .baselines import ResNetEmulator, UNetEmulator, IdentityEmulator
from .latent_fno import LatentFNOEmulator, largest_patch


@dataclass(frozen=True)
class ArchSpec:
    name: str
    # (out_channels, bounds=None) -> (() -> module). `bounds` is the PDE's physical
    # field range; only the bounded variants use it, the rest ignore it.
    make: Callable[..., Callable]
    scalar_only: bool = False
    note: str = ""


REGISTRY: dict[str, ArchSpec] = {
    "plain_nca": ArchSpec(
        "plain_nca", lambda C, bounds=None: (lambda: NCA(out_channels=C)),
        note="local residual NCA, no conservation"),
    "pi_nca": ArchSpec(
        "pi_nca", lambda C, bounds=None: (lambda: DeepFluxNCA(out_channels=C)),
        note="conservative flux-divergence NCA (per-field flux)"),
    "fno": ArchSpec(
        "fno", lambda C, bounds=None: (lambda: FNO2d(out_channels=C, width=24, modes=8, depth=4)),
        note="global spectral operator (~5.9e5 params)"),
    "fno_small": ArchSpec(
        "fno_small", lambda C, bounds=None: (lambda: FNO2d(out_channels=C, width=8, modes=4, depth=2)),
        note="iso-parameter FNO (~NCA budget) — A2 ablation: spectral mixing vs param count"),
    "mc_flux_nca": ArchSpec(
        "mc_flux_nca", lambda C, bounds=None: (lambda: MultiChannelFluxNCA(out_channels=C)),
        note="multi-channel per-field conservative flux NCA (SWE/FHN/GS)"),
    # --- standard learned surrogates with NO physics prior (the controls) ---
    # Without these, "the conservation prior helps" is confounded with "we never tried
    # an ordinary CNN". resnet is the local control for the NCA family; unet is the
    # multi-resolution route to a global receptive field, separating "global context
    # helps" from "Fourier mixing helps" (which the FNO alone cannot).
    "resnet": ArchSpec(
        "resnet", lambda C, bounds=None: (lambda: ResNetEmulator(out_channels=C, width=32, depth=4)),
        note="autoregressive residual CNN, circular pad (~7.4e4 params)"),
    "resnet_iso": ArchSpec(
        "resnet_iso", lambda C, bounds=None: (lambda: ResNetEmulator(out_channels=C, width=12, depth=2)),
        note="iso-parameter CNN control (~5.4e3, matched to the NCA budget)"),
    "unet": ArchSpec(
        "unet", lambda C, bounds=None: (lambda: UNetEmulator(out_channels=C, width=24, levels=2)),
        note="multi-resolution U-Net emulator (~2.7e5 params)"),
    "unet_iso": ArchSpec(
        "unet_iso", lambda C, bounds=None: (lambda: UNetEmulator(out_channels=C, width=4, levels=2)),
        note="iso-parameter U-Net control (~7.5e3, matched to the NCA budget)"),
    "identity": ArchSpec(
        "identity", lambda C, bounds=None: (lambda: IdentityEmulator(out_channels=C)),
        note="do-nothing floor g(x)=x -- any model above this learned worse than nothing"),
    # --- A4: conservation on/off at MATCHED backbone width (32/64, 3x3, single-scale) ---
    "abl_flux": ArchSpec(
        "abl_flux", lambda C, bounds=None: (lambda: AblationNCA(out_channels=C, head="flux")),
        scalar_only=True, note="A4: conservative flux head (matched backbone)"),
    "abl_residual": ArchSpec(
        "abl_residual", lambda C, bounds=None: (lambda: AblationNCA(out_channels=C, head="residual")),
        scalar_only=True, note="A4: residual head, no conservation (matched backbone)"),
    # --- A5: perception / receptive-field size (same head=flux, same widths) ---
    "abl_k3": ArchSpec(
        "abl_k3", lambda C, bounds=None: (lambda: AblationNCA(out_channels=C, kernel=3, dilations=(1,))),
        scalar_only=True, note="A5: 3x3 single-scale perception"),
    "abl_k5": ArchSpec(
        "abl_k5", lambda C, bounds=None: (lambda: AblationNCA(out_channels=C, kernel=5, dilations=(1,))),
        scalar_only=True, note="A5: 5x5 perception (wider single-scale)"),
    "abl_multiscale": ArchSpec(
        "abl_multiscale", lambda C, bounds=None: (lambda: AblationNCA(out_channels=C, kernel=3, dilations=(1, 2, 4))),
        scalar_only=True, note="A5: 3x3 dilated multi-scale (1,2,4)"),
    # --- A7: HOW mass is restored after the bound clip (bounded models only) ---
    # The naive uniform offset re-violates the bound it just enforced; the headroom
    # projection does not. Same backbone, same bounds, one line different.
    "abl_proj_uniform": ArchSpec(
        "abl_proj_uniform",
        lambda C, bounds=None: (lambda: MultiScaleFluxNCA(
            out_channels=C, conserve=True, bounds=bounds or (-1.0, 1.0), projection="uniform")),
        note="A7: uniform mass re-projection after clip (breaks the bound)"),
    "abl_proj_headroom": ArchSpec(
        "abl_proj_headroom",
        lambda C, bounds=None: (lambda: MultiScaleFluxNCA(
            out_channels=C, conserve=True, bounds=bounds or (-1.0, 1.0), projection="headroom")),
        note="A7: headroom-proportional re-projection (bounded AND conserving)"),
    "abl_proj_none": ArchSpec(
        "abl_proj_none",
        lambda C, bounds=None: (lambda: MultiScaleFluxNCA(
            out_channels=C, conserve=True, bounds=bounds or (-1.0, 1.0), projection="none")),
        note="A7: clip only, no re-projection (bounded, not conserving)"),

    # --- hybrids (scalar conservative fields) ---
    "bounded_cons_nca": ArchSpec(
        "bounded_cons_nca",
        lambda C, bounds=None: (lambda: BoundedConsFluxNCA(
            out_channels=C, bounds=bounds or (-1.0, 1.0))),
        note="flux NCA + clip + mass re-projection (bounded AND conserving)"),
    "spectral_flux_nca": ArchSpec(
        "spectral_flux_nca",
        lambda C, bounds=None: (lambda: SpectralFluxNCA(out_channels=C, conserve=True)),
        note="local conservative flux + global FNO spectral correction"),
    "multiscale_flux_nca": ArchSpec(
        "multiscale_flux_nca",
        lambda C, bounds=None: (lambda: MultiScaleFluxNCA(out_channels=C, conserve=True)),
        note="dilated multi-scale perception + conservative flux"),
    "bounded_multiscale_nca": ArchSpec(
        "bounded_multiscale_nca",
        lambda C, bounds=None: (lambda: MultiScaleFluxNCA(
            out_channels=C, conserve=True, bounds=bounds or (-1.0, 1.0))),
        note="UNIFIED: multi-scale perception + bounded + mass-conserving (stiff bounded fields)"),

    # --- latent-space operator: encode, evolve spectrally in the latent, decode ---
    # The architecture a JEPA-style latent objective needs. Registered so that it is
    # measured by the same harness, the same paired statistics, and against the same
    # identity floor as everything else -- a latent training objective is evaluated in
    # field space or not at all (see pinca_jax.jepa). `patch` is resolved against the
    # grid at construction time, so an indivisible grid degrades rather than crops.
    "latent_fno": ArchSpec(
        "latent_fno",
        lambda C, bounds=None, grid=48: (lambda: LatentFNOEmulator(
            out_channels=C, latent_dim=32, patch=largest_patch(grid, 4),
            width=32, modes=6, depth=4)),
        note="patch-encode -> latent FNO -> pixel-shuffle decode (~2.3e5 params)"),
    "latent_fno_iso": ArchSpec(
        "latent_fno_iso",
        lambda C, bounds=None, grid=48: (lambda: LatentFNOEmulator(
            out_channels=C, latent_dim=8, patch=largest_patch(grid, 4),
            width=8, modes=4, depth=2)),
        note="iso-parameter latent FNO control (~6e3, matched to the NCA budget)"),
}

# The bounded variants take their range from the caller. Benchmark drivers pass the
# PDE's measured physical range (harness.field_bounds), so "bounded" is a general
# technique rather than a Cahn-Hilliard special case: clipping heat to a hardcoded
# [-1,1] would destroy a field whose amplitudes run 5-10. The (-1,1) default is only
# a fallback for direct construction.


# Iso-parameter budget classes (TODO: fair efficiency claims). An accuracy comparison
# is only meaningful *within* a class; across classes report accuracy-at-equal-params
# and accuracy-at-equal-compute separately. Counts are for C=1 on a 32x32 grid.
BUDGET_CLASS = {
    "identity": "floor",
    "pi_nca": "small", "abl_flux": "small", "abl_residual": "small", "abl_k3": "small",
    "abl_k5": "small", "bounded_cons_nca": "small", "multiscale_flux_nca": "small",
    "bounded_multiscale_nca": "small", "resnet_iso": "small", "unet_iso": "small",
    "plain_nca": "small", "fno_small": "small", "abl_multiscale": "small",
    "mc_flux_nca": "small",
    "resnet": "medium",
    "abl_proj_uniform": "small", "abl_proj_headroom": "small", "abl_proj_none": "small",
    "latent_fno_iso": "small",
    "spectral_flux_nca": "large", "unet": "large", "fno": "large",
    "latent_fno": "large",
}


# Architectures compared on every phenomenon. The ablation entries (abl_*) are
# excluded: they are matched-backbone probes for A4/A5, not competitors.
BENCH_ARCHS = [k for k in REGISTRY if not k.startswith("abl_")]


def applicable(channels: int):
    """Archs runnable for a given channel count.

    Every architecture is channel-generic, so this is the full registry unless an
    entry is explicitly marked scalar_only (only the ablation probes are).
    """
    return {k: v for k, v in REGISTRY.items() if not (v.scalar_only and channels != 1)}


def bench_archs(channels: int):
    """The uniform comparison set: every competitor arch, same list for every PDE."""
    app = applicable(channels)
    return {k: app[k] for k in BENCH_ARCHS if k in app}
