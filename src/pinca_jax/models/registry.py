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
}

# The bounded variants take their range from the caller. Benchmark drivers pass the
# PDE's measured physical range (harness.field_bounds), so "bounded" is a general
# technique rather than a Cahn-Hilliard special case: clipping heat to a hardcoded
# [-1,1] would destroy a field whose amplitudes run 5-10. The (-1,1) default is only
# a fallback for direct construction.


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
