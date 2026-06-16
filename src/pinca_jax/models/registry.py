"""Architecture registry — name -> constructor, for benchmark sweeps.

Each entry is a factory `(out_channels) -> (() -> Flax module)` so the harness can
instantiate the right output width per PDE. `scalar_only` marks models that only
apply to single-channel conserved fields (the flux-form PI-NCA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .nca import NCA
from .flux_nca import DeepFluxNCA, MultiChannelFluxNCA
from .fno import FNO2d
from .hybrids import BoundedConsFluxNCA, SpectralFluxNCA, MultiScaleFluxNCA


@dataclass(frozen=True)
class ArchSpec:
    name: str
    make: Callable[[int], Callable]  # out_channels -> (()->module)
    scalar_only: bool = False
    note: str = ""


REGISTRY: dict[str, ArchSpec] = {
    "plain_nca": ArchSpec(
        "plain_nca", lambda C: (lambda: NCA(out_channels=C)),
        note="local residual NCA, no conservation"),
    "pi_nca": ArchSpec(
        "pi_nca", lambda C: (lambda: DeepFluxNCA()), scalar_only=True,
        note="conservative flux-divergence NCA (C=1)"),
    "fno": ArchSpec(
        "fno", lambda C: (lambda: FNO2d(out_channels=C, width=24, modes=8, depth=4)),
        note="global spectral operator (~5.9e5 params)"),
    "fno_small": ArchSpec(
        "fno_small", lambda C: (lambda: FNO2d(out_channels=C, width=8, modes=4, depth=2)),
        note="iso-parameter FNO (~NCA budget) — A2 ablation: spectral mixing vs param count"),
    "mc_flux_nca": ArchSpec(
        "mc_flux_nca", lambda C: (lambda: MultiChannelFluxNCA(out_channels=C)),
        note="multi-channel per-field conservative flux NCA (SWE/FHN/GS)"),
    # --- hybrids (scalar conservative fields) ---
    "bounded_cons_nca": ArchSpec(
        "bounded_cons_nca", lambda C: (lambda: BoundedConsFluxNCA(bounds=(-1.0, 1.0))),
        scalar_only=True, note="flux NCA + clip + mass re-projection (bounded AND conserving)"),
    "spectral_flux_nca": ArchSpec(
        "spectral_flux_nca", lambda C: (lambda: SpectralFluxNCA(conserve=True)),
        scalar_only=True, note="local conservative flux + global FNO spectral correction"),
    "multiscale_flux_nca": ArchSpec(
        "multiscale_flux_nca", lambda C: (lambda: MultiScaleFluxNCA(conserve=True)),
        scalar_only=True, note="dilated multi-scale perception + conservative flux"),
    "bounded_multiscale_nca": ArchSpec(
        "bounded_multiscale_nca",
        lambda C: (lambda: MultiScaleFluxNCA(conserve=True, bounds=(-1.0, 1.0))),
        scalar_only=True,
        note="UNIFIED: multi-scale perception + bounded + mass-conserving (for stiff bounded fields)"),
}

# bounds for hybrids assume the field is in [-1,1] (Cahn-Hilliard / Allen-Cahn).
# For unbounded fields (heat) use bounds wide enough to be inert, or the plain pi_nca.


def applicable(channels: int):
    """Archs runnable for a given channel count."""
    return {k: v for k, v in REGISTRY.items() if not (v.scalar_only and channels != 1)}
