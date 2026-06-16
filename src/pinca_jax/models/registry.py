"""Architecture registry — name -> constructor, for benchmark sweeps.

Each entry is a factory `(out_channels) -> (() -> Flax module)` so the harness can
instantiate the right output width per PDE. `scalar_only` marks models that only
apply to single-channel conserved fields (the flux-form PI-NCA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .nca import NCA
from .flux_nca import DeepFluxNCA
from .fno import FNO2d


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
        note="global spectral operator"),
}


def applicable(channels: int):
    """Archs runnable for a given channel count."""
    return {k: v for k, v in REGISTRY.items() if not (v.scalar_only and channels != 1)}
