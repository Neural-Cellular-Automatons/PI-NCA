"""Experiment configs. Reduced-scale (CPU) and full-scale (GPU) presets differ
ONLY in numeric fields, so identical code reproduces both (see docs/environment.md)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeatNCAConfig:
    # domain / data
    grid_size: int = 32
    batch_size: int = 16
    n_blobs: int = 4
    blob_sigma_frac: float = 0.08
    amp_low: float = 5.0
    amp_high: float = 10.0
    # physics
    alpha: float = 0.5
    dt: float = 0.1
    # NCA rollout (fixed horizon keeps jit cache hot; curriculum is optional)
    rollout_steps: int = 16
    conserve: bool = True
    # optimisation
    epochs: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-5
    seed: int = 0

    @property
    def alpha_dt(self) -> float:
        return self.alpha * self.dt


# Tiny preset for CI / smoke verification of the end-to-end JAX pipeline.
SMOKE = HeatNCAConfig(grid_size=16, batch_size=8, rollout_steps=8, epochs=40)

# Reduced-scale CPU preset (methodology demonstration).
CPU_REDUCED = HeatNCAConfig(grid_size=32, batch_size=16, rollout_steps=16, epochs=300)

# Full-scale GPU preset (re-run unchanged when hardware is available).
GPU_FULL = HeatNCAConfig(grid_size=64, batch_size=64, rollout_steps=64, epochs=900)
