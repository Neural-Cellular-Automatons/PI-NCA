"""Device banner + run provenance for benchmark drivers.

Headless GPU runs need to *prove* they ran on the GPU, so every driver prints the
backend up front and stamps it into the results JSON.
"""
from __future__ import annotations

import jax


def banner(tag: str) -> str:
    """Print backend/devices; return the backend name ('gpu'/'cpu'/'tpu')."""
    backend = jax.default_backend()
    print(f"[{tag}] jax {jax.__version__} | backend={backend} | devices={jax.devices()}")
    if backend == "cpu":
        print(f"[{tag}] WARNING: CPU backend. For the GPU run install requirements-gpu.txt "
              f"(jax[cuda12]); results will be reduced-scale/slow otherwise.")
    return backend


def peak_mem_mb() -> float:
    """Process-wide peak device memory (MB). 0.0 when the backend reports no stats.

    ponytail: whole-process peak, not per-architecture — XLA exposes no per-model
    reset. Report it once per driver run, not per row of a comparison table.
    """
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return 0.0
    return float(stats.get("peak_bytes_in_use", 0)) / 1e6


def provenance(tag: str) -> dict:
    """Device/version stamp to embed in results JSON."""
    return {"tag": tag, "jax": jax.__version__, "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()], "peak_mem_mb": peak_mem_mb()}
