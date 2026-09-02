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


class NotOnGPU(RuntimeError):
    """Raised when a benchmark would otherwise silently produce CPU numbers."""


def require_gpu(tag: str, allow_cpu: bool = False) -> str:
    """Print the backend and REFUSE to run a benchmark on CPU.

    A sweep that silently falls back to CPU produces numbers that cannot be compared
    with GPU numbers -- different throughput, different latency, different achievable
    scale -- and mixing the two inside one results table is worse than having no table.
    `allow_cpu` exists for local development only.
    """
    backend = banner(tag)
    if backend == "gpu" or allow_cpu:
        if backend != "gpu":
            print(f"[{tag}] proceeding on {backend} because --allow-cpu was passed; "
                  f"these numbers are NOT comparable to a GPU run.")
        return backend
    raise NotOnGPU(
        f"[{tag}] refusing to benchmark on the '{backend}' backend.\n"
        f"  Install the CUDA build:  pip install -r requirements-gpu.txt\n"
        f"  Then check:              python -c \"import jax; print(jax.devices())\"\n"
        f"  Expected:                [CudaDevice(id=0)]\n"
        f"  Native Windows cannot reach the GPU at all - use WSL2.\n"
        f"  To measure on CPU anyway (not comparable), pass --allow-cpu."
    )


def configure_memory():
    """Set the XLA memory knobs a long sweep needs, if the caller has not.

    Preallocation grabs most of VRAM up front, which makes every later allocation
    failure look like a hard OOM and leaves no headroom for nvidia-smi or a second
    process. Setting these here means the Python entry points behave correctly even
    when they are not launched through the shell wrapper.
    """
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
