"""Make `src/` importable in tests without an install step (no-admin friendly)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# On Ampere+ GPUs (RTX 30xx/40xx, A100...) JAX defaults conv/matmul to TF32, which is
# ~1e-3 less precise than true float32. That's fine for training but it blows through
# the 1e-5 atol / 1e-4 rtol these tests use to assert exact parity with the PyTorch
# reference, so force full float32 precision for the correctness gate regardless of
# backend. Must be set before jax reads it (first import), so this runs at collection.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
