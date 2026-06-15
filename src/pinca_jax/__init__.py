"""pinca_jax — JAX/Flax core for the PI-NCA research program.

Shared, migrated-from-PyTorch building blocks used by every research branch:
- equations/  : differentiable reference PDE solvers (teachers / metrics)
- models/     : NCA, PI-NCA, and (later) operator / hybrid architectures
- physics.py  : conservation operators (divergence, energy projection)

Design rules (see docs/literature_review.md §6 "Why JAX"):
- Pure functions + explicit PRNG keys (jax.random) for reproducibility.
- Steppers expose a single-step fn so rollouts compose with jax.lax.scan.
- Channels-last (NHWC) to match Flax conv conventions.
"""

__all__ = ["equations", "models", "physics"]
