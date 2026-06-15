"""Neural architectures (Flax linen) for the PI-NCA study.

linen (not nnx) is used for the migrated core because its explicit param pytrees
make weight-level correctness checks against PyTorch straightforward (the
migration gate). nnx-style usage is noted in docs/migration/.
"""
from . import flux_nca

__all__ = ["flux_nca"]
