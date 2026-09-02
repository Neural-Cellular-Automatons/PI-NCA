"""Evaluate whether the CAX accelerator improves NCA rollout performance.

CAX (arXiv:2410.02651) provides nnx-based CA primitives (ConvPerceive + NCAUpdate)
and a `ComplexSystem` whose multi-step call wraps the step in `nnx.scan` + `nnx.jit`
— structurally identical to our hand-written `jax.lax.scan` rollout. This script
builds a CAX NCA and times its K-step rollout against our lax.scan rollout of a
comparable Flax-linen NCA, to test whether CAX is faster on this CPU host.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import metrics
from .models.flux_nca import DeepFluxNCA


def build_cax_nca(channel_size=16, perception_size=48, hidden=(128,), seed=0):
    """Concrete CAX NCA: subclass ComplexSystem with perceive→update as _step."""
    from flax import nnx
    from cax.core.cs import ComplexSystem
    from cax.core.perceive.conv_perceive import ConvPerceive
    from cax.core.update.nca_update import NCAUpdate

    class CaxNCA(ComplexSystem):
        def __init__(self, rngs):
            self.perceive = ConvPerceive(channel_size=channel_size, perception_size=perception_size,
                                         padding="CIRCULAR", rngs=rngs)
            self.update = NCAUpdate(channel_size=channel_size, perception_size=perception_size,
                                    hidden_layer_sizes=hidden, rngs=rngs)

        def _step(self, state, input=None, *, sow=False):
            return self.update(state, self.perceive(state), input)

    return CaxNCA(nnx.Rngs(seed))


def main(grid=32, channel=16, steps=64, batch=8):
    print(f"CAX vs lax.scan NCA rollout — grid {grid}, {steps} steps, batch {batch}, CPU")

    # --- CAX path ---
    cax_ok = True
    try:
        cs = build_cax_nca(channel_size=channel)
        state = jnp.zeros((batch, grid, grid, channel))
        out = cs(state, num_steps=steps)          # jitted nnx.scan rollout
        assert jnp.all(jnp.isfinite(out))
        cax_t = metrics.time_callable(lambda s: cs(s, num_steps=steps), state)
        cax_per_step = cax_t / steps
    except Exception as e:  # CAX optional; record honestly if it fails
        cax_ok = False
        cax_per_step = None
        print("  CAX path error:", repr(e)[:160])

    # --- our lax.scan path (DeepFluxNCA, C=1) ---
    model = DeepFluxNCA()
    x = jnp.zeros((batch, grid, grid, 1))
    params = model.init(jax.random.PRNGKey(0), x)

    @jax.jit
    def scan_rollout(params, x):
        def body(s, _):
            return model.apply(params, s), None
        xf, _ = jax.lax.scan(body, x, xs=None, length=steps)
        return xf

    jax.block_until_ready(scan_rollout(params, x))
    ours_t = metrics.time_callable(lambda x: scan_rollout(params, x), x)
    ours_per_step = ours_t / steps

    print(f"  lax.scan DeepFluxNCA : {ours_per_step*1e3:.3f} ms/step  ({ours_t*1e3:.1f} ms / {steps})")
    if cax_ok:
        print(f"  CAX ComplexSystem    : {cax_per_step*1e3:.3f} ms/step  ({cax_t*1e3:.1f} ms / {steps})")
        print(f"  ratio (CAX/ours)     : {cax_per_step/ours_per_step:.2f}x")
    return {"ours_ms_per_step": ours_per_step * 1e3,
            "cax_ms_per_step": (cax_per_step * 1e3) if cax_ok else None}


if __name__ == "__main__":
    main()
