#!/usr/bin/env bash
# Full benchmark run on one NVIDIA GPU. Linux or WSL2, headless.
#
#   bash run_gpu.sh                 # the whole thing
#   bash run_gpu.sh --profile smoke # ~2 min wiring check at tiny scale
#   bash run_gpu.sh --profile bench # measurements + plots, no field figures
#
# This is a thin wrapper. All orchestration, resumption and error handling lives in
# python -m pinca_jax.runner, because a Python driver can catch an out-of-memory error
# on one model and carry on, which a shell script cannot.
#
# Any flags you pass are forwarded to the runner:
#   --only bench2d,plots   --skip figures   --force   --no-gate
set -euo pipefail
cd "$(dirname "$0")"

# Do not let XLA preallocate ~90% of VRAM: it turns every later allocation failure
# into a hard OOM and leaves no headroom for nvidia-smi or a second process.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Opt-in determinism. It costs speed, and the XLA flag name has moved between
# releases, so it is off unless asked for:  DETERMINISTIC=1 bash run_gpu.sh
if [ "${DETERMINISTIC:-0}" = "1" ]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
fi

nvidia-smi || { echo "nvidia-smi not found - is this the GPU box, and are you in WSL2?"; exit 1; }

exec python -m pinca_jax.runner "$@"
