#!/usr/bin/env bash
# One-command environment build for the benchmark run. Linux or WSL2.
#
#   bash setup_gpu.sh          # GPU stack (jax[cuda12])
#   bash setup_gpu.sh cpu      # CPU stack (jax[cpu]) — plots and sanity checks only
#
# Idempotent: safe to re-run. Creates ./.venv, installs the pinned stack, and
# verifies the backend before it claims success.
#
# A virtualenv cannot be shipped in git — it bakes in absolute paths, ships
# platform-specific binaries, and the CUDA wheels are several GB. This script is
# the portable equivalent.
set -euo pipefail

FLAVOUR="${1:-gpu}"
REQ="requirements-gpu.txt"
[ "$FLAVOUR" = "cpu" ] && REQ="requirements-jax.txt"

cd "$(dirname "$0")"

# --- 1. find an interpreter the pins support (jax 0.10.1 needs >= 3.12) ------
PY=""
for c in python3.14 python3.13 python3.12 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "No Python >= 3.12 found (jax 0.10.1 requires it). On Ubuntu:"
  echo "  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi
echo "==> interpreter: $PY ($($PY --version))"

# --- 2. venv (ubuntu ships python3 without the venv module) ------------------
if [ ! -d .venv ]; then
  if ! "$PY" -m venv .venv 2>/dev/null; then
    VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    echo "==> venv module missing; installing python${VER}-venv (needs sudo)"
    sudo apt-get update -qq
    sudo apt-get install -y "python${VER}-venv" python3-pip
    "$PY" -m venv .venv
  fi
  echo "==> created .venv"
else
  echo "==> reusing existing .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade -q pip setuptools wheel

# --- 3. the stack -----------------------------------------------------------
# torch is the migration-correctness reference ONLY — take the CPU build so pip
# does not drag in ~3 GB of CUDA libs that duplicate what the jax wheel ships.
echo "==> installing torch (CPU reference build)"
python -m pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

echo "==> installing $REQ"
python -m pip install -r "$REQ"

echo "==> installing this package (editable, src layout)"
python -m pip install -q -e .

# --- 4. verify --------------------------------------------------------------
echo
echo "==> verifying"
python - <<'EOF'
import sys, jax
print(f"  python  {sys.version.split()[0]}")
print(f"  jax     {jax.__version__}")
print(f"  backend {jax.default_backend()}")
print(f"  devices {jax.devices()}")
EOF

if [ "$FLAVOUR" = "gpu" ] && ! python -c "import jax,sys; sys.exit(0 if jax.default_backend()=='gpu' else 1)"; then
  cat <<'EOF'

  BACKEND IS NOT GPU.
  - Native Windows cannot do this at all: JAX has no Windows CUDA wheels. Use WSL2.
  - In WSL2/Linux: check `nvidia-smi` lists the card, and that the Windows-side
    NVIDIA driver is recent. Never install an NVIDIA driver inside WSL.
  See docs/gpu_runbook.md.
EOF
  exit 1
fi

cat <<'EOF'

Environment ready. Next:

  source .venv/bin/activate     # every new shell
  bash run_gpu.sh smoke         # ~2 min wiring check
  bash run_gpu.sh               # full run
EOF
