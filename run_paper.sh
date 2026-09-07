#!/usr/bin/env bash
# THE one command. Produces every number, table and figure in the paper.
#
#   bash run_paper.sh                 # the whole pipeline, `paper` preset
#   bash run_paper.sh --estimate      # measure two real trainings, project the cost, exit
#   bash run_paper.sh --profile full  # largest scale (expect multiple days on one GPU)
#   bash run_paper.sh --profile smoke # minutes; proves the wiring, numbers are meaningless
#
# Anything you pass is forwarded to `python -m pinca_jax.runner`, so
# `--only`, `--skip`, `--force`, `--no-gate` all work. `--list-stages` prints the names.
#
# Safe to interrupt. Every benchmark checkpoints per (pde, architecture) cell and the
# runner skips whole stages whose outputs already exist, so re-running this exact command
# after a crash, a Ctrl-C or a reboot continues instead of starting over.
#
# JAX has no native-Windows GPU support. On Windows, run this from WSL2.
set -euo pipefail
cd "$(dirname "$0")"

# Do not let XLA preallocate ~90% of VRAM: it turns every later allocation failure into a
# hard OOM and leaves no headroom for nvidia-smi or a second process.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Opt-in determinism. It costs speed and the XLA flag name has moved between releases, so
# it is off unless asked for:  DETERMINISTIC=1 bash run_paper.sh
if [ "${DETERMINISTIC:-0}" = "1" ]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
fi

PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || { echo "python not on PATH. Activate the venv first."; exit 1; }

# Accept a bare profile word as the first argument -- `run_paper.sh smoke` as well as
# `run_paper.sh --profile smoke`. The older launcher took the positional form and several
# documents still suggest it, and a wrong guess should not cost an argparse error at the
# start of an overnight run.
case "${1:-}" in
  smoke|paper|bench|full) set -- --profile "$1" "${@:2}";;
esac

# Make the package importable straight from a fresh clone, so `pip install -e .` is a
# convenience rather than a prerequisite. Subprocesses inherit this, so every stage sees
# it too. An editable install still takes precedence if one exists.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Import the package before anything else, so a missing install fails in one clear line
# rather than fifteen stages in.
if ! "$PY" -c "import pinca_jax" >/dev/null 2>&1; then
  cat <<'EOF'
The pinca_jax package is not importable. Install it first:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements-gpu.txt     # CUDA build of JAX
    pip install -e .

Or run bash setup_gpu.sh, which does the above. See docs/gpu_runbook.md.
EOF
  exit 1
fi

BACKEND="$("$PY" -c "import jax; print(jax.default_backend())" 2>/dev/null || echo unknown)"
"$PY" -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"

if [ "$BACKEND" != "gpu" ]; then
  # Only complain when the user actually wants real numbers. --allow-cpu, --estimate and
  # the smoke profile are legitimate CPU uses and should not be blocked here; the runner
  # itself enforces the GPU requirement for everything else.
  case " $* " in
    *" --allow-cpu "*|*" --estimate "*|*" --list-stages "*|*" --help "*|*" -h "*) ;;
    *)
      cat <<EOF

JAX is on the '$BACKEND' backend, not the GPU. Benchmark numbers from a CPU run are not
comparable with GPU numbers, so the runner will refuse to start.

  Linux / WSL2:  pip install -r requirements-gpu.txt   (or: bash setup_gpu.sh)
  Check:         python -c "import jax; print(jax.devices())"   -> [CudaDevice(id=0)]
  Native Windows cannot reach the GPU at all -- use WSL2.

To proceed on CPU anyway (development only), add --allow-cpu.
EOF
      exit 1;;
  esac
fi

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader || true

# Informational flags return in milliseconds; a banner about a long unattended run would
# just be noise in front of them.
case " $* " in
  *" --list-stages "*|*" --help "*|*" -h "*|*" --estimate "*)
    exec "$PY" -m pinca_jax.runner "$@";;
esac

echo
echo "Starting. This is a long unattended run -- see docs/gpu_runbook.md."
echo "Tip: bash run_paper.sh --estimate  measures the cost on this card before committing."
echo "Logging both to the terminal and to run_paper.log."
echo

# tee so an unattended run leaves a full record; PYTHONUNBUFFERED keeps it live.
export PYTHONUNBUFFERED=1
if [ $# -gt 0 ]; then
  exec "$PY" -m pinca_jax.runner "$@" 2>&1 | tee -a run_paper.log
else
  exec "$PY" -m pinca_jax.runner --profile paper 2>&1 | tee -a run_paper.log
fi
