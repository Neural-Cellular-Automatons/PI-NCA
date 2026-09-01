#!/usr/bin/env bash
# Final benchmark run on a single NVIDIA GPU (developed against an RTX 4090, 24 GB).
# Headless: no IDE, no notebook, no display. Everything writes to results/ and docs/figures/.
#
#   bash run_gpu.sh              # full run: benchmarks + figures
#   bash run_gpu.sh bench        # benchmarks + plots only, no field figures (much faster)
#   bash run_gpu.sh smoke        # ~2 min sanity pass at reduced scale, proves the wiring
#
# Resume-friendly: every stage writes its own files, so re-running skips nothing but
# also clobbers nothing it did not produce.
set -euo pipefail

MODE="${1:-full}"

# Do not let XLA preallocate 90% of VRAM — leaves room for nvidia-smi / a second run.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Opt-in determinism: costs speed, and the flag name has moved between XLA releases.
#   DETERMINISTIC=1 bash run_gpu.sh
if [ "${DETERMINISTIC:-0}" = "1" ]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
fi

case "$MODE" in
  smoke)
    SEEDS=1; EPOCHS=60;   GRID=24; BATCH=16; GRID3D=16; EPOCHS3D=40; RES_EPOCHS=40
    VIZ_GRID=24; VIZ_EPOCHS=60;   VIZ3D_GRID=16; VIZ3D_EPOCHS=40; DO_FIGURES=1
    MAX_MB=8 ;;
  bench)
    SEEDS=3; EPOCHS=2000; GRID=64; BATCH=64; GRID3D=32; EPOCHS3D=800; RES_EPOCHS=600
    DO_FIGURES=0; MAX_MB=64 ;;
  full)
    SEEDS=3; EPOCHS=2000; GRID=64; BATCH=64; GRID3D=32; EPOCHS3D=800; RES_EPOCHS=600
    # Figures are pictures, not measurements — they do not need benchmark-grade
    # training, and viz3d_volume renders every voxel through matplotlib on the CPU,
    # so 32^3 there costs far more wall-clock than it adds to the page.
    VIZ_GRID=48; VIZ_EPOCHS=400; VIZ3D_GRID=16; VIZ3D_EPOCHS=200; DO_FIGURES=1
    MAX_MB=64 ;;
  *)
    echo "usage: bash run_gpu.sh [full|bench|smoke]"; exit 2 ;;
esac

FAILED=()
# Figures and baselines must never destroy a completed benchmark run: record the
# failure, keep going, and report at the end. Benchmarks themselves stay fatal.
soft() {
  echo "--- $*"
  if ! "$@"; then
    echo "!!! FAILED (continuing): $*"
    FAILED+=("$*")
  fi
}

echo "=== 0. environment ==="
nvidia-smi || { echo "nvidia-smi missing — is this actually the GPU box?"; exit 1; }
python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"
python -c "import jax,sys; sys.exit(0 if jax.default_backend()=='gpu' else 1)" || {
  echo "JAX is not on the GPU backend. Install requirements-gpu.txt first."; exit 1; }

echo "=== 1. correctness gate (must be green before any benchmark) ==="
python -m pytest tests/ -q

echo "=== 2. 2-D suite + ablations  (seeds=$SEEDS grid=$GRID batch=$BATCH epochs=$EPOCHS) ==="
python -m pinca_jax.bench_all --group all --seeds "$SEEDS" --epochs "$EPOCHS" \
                              --grid "$GRID" --batch "$BATCH"

echo "=== 3. 3-D suite  (grid=${GRID3D}^3 epochs=$EPOCHS3D) ==="
python -m pinca_jax.bench3d --grid "$GRID3D" --epochs "$EPOCHS3D" --batch 16

echo "=== 4. resolution-transfer study ==="
python -m pinca_jax.res_study --pdes heat,allen_cahn,navier_stokes --epochs "$RES_EPOCHS"

# Plot as soon as the measurements exist, so the figure suite is on disk even if a
# later stage dies or you stop the run early. Costs seconds; repeated at the end.
echo "=== 5. benchmark plots (from results/ so far) ==="
soft python -m pinca_jax.plots

echo "=== 6. PINN / DeepONet / Darcy baselines ==="
soft python -m pinca_jax.pinn_heat
soft python -m pinca_jax.deeponet_heat
soft python -m pinca_jax.darcy

if [ "$DO_FIGURES" = "1" ]; then
  # 7a. Train ONCE per phenomenon and archive the raw trajectories. Everything after
  # this renders from those files — including anything you write later, on any
  # machine, with no GPU. viz3d and viz3d_volume previously each retrained the same
  # model separately, so this also halves the 3-D figure cost.
  echo "=== 7a. capture trajectories -> results/traj/*.npz ==="
  soft python -m pinca_jax.capture --dims both \
       --grid "$VIZ_GRID" --epochs "$VIZ_EPOCHS" \
       --grid3d "$VIZ3D_GRID" --epochs3d "$VIZ3D_EPOCHS" --max-mb "$MAX_MB"

  echo "=== 7b. field figures (rendered from the capture files, no training) ==="
  for pde in heat allen_cahn nagumo adv_diff gray_scott shallow_water \
             fitzhugh_nagumo wave cahn_hilliard navier_stokes; do
    f="results/traj/${pde}_2d.npz"
    # NB: `[ -f x ] && cmd` would abort the script under `set -e` when x is missing.
    if [ -f "$f" ]; then soft python -m pinca_jax.viz --npz "$f"; fi
  done
  for pde in heat adv_diff allen_cahn nagumo gray_scott fitzhugh_nagumo; do
    f="results/traj/${pde}_3d.npz"
    [ -f "$f" ] || continue
    soft python -m pinca_jax.viz3d --npz "$f"
    soft python -m pinca_jax.viz3d_volume --npz "$f"
  done
else
  echo "=== 7. field figures SKIPPED (mode=$MODE) — run capture+viz later if wanted ==="
fi

echo "=== 8. benchmark plots (final) ==="
soft python -m pinca_jax.plots

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "done, no failures."
else
  echo "done, but ${#FAILED[@]} non-fatal step(s) failed:"
  printf '  %s\n' "${FAILED[@]}"
fi
echo "tables    -> results/*.md"
echo "plots     -> docs/figures/bench/*.png"
echo "raw data  -> results/traj/*.npz   (rebuild any figure later, no GPU:"
echo "             python -m pinca_jax.viz3d_volume --npz results/traj/heat_3d.npz )"
