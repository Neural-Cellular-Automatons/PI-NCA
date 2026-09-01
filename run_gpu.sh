#!/usr/bin/env bash
# Final benchmark run on a single NVIDIA GPU (developed against an RTX 4090, 24 GB).
# Headless: no IDE, no notebook, no display. Everything writes to results/ and docs/figures/.
#
#   bash run_gpu.sh              # full run
#   bash run_gpu.sh smoke        # ~2 min sanity pass at CPU scale, proves the wiring
#
# Resume-friendly: every stage writes its own files, so re-running skips nothing but
# also clobbers nothing it did not produce.
set -euo pipefail

MODE="${1:-full}"

# Do not let XLA preallocate 90% of VRAM — leaves room for nvidia-smi / a second run.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Deterministic-ish reductions; drop this line if you want the last few % of speed.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"

if [ "$MODE" = "smoke" ]; then
  SEEDS=1; EPOCHS=60;   GRID=24; BATCH=16; GRID3D=16; EPOCHS3D=40; RES_EPOCHS=40
else
  SEEDS=3; EPOCHS=2000; GRID=64; BATCH=64; GRID3D=32; EPOCHS3D=800; RES_EPOCHS=600
fi

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

echo "=== 5. PINN / DeepONet / Darcy baselines ==="
python -m pinca_jax.pinn_heat
python -m pinca_jax.deeponet_heat
python -m pinca_jax.darcy

echo "=== 6. field figures (analytic vs model vs error) ==="
for pde in heat allen_cahn nagumo adv_diff gray_scott shallow_water \
           fitzhugh_nagumo wave cahn_hilliard navier_stokes; do
  python -m pinca_jax.viz --pde "$pde" --grid "$GRID" --epochs "$EPOCHS"
done
for pde in heat adv_diff allen_cahn nagumo gray_scott fitzhugh_nagumo; do
  python -m pinca_jax.viz3d        --pde "$pde" --grid "$GRID3D" --epochs "$EPOCHS3D"
  python -m pinca_jax.viz3d_volume --pde "$pde" --grid "$GRID3D" --epochs "$EPOCHS3D"
done

echo "=== 7. benchmark plots ==="
python -m pinca_jax.plots

echo
echo "done. tables -> results/*.md  figures -> docs/figures/bench/*.png"
