@echo off
REM Windows cmd.exe driver for the benchmark run.
REM
REM   run_gpu.bat            full run: benchmarks + figures
REM   run_gpu.bat bench      benchmarks + plots only, no field figures (much faster)
REM   run_gpu.bat smoke      ~2 min wiring check at reduced scale
REM
REM IMPORTANT: JAX has NO native-Windows GPU support (docs.jax.dev: Windows x86_64 = "no",
REM Windows WSL2 = "experimental"). Run from cmd and you get the CPU backend, however good
REM the GPU is. For the real 4090 run use WSL2 and run_gpu.sh -- see docs/gpu_runbook.md.
REM This script still stops and tells you rather than silently producing CPU numbers.

setlocal
set MODE=%1
if "%MODE%"=="" set MODE=full

REM Do not let XLA preallocate most of VRAM.
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_ALLOCATOR=platform

REM one `set` per line: a trailing space before `&` would end up inside the value
if "%MODE%"=="smoke" (
  set SEEDS=1
  set EPOCHS=60
  set GRID=24
  set BATCH=16
  set GRID3D=16
  set EPOCHS3D=40
  set RES_EPOCHS=40
  set VIZ_GRID=24
  set VIZ_EPOCHS=60
  set VIZ3D_GRID=16
  set VIZ3D_EPOCHS=40
  set DO_FIGURES=1
) else (
  set SEEDS=3
  set EPOCHS=2000
  set GRID=64
  set BATCH=64
  set GRID3D=32
  set EPOCHS3D=800
  set RES_EPOCHS=600
  REM Figures are pictures, not measurements: they do not need benchmark-grade
  REM training, and viz3d_volume renders every voxel through matplotlib on the CPU.
  set VIZ_GRID=48
  set VIZ_EPOCHS=400
  set VIZ3D_GRID=16
  set VIZ3D_EPOCHS=200
  set DO_FIGURES=1
)
if "%MODE%"=="bench" set DO_FIGURES=0

echo === 0. environment ===
where python >nul 2>&1 || (echo python not on PATH - activate the venv first: .venv\Scripts\activate.bat & exit /b 1)
python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())" || exit /b 1
python -c "import jax,sys; sys.exit(0 if jax.default_backend()=='gpu' else 1)"
if errorlevel 1 (
  echo.
  echo JAX is on the CPU backend. Native Windows cannot reach the GPU - use WSL2 + run_gpu.sh.
  echo Continue on CPU anyway? Ctrl-C to stop.
  pause
)

echo === 1. correctness gate ===
python -m pytest tests/ -q || exit /b 1

echo === 2. 2-D suite + ablations ===
python -m pinca_jax.bench_all --group all --seeds %SEEDS% --epochs %EPOCHS% --grid %GRID% --batch %BATCH% || exit /b 1

echo === 3. 3-D suite ===
python -m pinca_jax.bench3d --grid %GRID3D% --epochs %EPOCHS3D% --batch 16 || exit /b 1

echo === 4. resolution-transfer study ===
python -m pinca_jax.res_study --pdes heat,allen_cahn,navier_stokes --epochs %RES_EPOCHS% || exit /b 1

REM Plot as soon as measurements exist, so figures survive an early stop.
echo === 5. benchmark plots (from results so far) ===
python -m pinca_jax.plots

REM Baselines and figures are NON-FATAL: they must never destroy a completed
REM benchmark run. Failures are reported, not propagated.
echo === 6. PINN / DeepONet / Darcy baselines ===
python -m pinca_jax.pinn_heat
python -m pinca_jax.deeponet_heat
python -m pinca_jax.darcy

if "%DO_FIGURES%"=="0" (
  echo === 7. field figures SKIPPED ^(mode=%MODE%^) ===
) else (
  echo === 7. field figures ===
  for %%p in (heat allen_cahn nagumo adv_diff gray_scott shallow_water fitzhugh_nagumo wave cahn_hilliard navier_stokes) do (
    python -m pinca_jax.viz --pde %%p --grid %VIZ_GRID% --epochs %VIZ_EPOCHS%
  )
  for %%p in (heat adv_diff allen_cahn nagumo gray_scott fitzhugh_nagumo) do (
    python -m pinca_jax.viz3d --pde %%p --grid %VIZ3D_GRID% --epochs %VIZ3D_EPOCHS%
    python -m pinca_jax.viz3d_volume --pde %%p --grid %VIZ3D_GRID% --epochs %VIZ3D_EPOCHS%
  )
)

echo === 8. benchmark plots (final) ===
python -m pinca_jax.plots

echo.
echo done. tables -^> results\*.md   figures -^> docs\figures\bench\*.png
endlocal
