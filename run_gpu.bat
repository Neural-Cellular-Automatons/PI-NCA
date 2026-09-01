@echo off
REM Windows cmd.exe driver for the benchmark run.
REM
REM   run_gpu.bat            full run
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
) else (
  set SEEDS=3
  set EPOCHS=2000
  set GRID=64
  set BATCH=64
  set GRID3D=32
  set EPOCHS3D=800
  set RES_EPOCHS=600
)

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

echo === 5. PINN / DeepONet / Darcy baselines ===
python -m pinca_jax.pinn_heat || exit /b 1
python -m pinca_jax.deeponet_heat || exit /b 1
python -m pinca_jax.darcy || exit /b 1

echo === 6. field figures ===
for %%p in (heat allen_cahn nagumo adv_diff gray_scott shallow_water fitzhugh_nagumo wave cahn_hilliard navier_stokes) do (
  python -m pinca_jax.viz --pde %%p --grid %GRID% --epochs %EPOCHS% || exit /b 1
)
for %%p in (heat adv_diff allen_cahn nagumo gray_scott fitzhugh_nagumo) do (
  python -m pinca_jax.viz3d --pde %%p --grid %GRID3D% --epochs %EPOCHS3D% || exit /b 1
  python -m pinca_jax.viz3d_volume --pde %%p --grid %GRID3D% --epochs %EPOCHS3D% || exit /b 1
)

echo === 7. benchmark plots ===
python -m pinca_jax.plots || exit /b 1

echo.
echo done. tables -^> results\*.md   figures -^> docs\figures\bench\*.png
endlocal
