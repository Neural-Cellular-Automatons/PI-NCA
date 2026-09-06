@echo off
REM Windows driver for the benchmark run.
REM
REM   run_gpu.bat            full run: every stage, including field figures
REM   run_gpu.bat bench      measurements + plots only, no field figures (much faster)
REM   run_gpu.bat smoke      minutes-long wiring check at tiny scale (numbers meaningless)
REM
REM This is a thin wrapper. Every stage, scale preset, resumption rule and error-handling
REM decision lives in `python -m pinca_jax.runner`, and this file must not restate them:
REM it previously carried its own copy of the stage list, which is exactly how a launcher
REM silently stops running the experiments that were added after it was written.
REM
REM Any extra arguments are forwarded to the runner, e.g.
REM   run_gpu.bat bench --only bench2d,stability   run_gpu.bat full --force --no-gate
REM
REM IMPORTANT: JAX has NO native-Windows GPU support (docs.jax.dev: Windows x86_64 = "no",
REM Windows WSL2 = "experimental"). Run this from cmd and you get the CPU backend however
REM good the GPU is. For a real GPU run use WSL2 and run_gpu.sh -- see docs/gpu_runbook.md.
REM The runner refuses to start on CPU unless --allow-cpu is passed, so a CPU run cannot be
REM mistaken for a GPU one after the fact.

setlocal
cd /d "%~dp0"

set MODE=%1
if "%MODE%"=="" set MODE=full
if "%MODE%"=="full"  shift & goto :ok
if "%MODE%"=="bench" shift & goto :ok
if "%MODE%"=="smoke" shift & goto :ok
REM No profile given: treat every argument as a runner flag.
set MODE=full
goto :ok

:ok
REM Do not let XLA preallocate most of VRAM: it turns every later allocation failure into
REM a hard OOM and leaves no headroom for nvidia-smi or a second process.
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_ALLOCATOR=platform

where python >nul 2>&1 || (echo python not on PATH - activate the venv first: .venv\Scripts\activate.bat & exit /b 1)
python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())" || exit /b 1

python -m pinca_jax.runner --profile %MODE% %1 %2 %3 %4 %5 %6 %7 %8 %9
endlocal
