@echo off
REM THE one command, Windows edition. Produces every paper artifact.
REM
REM   run_paper.bat                  the whole pipeline, `paper` preset
REM   run_paper.bat --estimate       measure two real trainings, project the cost, exit
REM   run_paper.bat --profile smoke  minutes; proves the wiring, numbers are meaningless
REM
REM Anything passed here is forwarded to `python -m pinca_jax.runner`, so --only, --skip,
REM --force, --no-gate and --list-stages all work.
REM
REM Safe to interrupt: benchmarks checkpoint per (pde, architecture) cell and the runner
REM skips whole stages whose outputs already exist, so re-running continues where it
REM stopped.
REM
REM IMPORTANT: JAX has NO native-Windows GPU support (docs.jax.dev: Windows x86_64 = "no",
REM Windows WSL2 = "experimental"). Run this from cmd and you get the CPU backend however
REM good the card is, and the runner will refuse to produce benchmark numbers. For a real
REM GPU run use WSL2 and run_paper.sh -- see docs/gpu_runbook.md.

setlocal
cd /d "%~dp0"

REM Do not let XLA preallocate most of VRAM.
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_ALLOCATOR=platform
set PYTHONUNBUFFERED=1

REM Make the package importable straight from a fresh clone, so `pip install -e .` is a
REM convenience rather than a prerequisite (it also sidesteps the Windows MAX_PATH failure
REM that orbax-checkpoint triggers without admin rights). Subprocesses inherit this.
set PYTHONPATH=%~dp0src;%PYTHONPATH%

where python >nul 2>&1 || (echo python not on PATH - activate the venv first: .venv\Scripts\activate.bat & exit /b 1)

python -c "import pinca_jax" >nul 2>&1
if errorlevel 1 (
  echo The pinca_jax package is not importable. Install it first:
  echo     python -m venv .venv ^&^& .venv\Scripts\activate.bat
  echo     pip install -r requirements-jax.txt
  echo     pip install -e .
  echo See docs/gpu_runbook.md.
  exit /b 1
)

python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())" || exit /b 1

if "%~1"=="" (
  python -m pinca_jax.runner --profile paper
) else (
  python -m pinca_jax.runner %*
)
endlocal
