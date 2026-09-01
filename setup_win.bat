@echo off
REM Windows environment build. Works from cmd.exe and from PowerShell (.\setup_win.bat).
REM
REM   setup_win.bat
REM
REM CPU ONLY, by necessity. JAX publishes no jax-cuda12-plugin wheel for Windows
REM (jaxlib ships win_amd64; the CUDA plugin is manylinux-only), so a native
REM Windows install cannot reach an NVIDIA GPU no matter what nvidia-smi reports.
REM For the real GPU run use WSL2 + setup_gpu.sh -- see docs/gpu_runbook.md.
REM
REM This env is still useful: correctness gate, `python -m pinca_jax.plots`,
REM and reduced-scale sanity checks.

setlocal
cd /d "%~dp0"

echo === 1. interpreter ===
set PY=
for %%v in (3.14 3.13 3.12) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set PY=py -%%v
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,12) else 1)" >nul 2>&1 && set PY=python
)
if not defined PY (
  echo No Python ^>= 3.12 found. Install from python.org, tick "Add python.exe to PATH",
  echo then open a NEW terminal ^(PATH does not refresh in the current one^).
  exit /b 1
)
%PY% --version

echo === 2. venv ===
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || exit /b 1
  echo created .venv
) else (
  echo reusing existing .venv
)
set VPY=.venv\Scripts\python.exe

echo === 3. packages ===
"%VPY%" -m pip install --upgrade -q pip setuptools wheel || exit /b 1
REM torch is the migration-correctness reference only -- take the CPU build.
"%VPY%" -m pip install -q torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu || exit /b 1
"%VPY%" -m pip install -r requirements-jax.txt || exit /b 1
"%VPY%" -m pip install -q -e . || exit /b 1

echo === 4. verify ===
"%VPY%" -c "import sys,jax; print('  python ',sys.version.split()[0]); print('  jax    ',jax.__version__); print('  backend',jax.default_backend()); print('  devices',jax.devices())" || exit /b 1

echo.
echo Environment ready ^(CPU backend -- see note at the top of this file^).
echo.
echo Activate it in this shell:
echo   cmd.exe      .venv\Scripts\activate.bat
echo   PowerShell   .\.venv\Scripts\Activate.ps1
echo.
echo Then:
echo   python -m pytest tests/ -q      correctness gate
echo   python -m pinca_jax.plots       regenerate all benchmark figures
echo   run_gpu.bat smoke               reduced-scale end-to-end pass
endlocal
