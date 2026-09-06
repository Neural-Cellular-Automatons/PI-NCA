@echo off
REM Alias for run_paper.bat, kept because earlier docs and notes point here.
REM
REM There is exactly one launcher now, and one place where the stage list, the scale
REM presets and the resume logic live (python -m pinca_jax.runner). Two scripts each
REM carrying their own copy of the pipeline is how a launcher silently stops running the
REM experiments added after it was written -- which had already happened once here.
call "%~dp0run_paper.bat" %*
