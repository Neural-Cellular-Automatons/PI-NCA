#!/usr/bin/env bash
# Alias for run_paper.sh, kept because earlier docs, notes and muscle memory point here.
#
# There is exactly one launcher now, and one place where the stage list, the scale presets
# and the resume logic live (`python -m pinca_jax.runner`). Two scripts that each carry
# their own copy of the pipeline is how a launcher silently stops running the experiments
# added after it was written -- which had already happened once here.
exec bash "$(dirname "$0")/run_paper.sh" "$@"
