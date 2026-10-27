#!/usr/bin/env bash
# Run a configuration selected by the submitter's SLURM array.
# Example: python scripts/mlp_hpo/hpo_job_submitter.py --dry-run
set -euo pipefail

# The submitter supplies the Python executable, runner and all path arguments.
# Load the cluster Python runtime if modules are available; local environments
# can instead pass their own executable with the submitter's --python option.
if type module >/dev/null 2>&1; then
    module load python/3.13.2
fi
export PYTHONNOUSERSITE=1
exec "$@" --config-index "${SLURM_ARRAY_TASK_ID:?Missing SLURM array index}"
