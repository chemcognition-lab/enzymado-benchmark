#!/usr/bin/env bash
#SBATCH --job-name=enzymado_bootstrap
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="${SLURM_SUBMIT_DIR}"
else
    REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi

module load python/3.13.2 >/dev/null 2>&1 || true

if [[ ! -d "${REPO}/.venv" ]]; then
    echo "ERROR: missing training environment ${REPO}/.venv" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

: "${FILTER_TASKS:=}"
: "${SKIP_EXISTING:=1}"
: "${N_REPLICATES:=1000}"
: "${SEED:=0}"
# Roots default to the script's own defaults (results/predictions), which is the
# pre-protocol-v3 tree. Always set these when scoring a different run root.
: "${PREDICTIONS_ROOT:=}"
: "${BOOTSTRAP_ROOT:=}"
: "${STATISTICS_ROOT:=}"

ARGS=(--n-replicates "${N_REPLICATES}" --seed "${SEED}")
if [[ -n "${PREDICTIONS_ROOT}" ]]; then
    ARGS+=(--predictions-root "${PREDICTIONS_ROOT}")
fi
if [[ -n "${BOOTSTRAP_ROOT}" ]]; then
    ARGS+=(--bootstrap-root "${BOOTSTRAP_ROOT}")
fi
if [[ -n "${STATISTICS_ROOT}" ]]; then
    ARGS+=(--statistics-root "${STATISTICS_ROOT}")
fi
if [[ "${SKIP_EXISTING}" == "1" || "${SKIP_EXISTING}" == "true" || "${SKIP_EXISTING}" == "TRUE" ]]; then
    ARGS+=(--skip-existing)
fi
if [[ -n "${FILTER_TASKS}" ]]; then
    IFS=',' read -r -a _tasks <<< "${FILTER_TASKS}"
    for _task in "${_tasks[@]}"; do
        if [[ -n "${_task}" ]]; then
            ARGS+=(--filter-task "${_task}")
        fi
    done
fi

echo "Filter tasks: ${FILTER_TASKS:-<none>}"
echo "Skip existing: ${SKIP_EXISTING}"
echo "N replicates: ${N_REPLICATES}"
echo "Predictions root: ${PREDICTIONS_ROOT:-<default>}"
echo "Bootstrap root: ${BOOTSTRAP_ROOT:-<default>}"
python -u "${REPO}/scripts/results/compute_bootstrap_metrics.py" "${ARGS[@]}"
