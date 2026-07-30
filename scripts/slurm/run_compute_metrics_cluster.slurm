#!/usr/bin/env bash
#SBATCH --job-name=enzymado_compute_metrics
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="${SLURM_SUBMIT_DIR}"
else
    REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

ARGS=()
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
python -u "${REPO}/scripts/results/compute_metrics.py" "${ARGS[@]}"
