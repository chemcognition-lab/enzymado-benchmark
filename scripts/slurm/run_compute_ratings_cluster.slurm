#!/usr/bin/env bash
#SBATCH --job-name=enzymado_ratings
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

# Roots default to the script's own defaults (results/predictions and
# results/bootstrap), which is the pre-protocol-v3 tree. Always set these when
# rating a different run root.
: "${PREDICTIONS_ROOT:=}"
: "${BOOTSTRAP_ROOT:=}"
: "${RATINGS_ROOT:=}"
: "${N_BATTLES:=1000}"

ARGS=(--n-battles "${N_BATTLES}")
if [[ -n "${PREDICTIONS_ROOT}" ]]; then
    ARGS+=(
        --properties-predictions-root "${PREDICTIONS_ROOT}"
        --real-case-no-attention-predictions-root "${PREDICTIONS_ROOT}"
        --real-case-attention-predictions-root "${PREDICTIONS_ROOT}"
    )
fi
if [[ -n "${BOOTSTRAP_ROOT}" ]]; then
    ARGS+=(
        --properties-bootstrap-root "${BOOTSTRAP_ROOT}"
        --real-case-no-attention-bootstrap-root "${BOOTSTRAP_ROOT}"
        --real-case-attention-bootstrap-root "${BOOTSTRAP_ROOT}"
    )
fi
if [[ -n "${RATINGS_ROOT}" ]]; then
    ARGS+=(--ratings-root "${RATINGS_ROOT}")
fi

echo "Predictions root: ${PREDICTIONS_ROOT:-<default>}"
echo "Bootstrap root: ${BOOTSTRAP_ROOT:-<default>}"
echo "Ratings root: ${RATINGS_ROOT:-<default>}"
echo "N battles: ${N_BATTLES}"
python -u "${REPO}/scripts/results/compute_ratings.py" "${ARGS[@]}"
