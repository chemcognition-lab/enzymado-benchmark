#!/usr/bin/env bash
#SBATCH --job-name=task_pooled
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=65536M
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=NONE
#
# Train every pooled (no-attention) model on one task against the mean-pooled
# manifest written by submit_task_embeddings.sh, on both splits, then score.
# XGBoost's standard entries request CUDA, hence the GPU.
#
#   sbatch --account=aip-bensl scripts/slurm/train_task_pooled.slurm
#   sbatch --account=aip-bensl --export=ALL,TASK=km scripts/slurm/train_task_pooled.slurm

set -euo pipefail

: "${TASK:=ph_opt}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module load python/3.12.4 >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

MANIFEST="${REPO}/embeddings_${TASK}_pooled.json"
OUT_ROOT="${REPO}/results_${TASK}"

for SPLIT in random similarity; do
    echo "=== ${TASK} / ${SPLIT} ==="
    python -u scripts/run_benchmark.py "${MANIFEST}" \
        --output_dir "${OUT_ROOT}/predictions" \
        --task_name "${TASK}" \
        --task_scope properties \
        --split_variant "${SPLIT}" \
        --embedding_mode no_attention
done

python -u scripts/results/compute_metrics.py \
    --predictions-root "${OUT_ROOT}/predictions" \
    --statistics-root "${OUT_ROOT}/statistics" \
    --benchmark-card benchmark_card.json \
    --filter-task "${TASK}"
