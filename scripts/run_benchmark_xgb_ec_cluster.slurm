#!/usr/bin/env bash
#SBATCH --job-name=enzymado_xgb_ec_bench
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

# Submit examples:
#   AIP_ACCOUNT="$(ls -d ~/projects/*/ | head -n 1 | sed 's|/$||' | sed 's|.*/||')"
#   sbatch --account="$AIP_ACCOUNT" --export=ALL,EMB_NAME=ankh3_large scripts/run_benchmark_xgb_ec_cluster.slurm
#   sbatch --account="$AIP_ACCOUNT" --export=ALL,EMB_NAME=ankh3_large,MAX_TRAIN_ROWS=10000 scripts/run_benchmark_xgb_ec_cluster.slurm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO="${SLURM_SUBMIT_DIR}"
else
    REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi

if [[ -f "${REPO}/cluster_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO}/cluster_env.sh"
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/cluster_scratch_paths.sh"
init_cluster_runtime_dirs

: "${EMB_NAME:=ankh3_large}"
: "${MAX_TRAIN_ROWS:=10000}"
: "${SEED:=0}"
: "${PROFILE_DIR:=${SCRATCH_ROOT}/results/xgb_bench_profiles/${EMB_NAME}}"
: "${OUTPUT_JSON:=${SCRATCH_ROOT}/results/xgb_benchmark_${EMB_NAME}.json}"
: "${COMMON_IDS_PATH:=}"

if [[ -z "${CARD_PATH:-}" ]]; then
    CARD_PATH="$(ensure_scratch_card "${REPO}/benchmark_card.json")"
fi

if [[ -z "${EMBEDDINGS_JSON:-}" ]]; then
    EMBEDDINGS_JSON="${REPO}/embeddings_models_output_mean_pooled.json"
fi

module load python/3.13.2 git-lfs/3.4.0 >/dev/null 2>&1 || module load python/3.13.2 >/dev/null 2>&1

if [[ ! -d "${REPO}/.venv" ]]; then
    echo "ERROR: missing training environment ${REPO}/.venv" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

mkdir -p "$(dirname "${OUTPUT_JSON}")" "${PROFILE_DIR}"

echo "Job host: $(hostname)"
echo "Embedding model: ${EMB_NAME}"
echo "Card path: ${CARD_PATH}"
echo "Embeddings json: ${EMBEDDINGS_JSON}"
echo "Max train rows: ${MAX_TRAIN_ROWS}"
echo "Seed: ${SEED}"
echo "Profile dir: ${PROFILE_DIR}"
echo "Output json: ${OUTPUT_JSON}"
if [[ -n "${COMMON_IDS_PATH}" ]]; then
    echo "Allowed IDs: ${COMMON_IDS_PATH}"
fi

BENCH_ARGS=(
    "${CARD_PATH}"
    "${EMBEDDINGS_JSON}"
    --emb_name "${EMB_NAME}"
    --max_train_rows "${MAX_TRAIN_ROWS}"
    --seed "${SEED}"
    --profile_dir "${PROFILE_DIR}"
    --output_json "${OUTPUT_JSON}"
)

if [[ -n "${COMMON_IDS_PATH}" ]]; then
    BENCH_ARGS+=(--allowed_ids "${COMMON_IDS_PATH}")
fi

python -u "${REPO}/scripts/benchmark_xgb_ec.py" "${BENCH_ARGS[@]}"
