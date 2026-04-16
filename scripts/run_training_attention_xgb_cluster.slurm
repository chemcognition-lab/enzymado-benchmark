#!/usr/bin/env bash
#SBATCH --job-name=enzymado_attn_xgb
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out

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

if [[ -f "${REPO}/cluster_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO}/cluster_env.sh"
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/cluster_scratch_paths.sh"
init_cluster_runtime_dirs

if [[ -z "${CARD_PATH:-}" ]]; then
    if [[ -f "${REPO}/benchmark_card_attention_xgb.json" ]]; then
        CARD_PATH="$(ensure_scratch_card "${REPO}/benchmark_card_attention_xgb.json")"
    elif [[ -f "${REPO}/benchmark_card_attention.json" ]]; then
        CARD_PATH="$(ensure_scratch_card "${REPO}/benchmark_card_attention.json")"
    else
        CARD_PATH="$(ensure_scratch_card "${REPO}/benchmark_card.json")"
    fi
fi
: "${FEATURES_DIR:=${SCRATCH_ROOT}/results/attention_features}"
: "${OUTPUT_DIR:=${SCRATCH_ROOT}/results/predictions_attention_xgb}"
: "${SOURCE_MODEL_PREFIX:=attention_sgd_}"

EMB_MODELS=(
    ankh3_large
    ankh3_xl
    ankh_base
    ankh_large
    esm2_3B
    esm2_650M
    esm3_open
    esmc_300m
    esmc_600m
    profluent_e1_300m
    profluent_e1_600m
    prost5
    prot5_xl_half_uniref50
    prot5_xl_uniref50
)

select_index() {
    if [[ -n "${EMB_NAME:-}" ]]; then
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
            echo "EMB_NAME=${EMB_NAME} provided; skipping duplicate array task ${SLURM_ARRAY_TASK_ID}."
            exit 0
        fi
        local index
        for index in "${!EMB_MODELS[@]}"; do
            if [[ "${EMB_MODELS[$index]}" == "${EMB_NAME}" ]]; then
                printf '%s\n' "${index}"
                return 0
            fi
        done
        echo "ERROR: Unknown EMB_NAME='${EMB_NAME}'." >&2
        echo "Known values: ${EMB_MODELS[*]}" >&2
        exit 1
    fi

    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "ERROR: Provide EMB_NAME for a single job, or submit with --array=0-13%2 for all embeddings." >&2
        exit 1
    fi

    printf '%s\n' "${SLURM_ARRAY_TASK_ID}"
}

TASK_INDEX="$(select_index)"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#EMB_MODELS[@]} )); then
    echo "ERROR: task index ${TASK_INDEX} out of range." >&2
    exit 1
fi

EMBEDDING_NAME="${EMB_MODELS[$TASK_INDEX]}"

module load python/3.13.2 git-lfs/3.4.0 >/dev/null 2>&1 || module load python/3.13.2 >/dev/null 2>&1

if [[ ! -d "${REPO}/.venv" ]]; then
    echo "ERROR: missing training environment ${REPO}/.venv" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

mkdir -p "${OUTPUT_DIR}"

echo "Job host: $(hostname)"
echo "Embedding model: ${EMBEDDING_NAME}"
echo "Card path: ${CARD_PATH}"
echo "Features dir: ${FEATURES_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Source model prefix: ${SOURCE_MODEL_PREFIX}"

TRAIN_ARGS=(
    "${CARD_PATH}"
    "${FEATURES_DIR}"
    "${OUTPUT_DIR}"
    --emb_name "${EMBEDDING_NAME}"
    --source_model_prefix "${SOURCE_MODEL_PREFIX}"
)

python -u "${REPO}/models/train_attention_xgb.py" "${TRAIN_ARGS[@]}"
