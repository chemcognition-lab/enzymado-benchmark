#!/usr/bin/env bash
#SBATCH --job-name=enzymado_train
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

# Submit examples:
#   AIP_ACCOUNT="$(ls -d ~/projects/*/ | head -n 1 | sed 's|/$||' | sed 's|.*/||')"
#   sbatch --account="$AIP_ACCOUNT" --array=0-13%2 scripts/run_training_killarney.slurm
#   sbatch --account="$AIP_ACCOUNT" --export=ALL,EMB_NAME=esm2_650M scripts/run_training_killarney.slurm
#   sbatch --account="$AIP_ACCOUNT" --export=ALL,EMBEDDINGS_JSON=${PWD}/embeddings_models_output_mean_pooled.json,OUTPUT_DIR=${PWD}/results/predictions_mean_pooled_mlp,MODEL_PREFIX=mlp_,SKIP_LABEL_KINDS=residue_binary_vector scripts/run_training_killarney.slurm

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

: "${UV_CACHE_DIR:=/scratch/${USER}/enzymado/uv-cache}"
: "${HF_HOME:=/scratch/${USER}/enzymado/hf-cache}"
: "${TORCH_HOME:=/scratch/${USER}/enzymado/torch-cache}"
: "${XDG_CACHE_HOME:=/scratch/${USER}/enzymado/xdg-cache}"
: "${TMPDIR:=/scratch/${USER}/enzymado/tmp}"
: "${UV_LINK_MODE:=copy}"
: "${CARD_PATH:=${REPO}/benchmark_card.json}"
: "${EMBEDDINGS_JSON:=${REPO}/embeddings_models_output.json}"
: "${OUTPUT_DIR:=${REPO}/results/predictions}"
: "${MODEL_PREFIX:=}"
: "${SKIP_LABEL_KINDS:=}"

export UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME TMPDIR UV_LINK_MODE
export PYTHONNOUSERSITE=1

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

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
echo "Embeddings json: ${EMBEDDINGS_JSON}"
echo "Output dir: ${OUTPUT_DIR}"
if [[ -n "${MODEL_PREFIX}" ]]; then
    echo "Model prefix: ${MODEL_PREFIX}"
fi
if [[ -n "${SKIP_LABEL_KINDS}" ]]; then
    echo "Skip label kinds: ${SKIP_LABEL_KINDS}"
fi

TRAIN_ARGS=(
    "${CARD_PATH}"
    "${EMBEDDINGS_JSON}"
    "${OUTPUT_DIR}"
    --emb_name "${EMBEDDING_NAME}"
)

if [[ -n "${MODEL_PREFIX}" ]]; then
    TRAIN_ARGS+=(--model_prefix "${MODEL_PREFIX}")
fi

if [[ -n "${SKIP_LABEL_KINDS}" ]]; then
    IFS=',' read -r -a _skip_kinds <<< "${SKIP_LABEL_KINDS}"
    for _skip_kind in "${_skip_kinds[@]}"; do
        if [[ -n "${_skip_kind}" ]]; then
            TRAIN_ARGS+=(--skip_label_kind "${_skip_kind}")
        fi
    done
fi

python -u "${REPO}/models/train.py" "${TRAIN_ARGS[@]}"
