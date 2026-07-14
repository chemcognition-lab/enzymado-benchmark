#!/usr/bin/env bash
#SBATCH --job-name=enzymado_attn
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=josechemistry.qb@gmail.com

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

: "${CLUSTER_RUNTIME_ROOT:=${REPO}/.cluster_runtime}"
: "${UV_LINK_MODE:=copy}"
if [[ -z "${UV_CACHE_DIR:-}" || "${UV_CACHE_DIR}" == /scratch/* ]]; then
    UV_CACHE_DIR="${CLUSTER_RUNTIME_ROOT}/uv-cache"
fi
if [[ -z "${HF_HOME:-}" || "${HF_HOME}" == /scratch/* ]]; then
    HF_HOME="${CLUSTER_RUNTIME_ROOT}/hf-cache"
fi
if [[ -z "${TORCH_HOME:-}" || "${TORCH_HOME}" == /scratch/* ]]; then
    TORCH_HOME="${CLUSTER_RUNTIME_ROOT}/torch-cache"
fi
if [[ -z "${XDG_CACHE_HOME:-}" || "${XDG_CACHE_HOME}" == /scratch/* ]]; then
    XDG_CACHE_HOME="${CLUSTER_RUNTIME_ROOT}/xdg-cache"
fi
if [[ -z "${TMPDIR:-}" || "${TMPDIR}" == /scratch/* ]]; then
    TMPDIR="${CLUSTER_RUNTIME_ROOT}/tmp"
fi
export CLUSTER_RUNTIME_ROOT UV_CACHE_DIR HF_HOME TORCH_HOME XDG_CACHE_HOME TMPDIR UV_LINK_MODE
export PYTHONNOUSERSITE=1
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

: "${EMBEDDINGS_JSON:=${REPO}/embeddings_models_output.json}"
: "${ATTENTION_EMBEDDINGS_JSON:=${EMBEDDINGS_JSON}}"
: "${OUTPUT_DIR:=${REPO}/results/predictions}"
: "${MODEL_PREFIX:=}"
: "${TASK_NAMES:=}"
: "${TASK_SCOPES:=}"
: "${SPLIT_VARIANT:=}"
: "${SKIP_LABEL_KINDS:=}"
# Default to the embedding-intersection allowlist so every embedding is evaluated
# on the exact same subset of IDs. Override with `COMMON_IDS_PATH= sbatch ...` to
# disable, or point it elsewhere.
: "${COMMON_IDS_PATH:=${REPO}/results/embedding_intersection/common_ids.txt}"
# This launcher trains attention-pooling models exclusively; restricting the
# results embedding mode skips the redundant non-attention pass over the same
# embeddings_json.
: "${EMBEDDING_MODES:=attention}"
: "${OVERWRITE_RESULTS:=0}"

for _write_path in "${OUTPUT_DIR}" "${CLUSTER_RUNTIME_ROOT}" "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"; do
    if [[ "${_write_path}" == /scratch/* ]]; then
        echo "ERROR: training jobs may not write to scratch: ${_write_path}" >&2
        exit 1
    fi
done

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
echo "Embeddings json: ${EMBEDDINGS_JSON}"
echo "Attention embeddings json: ${ATTENTION_EMBEDDINGS_JSON}"
echo "Output dir: ${OUTPUT_DIR}"
if [[ -n "${MODEL_PREFIX}" ]]; then
    echo "Model prefix: ${MODEL_PREFIX}"
fi
if [[ -n "${TASK_NAMES}" ]]; then
    echo "Task names: ${TASK_NAMES}"
fi
if [[ -n "${TASK_SCOPES}" ]]; then
    echo "Task scopes: ${TASK_SCOPES}"
fi
if [[ -n "${SPLIT_VARIANT}" ]]; then
    echo "Split variant: ${SPLIT_VARIANT}"
fi
if [[ -n "${SKIP_LABEL_KINDS}" ]]; then
    echo "Skip label kinds: ${SKIP_LABEL_KINDS}"
fi
if [[ -n "${COMMON_IDS_PATH}" ]]; then
    echo "Allowed IDs: ${COMMON_IDS_PATH}"
fi
if [[ -n "${EMBEDDING_MODES}" ]]; then
    echo "Embedding modes: ${EMBEDDING_MODES}"
fi
echo "Overwrite results: ${OVERWRITE_RESULTS}"

TRAIN_ARGS=(
    "${EMBEDDINGS_JSON}"
    --output_dir "${OUTPUT_DIR}"
    --emb_name "${EMBEDDING_NAME}"
    --attention_embeddings_json "${ATTENTION_EMBEDDINGS_JSON}"
)

if [[ -n "${MODEL_PREFIX}" ]]; then
    TRAIN_ARGS+=(--model_prefix "${MODEL_PREFIX}")
fi

if [[ -n "${SPLIT_VARIANT}" ]]; then
    TRAIN_ARGS+=(--split_variant "${SPLIT_VARIANT}")
fi

if [[ -n "${TASK_NAMES}" ]]; then
    IFS=',' read -r -a _task_names <<< "${TASK_NAMES}"
    for _task_name in "${_task_names[@]}"; do
        if [[ -n "${_task_name}" ]]; then
            TRAIN_ARGS+=(--task_name "${_task_name}")
        fi
    done
fi

if [[ -n "${TASK_SCOPES}" ]]; then
    IFS=',' read -r -a _task_scopes <<< "${TASK_SCOPES}"
    for _task_scope in "${_task_scopes[@]}"; do
        if [[ -n "${_task_scope}" ]]; then
            TRAIN_ARGS+=(--task_scope "${_task_scope}")
        fi
    done
fi

if [[ -n "${SKIP_LABEL_KINDS}" ]]; then
    IFS=',' read -r -a _skip_kinds <<< "${SKIP_LABEL_KINDS}"
    for _skip_kind in "${_skip_kinds[@]}"; do
        if [[ -n "${_skip_kind}" ]]; then
            TRAIN_ARGS+=(--skip_label_kind "${_skip_kind}")
        fi
    done
fi

if [[ -n "${COMMON_IDS_PATH}" ]]; then
    TRAIN_ARGS+=(--allowed_ids "${COMMON_IDS_PATH}")
fi

if [[ -n "${EMBEDDING_MODES}" ]]; then
    IFS=',' read -r -a _embedding_modes <<< "${EMBEDDING_MODES}"
    for _embedding_mode in "${_embedding_modes[@]}"; do
        if [[ -n "${_embedding_mode}" ]]; then
            TRAIN_ARGS+=(--embedding_mode "${_embedding_mode}")
        fi
    done
fi

if [[ "${OVERWRITE_RESULTS}" == "1" || "${OVERWRITE_RESULTS}" == "true" || "${OVERWRITE_RESULTS}" == "TRUE" ]]; then
    TRAIN_ARGS+=(--overwrite_results)
fi

python -u "${REPO}/scripts/run_benchmark.py" "${TRAIN_ARGS[@]}"
