#!/usr/bin/env bash
#SBATCH --job-name=enzymado_embed
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

# Submit one embedding model per array task or override `EMBED_MODEL` directly.

# Submit examples:
#   AIP_ACCOUNT="$(ls -d ~/projects/*/ | head -n 1 | sed 's|/$||' | sed 's|.*/||')"
#   sbatch --account="$AIP_ACCOUNT" --array=0-13%2 scripts/run_embeddings_cluster.slurm
#   sbatch --account="$AIP_ACCOUNT" --export=ALL,EMBED_MODEL=esm3_open scripts/run_embeddings_cluster.slurm
#   sbatch --account="$AIP_ACCOUNT" --export=ALL,EMBED_MODEL=esm2_650M,PER_PROTEIN=1 scripts/run_embeddings_cluster.slurm

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
    # Local, ignored file for cluster-specific caches and optional HF_TOKEN.
    # shellcheck disable=SC1091
    source "${REPO}/cluster_env.sh"
fi

# shellcheck disable=SC1091
source "${REPO}/scripts/cluster_scratch_paths.sh"
init_cluster_runtime_dirs

: "${PER_PROTEIN:=0}"
: "${ALLOW_PARTIAL:=1}"
: "${EMBED_OUTPUT_ROOT:=${SCRATCH_ROOT}/data}"

export PER_PROTEIN ALLOW_PARTIAL EMBED_OUTPUT_ROOT

MODELS=(
    prot5_xl_half_uniref50
    prot5_xl_uniref50
    prost5
    esm2_650M
    esm2_3B
    esm3_open
    esmc_300m
    esmc_600m
    ankh_base
    ankh_large
    ankh3_large
    ankh3_xl
    profluent_e1_300m
    profluent_e1_600m
)

PYTHON_MODULES=(
    python/3.13.2
    python/3.13.2
    python/3.13.2
    python/3.13.2
    python/3.13.2
    python/3.12.4
    python/3.12.4
    python/3.12.4
    python/3.12.4
    python/3.12.4
    python/3.12.4
    python/3.12.4
    python/3.12.4
    python/3.12.4
)

VENVS=(
    .venv-prot5
    .venv-prot5
    .venv-prot5
    .venv-esm2
    .venv-esm2
    .venv-esm3-esmc
    .venv-esm3-esmc
    .venv-esm3-esmc
    .venv-ankh
    .venv-ankh
    .venv-ankh
    .venv-ankh
    .venv-profluent
    .venv-profluent
)

SCRIPTS=(
    scripts/enzyme_utils/embedding_scripts/prot5_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/prot5_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/prost5_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/esm2_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/esm2_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/esm3_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/esmc_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/esmc_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/ankh_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/ankh_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/ankh_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/ankh_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/profluent_embedding_script.py
    scripts/enzyme_utils/embedding_scripts/profluent_embedding_script.py
)

MODEL_REFS=(
    Rostlab/prot_t5_xl_half_uniref50-enc
    Rostlab/prot_t5_xl_uniref50
    Rostlab/ProstT5
    facebook/esm2_t33_650M_UR50D
    facebook/esm2_t36_3B_UR50D
    esm3-open
    esmc_300m
    esmc_600m
    ankh_base
    ankh_large
    ankh3_large
    ankh3_xl
    Profluent-Bio/E1-300m
    Profluent-Bio/E1-600m
)

MODEL_DIMS=(
    1024
    1024
    1024
    1280
    2560
    1536
    960
    1152
    768
    1536
    1536
    2560
    1024
    1280
)

select_index() {
    if [[ -n "${EMBED_MODEL:-}" ]]; then
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
            echo "EMBED_MODEL=${EMBED_MODEL} provided; skipping duplicate array task ${SLURM_ARRAY_TASK_ID}."
            exit 0
        fi
        local index
        for index in "${!MODELS[@]}"; do
            if [[ "${MODELS[$index]}" == "${EMBED_MODEL}" ]]; then
                printf '%s\n' "${index}"
                return 0
            fi
        done
        echo "ERROR: Unknown EMBED_MODEL='${EMBED_MODEL}'." >&2
        echo "Known values: ${MODELS[*]}" >&2
        exit 1
    fi

    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "ERROR: Provide EMBED_MODEL for a single job, or submit with --array=0-13%2 for all models." >&2
        exit 1
    fi

    printf '%s\n' "${SLURM_ARRAY_TASK_ID}"
}

TASK_INDEX="$(select_index)"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#MODELS[@]} )); then
    echo "ERROR: task index ${TASK_INDEX} out of range." >&2
    exit 1
fi

MODEL_NAME="${MODELS[$TASK_INDEX]}"
PYTHON_MODULE="${PYTHON_MODULES[$TASK_INDEX]}"
VENV_PATH="${VENVS[$TASK_INDEX]}"
SCRIPT_PATH="${SCRIPTS[$TASK_INDEX]}"
MODEL_REF="${MODEL_REFS[$TASK_INDEX]}"
MODEL_DIM="${MODEL_DIMS[$TASK_INDEX]}"

module load "${PYTHON_MODULE}" git-lfs/3.4.0 >/dev/null 2>&1 || module load "${PYTHON_MODULE}" >/dev/null 2>&1

if [[ ! -d "${REPO}/${VENV_PATH}" ]]; then
    echo "ERROR: missing virtual environment ${REPO}/${VENV_PATH}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${REPO}/${VENV_PATH}/bin/activate"

CSV_PATH="${REPO}/data/processed/protein_sequences_unique.csv"
FASTA_PATH="${REPO}/data/fastas/protein_sequences_unique.fasta"

if [[ "${PER_PROTEIN}" == "1" ]]; then
    OUT_DIR="${EMBED_OUTPUT_ROOT}/embeddings_mean_pooled/protein_sequences_unique"
else
    OUT_DIR="${EMBED_OUTPUT_ROOT}/embeddings/protein_sequences_unique"
fi

OUT_PATH="${OUT_DIR}/${MODEL_NAME}.h5"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${CSV_PATH}" ]]; then
    echo "ERROR: missing input CSV ${CSV_PATH}" >&2
    exit 1
fi

if [[ ! -f "${FASTA_PATH}" ]]; then
    python "${REPO}/scripts/enzyme_utils/make_fasta_from_unique_sequences.py" \
        --input "${CSV_PATH}" \
        --output "${FASTA_PATH}"
fi

if [[ "${MODEL_NAME}" == "esm3_open" && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: esm3_open requires HF_TOKEN to access the gated Hugging Face repo." >&2
    exit 1
fi

if [[ "${MODEL_NAME}" == "ankh3_large" || "${MODEL_NAME}" == "ankh3_xl" ]]; then
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
fi

echo "Job host: $(hostname)"
echo "Model: ${MODEL_NAME}"
echo "Python module: ${PYTHON_MODULE}"
echo "Virtualenv: ${VENV_PATH}"
echo "Script: ${SCRIPT_PATH}"
echo "Per-protein: ${PER_PROTEIN}"
echo "Output: ${OUT_PATH}"

python "${REPO}/${SCRIPT_PATH}" \
    --input "${FASTA_PATH}" \
    --output "${OUT_PATH}" \
    --model "${MODEL_REF}" \
    --per_protein "${PER_PROTEIN}"

EXPECTED_COUNT="$(grep -c '^>' "${FASTA_PATH}")"
if [[ "${PER_PROTEIN}" == "1" ]]; then
    VALIDATION_LAYOUT="pooled"
else
    VALIDATION_LAYOUT="residue"
fi
if python "${REPO}/scripts/enzyme_utils/validate_embedding_h5.py" \
    --h5 "${OUT_PATH}" \
    --expected-count "${EXPECTED_COUNT}" \
    --expected-dim "${MODEL_DIM}" \
    --layout "${VALIDATION_LAYOUT}"; then
    echo "Embedding file validated successfully."
elif [[ "${ALLOW_PARTIAL}" == "1" ]]; then
    echo "WARNING: validation failed, but ALLOW_PARTIAL=1 so the job will still exit successfully." >&2
else
    echo "ERROR: embedding output validation failed." >&2
    exit 1
fi
