#!/usr/bin/env bash
#SBATCH --job-name=enzymado_train
#SBATCH --array=0-13
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --output=scripts/logs/train_%A_%a.out
#SBATCH --nodelist=gpu4

# Run one embedding configuration per array task on the legacy SLURM setup.
#
# Run from the git root:
#   sbatch scripts/run_training_slurm.sh

source /home/ralmadamonter/miniconda3/etc/profile.d/conda.sh

REPO="/uss/ralmadamonter/enzymado-benchmark"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

mkdir -p scripts/logs
mkdir -p results/predictions

# Embedding model order (must match embeddings_models_output.json keys, 0-indexed)
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

EMB_NAME="${EMB_MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task $SLURM_ARRAY_TASK_ID → embedding: $EMB_NAME"

TRAIN_ARGS=(
    benchmark_card.json
    embeddings_models_output.json
    results/predictions
    --emb_name "$EMB_NAME"
    --results_card results_card.json
    --embedding_storage_format "${EMBEDDING_STORAGE_FORMAT:-safetensors}"
    --residue_store_mode "${RESIDUE_STORE_MODE:-auto}"
)

if [[ -n "${EMBEDDING_MODES:-}" ]]; then
    IFS=',' read -r -a _embedding_modes <<< "${EMBEDDING_MODES}"
    for _embedding_mode in "${_embedding_modes[@]}"; do
        if [[ -n "${_embedding_mode}" ]]; then
            TRAIN_ARGS+=(--embedding_mode "${_embedding_mode}")
        fi
    done
fi

if [[ "${OVERWRITE_RESULTS:-0}" == "1" || "${OVERWRITE_RESULTS:-0}" == "true" || "${OVERWRITE_RESULTS:-0}" == "TRUE" ]]; then
    TRAIN_ARGS+=(--overwrite_results)
fi

conda run --no-capture-output -n prot5_blackwell python -u src/training/train.py "${TRAIN_ARGS[@]}"
