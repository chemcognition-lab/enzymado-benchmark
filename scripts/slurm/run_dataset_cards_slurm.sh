#!/usr/bin/env bash
#SBATCH --job-name=cards_umap
#SBATCH --array=0-13
#SBATCH --mem=32G
#SBATCH --cpus-per-task=40
#SBATCH --output=scripts/logs/cards_%A_%a.out

# Generates dataset cards (stats + UMAP plots) for all properties,
# one embedding model per array task.
#
# Run from the git root:
#   sbatch scripts/run_dataset_cards_slurm.sh          # all 14 models
#   sbatch --array=5 scripts/run_dataset_cards_slurm.sh  # test: esm2_650M only

source /home/ralmadamonter/miniconda3/etc/profile.d/conda.sh

REPO="/uss/ralmadamonter/enzymado-benchmark"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

mkdir -p scripts/logs

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

conda run --no-capture-output -n dimensionality_reduction python -u \
    scripts/data_prep/create_dataset_cards.py \
    --base_dir data/processed/properties \
    --emb_model "$EMB_NAME" \
    --embeddings_dir data/embeddings_mean_pooled/protein_sequences_unique \
    --seq_identity 0.5
