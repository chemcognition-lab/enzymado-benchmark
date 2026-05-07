#!/usr/bin/env bash
#SBATCH --job-name=enzymado_pool
#SBATCH --array=0-13
#SBATCH --mem=20G
#SBATCH --cpus-per-task=20
#SBATCH --output=scripts/logs/pool_%A_%a.out
#SBATCH --nodelist=gpu4

# Run from the git root:
#   sbatch scripts/run_pooling_slurm.sh

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

INPUT="$REPO/data/embeddings/protein_sequences_unique/${EMB_NAME}.h5"
OUTPUT="$REPO/data/embeddings_pooled/protein_sequences_unique/${EMB_NAME}.h5"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: input not found: $INPUT"
    exit 1
fi

conda run --no-capture-output -n prot5_blackwell python -u \
    scripts/embeddings/pool_embeddings.py "$INPUT" "$OUTPUT"
