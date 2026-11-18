#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --job-name=mlip_embed
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=kn051,kn092,kn101,kn104,kn116,kn117
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/chemist/enzymado/data/embeddings/mlip/logs/%x_%A_%a.log

# One array task embeds one slice (SLICES/slice_<task>.txt) with one MLIP.
# All tasks of a model write to the same <MODEL>_shards directory with their
# own shard prefix (<TAG><task>_), and to their own pooled file under
# pooled_slices/<MODEL>/; merge_pooled_slices.py joins those afterwards.
# TAG (default s) must differ between slice sets of the same model, for
# example s for the first set and a for proteins added later.
# Re-submitting the same task resumes: IDs already in its shards are skipped.
#
# Submit from a login node, for example:
#   sbatch --job-name=so3lr_m --array=0-39 \
#       --export=ALL,MODEL=so3lr_tauri_m_2606,SLICES=/scratch/chemist/enzymado/data/structures_prod/slices_40 \
#       embedding_scripts/mlip_embed_array.sbatch
# Values passed with --export must not contain commas.

set -euo pipefail
: "${MODEL:?set MODEL}" "${SLICES:?set SLICES}"
TAG=${TAG:-s}

REPO=/project/6111759/chemist/enzymado/enzymado-benchmark
V=/scratch/chemist/enzymado/venvs
W=/scratch/chemist/enzymado/mlip-models
MACE_W=/scratch/chemist/enzymado/mace-models
STRUCT=/scratch/chemist/enzymado/data/structures_prod/prepared
EMB=${EMB_DIR:-/scratch/chemist/enzymado/data/embeddings/mlip}  # EMB_DIR: tests only

task=$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")
ids="${SLICES}/slice_${task}.txt"
[ -f "${ids}" ] || { echo "missing ${ids}"; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${REPO}/embedding_scripts" PYTHONUNBUFFERED=1

mace() { echo "${V}/mace/bin/python ${REPO}/embedding_scripts/mace_embedding_script.py --device cuda $*"; }
case "${MODEL}" in
    mace_off23_small)   cmd=$(mace --family off --model_path "${MACE_W}/MACE-OFF23_small.model") ;;
    mace_off23_medium)  cmd=$(mace --family off --model_path "${MACE_W}/MACE-OFF23_medium.model" --chunk_atoms 16000) ;;
    mace_off23_large)   cmd=$(mace --family off --model_path "${MACE_W}/MACE-OFF23_large.model" --chunk_atoms 4000) ;;
    mace_mp_0b3_medium) cmd=$(mace --family mp --model_path "${MACE_W}/mace-mp-0b3-medium.model" --chunk_atoms 10000) ;;
    aimnet2_wb97m_d3_0)
        cmd="${V}/aimnet2/bin/python ${REPO}/embedding_scripts/aimnet2_embedding_script.py --device cuda --model_path ${W}/aimnet2/aimnet2_wb97m_d3_0.pt" ;;
    so3lr_tauri_s_2606|so3lr_tauri_m_2606|so3lr_tauri_l_2606)
        size=$(echo "${MODEL}" | sed -E 's/so3lr_tauri_([sml])_2606/\1/' | tr 'sml' 'SML')
        export JAX_PLATFORMS=cuda  # fail instead of falling back to CPU
        cmd="${V}/so3lr/bin/python ${REPO}/embedding_scripts/so3lr_embedding_script.py --model ${size}" ;;
    *) echo "unknown MODEL ${MODEL}"; exit 1 ;;
esac

mkdir -p "${EMB}/${MODEL}_shards" "${EMB}/pooled_slices/${MODEL}"
echo "host $(hostname) model ${MODEL} slice ${TAG}${task} ($(wc -l < "${ids}") ids) start $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${cmd} \
    -i "${STRUCT}" \
    -o "${EMB}/${MODEL}_shards" \
    --mean_pooled_output "${EMB}/pooled_slices/${MODEL}/${TAG}${task}.safetensors" \
    --ids_file "${ids}" \
    --shard_prefix "${TAG}${task}_"
echo "end $(date -Iseconds)"
