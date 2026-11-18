#!/bin/bash
# Chain the MLIP production run with Slurm dependencies (2026-11-11).
#
# Stage A, proteins matched in the local AFDB files:
#   prep arrays PREP_JOBS -> slicing job -> one embedding array per model.
# Stage B, proteins only the AFDB API had:
#   API fetch FETCH_JOB -> merge manifests -> prep array (tag api) ->
#   slicing job -> one small embedding array per model (shard tag a).
#
# Slices per model aim at about 1 h per task from the pilot speeds; the
# 3 h limit leaves room for the largest-protein slices. %N caps how many GPUs
# one model uses at a time.
#
# Usage, from a login node in the repo root:
#   PREP_JOBS=7467889:7467930 FETCH_JOB=7467390 \
#       bash scripts/structures/launch_mlip_production.sh

set -euo pipefail
: "${PREP_JOBS:?colon-separated prep array job ids}" "${FETCH_JOB:?API fetch job id}"

REPO=/project/6111759/chemist/enzymado/enzymado-benchmark
SP=/scratch/chemist/enzymado/data/structures_prod
API=/scratch/chemist/enzymado/data/afdb_api
PY=/scratch/chemist/enzymado/venvs/structures/bin/python
SLICER="${PY} ${REPO}/scripts/structures/make_embedding_slices.py --manifest_dir ${SP}/prepare_manifests"
EMBED="${REPO}/embedding_scripts/mlip_embed_array.sbatch"
CPU_JOB=(--account=aip-bensl --cpus-per-task=1 --mem=8G --time=1:00:00 --parsable)

# model n_slices max_concurrent
MODELS=(
    "aimnet2_wb97m_d3_0 20 10"
    "so3lr_tauri_s_2606 20 10"
    "so3lr_tauri_m_2606 20 10"
    "so3lr_tauri_l_2606 80 20"
    "mace_off23_small 40 20"
    "mace_off23_medium 40 20"
    "mace_mp_0b3_medium 80 20"
    "mace_off23_large 250 30"
)

cd "${REPO}"

# Stage A.
slice_a=$(sbatch "${CPU_JOB[@]}" --job-name=mlip_slices_local \
    --output="${SP}/logs/slices_local_%j.log" --dependency="afterany:${PREP_JOBS}" \
    --wrap "set -e; for n in 20 40 80 250; do ${SLICER} --pattern 'local_*.csv' --n_slices \$n --output_dir ${SP}/slices_local_\$n; done")
echo "slicing (local): ${slice_a}"
for spec in "${MODELS[@]}"; do
    read -r model n cap <<< "${spec}"
    jid=$(sbatch --parsable --job-name="${model}" --array="0-$((n - 1))%${cap}" \
        --dependency="afterok:${slice_a}" \
        --export="ALL,MODEL=${model},SLICES=${SP}/slices_local_${n},TAG=s" "${EMBED}")
    echo "embed ${model} local: ${jid} (${n} slices, at most ${cap} at once)"
done

# Stage B.
merge=$(sbatch "${CPU_JOB[@]}" --job-name=mlip_merge_api \
    --output="${SP}/logs/merge_api_%j.log" --dependency="afterok:${FETCH_JOB}" \
    --wrap "${PY} ${REPO}/scripts/structures/merge_fetch_manifests.py --local_manifest ${SP}/fetch_manifest_local.csv --api_manifest ${API}/fetch_manifest.csv --output_dir ${SP}")
echo "merge manifests: ${merge}"
# About 1,700 API rows are expected; 12 tasks of 250 cover up to 3,000, and
# tasks past the end have no rows and exit at once.
prep_b=$(sbatch --parsable --array=0-11 --dependency="afterok:${merge}" \
    --export="ALL,FETCH_MANIFEST=${SP}/fetch_manifest_api.csv,CHUNK=250,TAG=api" \
    "${REPO}/scripts/structures/prepare_structures_array.sbatch")
echo "prep (api): ${prep_b}"
slice_b=$(sbatch "${CPU_JOB[@]}" --job-name=mlip_slices_api \
    --output="${SP}/logs/slices_api_%j.log" --dependency="afterany:${prep_b}" \
    --wrap "${SLICER} --pattern 'api_*.csv' --n_slices 4 --output_dir ${SP}/slices_api_4")
echo "slicing (api): ${slice_b}"
for spec in "${MODELS[@]}"; do
    read -r model _ _ <<< "${spec}"
    jid=$(sbatch --parsable --job-name="${model}_api" --array=0-3 \
        --dependency="afterok:${slice_b}" \
        --export="ALL,MODEL=${model},SLICES=${SP}/slices_api_4,TAG=a" "${EMBED}")
    echo "embed ${model} api: ${jid}"
done
