#!/bin/bash
#SBATCH --account=aip-bensl
#SBATCH --job-name=prep_structures
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/chemist/enzymado/data/structures_prod/logs/prep_%A_%a.log

# One array task prepares CHUNK consecutive ok rows of FETCH_MANIFEST, on one
# CPU (OpenMM single-threaded, so hydrogens are repeatable). 250 rows take
# about 4 h at ~60 s per protein (N^1.55 scaling measured on the pilot).
# Each task appends to its own manifest, prepare_manifests/<TAG>_<task>.csv,
# so tasks never
# write the same file. Re-submitting the same array resumes: rows already in
# a task's manifest are skipped, except status error, which is retried.
#
# Submit from a login node, for example the first 30 tasks of 250 rows:
#   sbatch --array=0-29 --export=ALL,FETCH_MANIFEST=/scratch/chemist/enzymado/data/structures_prod/fetch_manifest_local.csv,CHUNK=250,TAG=local \
#       scripts/structures/prepare_structures_array.sbatch
# Values passed with --export must not contain commas.

set -euo pipefail
: "${FETCH_MANIFEST:?set FETCH_MANIFEST}" "${CHUNK:?set CHUNK}" "${TAG:?set TAG}"

REPO=/project/6111759/chemist/enzymado/enzymado-benchmark
OUT=/scratch/chemist/enzymado/data/structures_prod
PY=/scratch/chemist/enzymado/venvs/structures/bin/python

export OPENMM_CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

task=${SLURM_ARRAY_TASK_ID}
start=$((task * CHUNK))
stop=$((start + CHUNK))
mkdir -p "${OUT}/prepare_manifests"
manifest="${OUT}/prepare_manifests/${TAG}_$(printf '%04d' "${task}").csv"

echo "host $(hostname) task ${task} rows [${start}, ${stop}) start $(date -Iseconds)"
"${PY}" -u "${REPO}/scripts/structures/prepare_structures.py" \
    --fetch_manifest "${FETCH_MANIFEST}" \
    --input "${REPO}/data/processed/protein_pool.csv" \
    --output_dir "${OUT}" \
    --start "${start}" --stop "${stop}" \
    --prepare_manifest "${manifest}"
echo "end $(date -Iseconds)"
