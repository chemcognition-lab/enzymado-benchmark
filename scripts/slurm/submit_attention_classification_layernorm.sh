#!/usr/bin/env bash
# Submit all attention-classification reruns with stabilized optimization.

set -euo pipefail

ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN="${REPO}/scripts/slurm/run_training_attention_cluster.slurm"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-attention_regression_layernorm_clip_v1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOBLOG="${REPO}/logs/attention_classification_layernorm_${STAMP}.tsv"

mkdir -p "${REPO}/logs"
printf 'jobid\tname\tmem\ttime\texports\n' > "${JOBLOG}"

submit() {
    local name="$1"
    local mem="$2"
    local time_limit="$3"
    local exports="$4"
    local job_id

    job_id="$(
        sbatch --parsable \
            --account="${ACCOUNT}" \
            --job-name="${name}" \
            --array=0-13 \
            --mem="${mem}" \
            --time="${time_limit}" \
            --mail-type=FAIL \
            --output="${REPO}/logs/${name}_%A_%a.out" \
            --export="ALL,SEED=42,EXPERIMENT_NAME=${EXPERIMENT_NAME},OVERWRITE_RESULTS=0,ATTENTION_PRE_LAYER_NORM=1,ATTENTION_GRADIENT_CLIP_NORM=1.0,${exports}" \
            "${TRAIN}"
    )"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${name}" "${mem}" "${time_limit}" "${exports}" \
        >> "${JOBLOG}"
    printf '%s\t%s\n' "${job_id}" "${name}"
}

for split in random similarity; do
    for family in linear mlp; do
        prefix="attention_${family}_"
        submit \
            "lncls_ec_${split}_${family}" \
            160G \
            24:00:00 \
            "TASK_SCOPES=properties,TASK_NAMES=ec_number,SPLIT_VARIANT=${split},MODEL_PREFIX=${prefix}"
        submit \
            "lncls_binding_site_${split}_${family}" \
            160G \
            24:00:00 \
            "TASK_SCOPES=properties,TASK_NAMES=binding_site,SPLIT_VARIANT=${split},MODEL_PREFIX=${prefix}"
        submit \
            "lncls_active_sites_${split}_${family}" \
            160G \
            16:00:00 \
            "TASK_SCOPES=properties,TASK_NAMES=active_sites,SPLIT_VARIANT=${split},MODEL_PREFIX=${prefix}"
    done
done

for task in binding solubility; do
    for split in random similarity; do
        submit \
            "lncls_${task}_${split}" \
            64G \
            08:00:00 \
            "TASK_SCOPES=properties,TASK_NAMES=${task},SPLIT_VARIANT=${split}"
    done
done

printf 'Submission log: %s\n' "${JOBLOG}"
