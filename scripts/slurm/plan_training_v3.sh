#!/usr/bin/env bash
# Print an isolated protocol-v3 submission plan without submitting jobs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT="${RUN_ROOT:-${REPO}/results/protocol_v3}"
AIP_ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
ARRAY_SPEC="${ARRAY_SPEC:-0-13%2}"
ATTACH_PROBABILITIES="${ATTACH_PROBABILITIES:-0}"

printf '# Review before executing: eight arrays, at most two tasks per array by default.\n'
printf 'cd %q\n' "${REPO}"
printf 'mkdir -p %q\n' "${RUN_ROOT}/logs"

plan_job() {
    local scope=$1 split=$2 mode=$3 manifest=$4 tasks=$5 skip=$6
    local launcher="${REPO}/scripts/slurm/run_training_cluster.slurm"
    if [[ "${mode}" == attention ]]; then
        launcher="${REPO}/scripts/slurm/run_training_attention_cluster.slurm"
    fi
    local command=(
        env "OUTPUT_DIR=${RUN_ROOT}/predictions"
        "EMBEDDINGS_JSON=${REPO}/${manifest}"
        "ATTENTION_EMBEDDINGS_JSON=${REPO}/embeddings_models_output.json"
        "TASK_SCOPES=${scope}" "SPLIT_VARIANT=${split}"
        "EMBEDDING_MODES=${mode}" "TASK_NAMES=${tasks}"
        "SKIP_LABEL_KINDS=${skip}" "MODEL_PREFIX=" "EMB_NAME="
        "SEED=42" "OVERWRITE_RESULTS=0" "EXPERIMENT_NAME="
        "ATTENTION_LR=" "ATTENTION_GRADIENT_CLIP_NORM="
        "ATTENTION_BATCH_DIAGNOSTICS=0"
        "COMMON_IDS_PATH=${REPO}/results/embedding_intersection/common_ids.txt"
        "ATTACH_PROBABILITIES=${ATTACH_PROBABILITIES}"
        sbatch "--account=${AIP_ACCOUNT}" "--array=${ARRAY_SPEC}"
        "--job-name=v3_${scope}_${split}_${mode}${tasks:+_sites}"
        "--output=${RUN_ROOT}/logs/%x_%A_%a.out" --export=ALL "${launcher}"
    )
    printf '%q ' "${command[@]}"
    printf '\n'
}

for split in random similarity; do
    plan_job properties "${split}" no_attention \
        embeddings_models_output_mean_pooled.json '' residue_binary_vector
    plan_job properties "${split}" no_attention \
        embeddings_models_output.json active_sites,binding_site ''
    plan_job properties "${split}" attention embeddings_models_output.json '' ''
done
plan_job real_case random no_attention embeddings_models_output_mean_pooled.json '' ''
plan_job real_case random attention embeddings_models_output.json '' ''
