#!/usr/bin/env bash
# Relaunch only the four pooled arrays cancelled while the random splits were
# regenerated. Same sizing as submit_training_v3_resubmit.sh (124G / 8 cpus /
# 4h): the pooled tasks are the cheap ones, they just had to wait for the new
# split CSVs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT="${RUN_ROOT:-${REPO}/results/protocol_v3}"
OUT="${RUN_ROOT}/predictions"
LOGS="${RUN_ROOT}/logs"
ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
ARRAY="${ARRAY_SPEC:-0-13%14}"
MAIL="${MAIL_USER:-josechemistry.qb@gmail.com}"
MAILTYPE="${MAIL_TYPE:-FAIL,END}"
DRYRUN="${DRYRUN:-1}"
POOLED="${REPO}/embeddings_models_output_mean_pooled.json"
RESIDUE="${REPO}/embeddings_models_output.json"

mkdir -p "${LOGS}"

submit() {
    local name=$1 scope=$2 split=$3 mode=$4 manifest=$5 tasks=$6 skip=$7 prefix=$8 mem=$9 time=${10} cpus=${11}
    local launcher="${REPO}/scripts/slurm/run_training_cluster.slurm"
    [[ "${mode}" == attention ]] && launcher="${REPO}/scripts/slurm/run_training_attention_cluster.slurm"
    local -a cmd=(
        env "OUTPUT_DIR=${OUT}" "EMBEDDINGS_JSON=${manifest}"
        "ATTENTION_EMBEDDINGS_JSON=${RESIDUE}"
        "TASK_SCOPES=${scope}" "SPLIT_VARIANT=${split}" "EMBEDDING_MODES=${mode}"
        "TASK_NAMES=${tasks}" "SKIP_LABEL_KINDS=${skip}" "MODEL_PREFIX=${prefix}"
        "EMB_NAME=" "SEED=42" "OVERWRITE_RESULTS=0" "ATTACH_PROBABILITIES=0"
        "COMMON_IDS_PATH=${REPO}/results/embedding_intersection/common_ids.txt"
        sbatch --parsable "--account=${ACCOUNT}" "--array=${ARRAY}"
        --gres=gpu:l40s:1 "--cpus-per-task=${cpus}" "--mem=${mem}" "--time=${time}"
        "--mail-user=${MAIL}" "--mail-type=${MAILTYPE}"
        "--job-name=v3_${name}" "--output=${LOGS}/%x_%A_%a.out"
        --export=ALL "${launcher}"
    )
    if [[ "${DRYRUN}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n\n'
    else
        local id; id=$("${cmd[@]}")
        printf '%-24s %s\n' "v3_${name}" "${id}"
    fi
}

for split in random similarity; do
    s=${split:0:4}
    submit "pooled_attn_${s}" properties "${split}" attention    "${RESIDUE}" '' residue_binary_vector,multilabel_ec '' 124G 04:00:00 8
    submit "pooled_na_${s}"   properties "${split}" no_attention "${POOLED}"  '' residue_binary_vector '' 124G 04:00:00 8
done
