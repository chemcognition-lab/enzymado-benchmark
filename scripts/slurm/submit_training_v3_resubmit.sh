#!/usr/bin/env bash
# Submit the full protocol-v3 benchmark as 22 arrays, longest first.
#
# Sizing comes from sacct on the runs that produced the published results:
#   - billing is max(cpu*321.88, mem*83.06, gpu*10300), so <=124G and <=32 cpus
#     cost exactly the same as the GPU alone. Anything above 124G bills more.
#   - the three profiles that need more are the ones with measured OOM/peaks.
#   - walltimes come from completed *and* timed-out jobs, so they are not
#     censored by the old limits that cut runs short.
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

# Arrays that were still running from the original 32-cpu submission when this
# resubmission went out; they were left alone rather than duplicated. Override
# with SKIP="..." to reuse the script for a different partial resubmission.
: "${SKIP:=sites_na_xgb_rand sites_na_xgb_simi bsite_attn_lin_rand bsite_na_mlp_rand}"

submit() {
    local name=$1 scope=$2 split=$3 mode=$4 manifest=$5 tasks=$6 skip=$7 prefix=$8 mem=$9 time=${10} cpus=${11}
    for _skip in ${SKIP}; do
        if [[ "${name}" == "${_skip}" ]]; then
            printf '# skipped (already running): v3_%s\n' "${name}"
            return 0
        fi
    done
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
        printf '%q ' "${cmd[@]}"; printf '\n'
    else
        local id; id=$("${cmd[@]}")
        printf '%-34s %s\n' "v3_${name}" "${id}"
    fi
}

for split in random similarity; do
    s=${split:0:4}
    # --- longest first: these define the finish date ---
    submit "bsite_attn_mlp_${s}"  properties "${split}" attention    "${RESIDUE}" binding_site '' attention_mlp_    124G 24:00:00 8
    submit "bsite_na_mlp_${s}"    properties "${split}" no_attention "${RESIDUE}" binding_site '' mlp_              200G 24:00:00 8
    submit "bsite_attn_lin_${s}"  properties "${split}" attention    "${RESIDUE}" binding_site '' attention_linear_ 124G 12:00:00 8
    submit "ec_attn_lin_${s}"     properties "${split}" attention    "${RESIDUE}" ec_number    '' attention_linear_ 124G 12:00:00 8
    submit "asite_attn_${s}"      properties "${split}" attention    "${RESIDUE}" active_sites '' ''                124G 12:00:00 8
    submit "sites_na_xgb_${s}"    properties "${split}" no_attention "${RESIDUE}" active_sites,binding_site '' xgb_ 256G 12:00:00 8
    # --- short: fill the gaps by backfill ---
    submit "ec_attn_mlp_${s}"     properties "${split}" attention    "${RESIDUE}" ec_number    '' attention_mlp_    124G 04:00:00 8
    submit "sites_na_lin_${s}"    properties "${split}" no_attention "${RESIDUE}" active_sites,binding_site '' linear_ 124G 04:00:00 8
    submit "pooled_attn_${s}"     properties "${split}" attention    "${RESIDUE}" '' residue_binary_vector,multilabel_ec '' 124G 04:00:00 8
    submit "pooled_na_${s}"       properties "${split}" no_attention "${POOLED}"  '' residue_binary_vector '' 124G 04:00:00 8
done
submit "realcase_attn" real_case random attention    "${RESIDUE}" '' '' '' 124G 04:00:00 8
submit "realcase_na"   real_case random no_attention "${POOLED}"  '' '' '' 124G 04:00:00 8
