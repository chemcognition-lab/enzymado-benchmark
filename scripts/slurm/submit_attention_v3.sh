#!/usr/bin/env bash
# Retrain every global attention baseline with the residual attention block.
#
# Global (pooled) tasks only: the residue tasks keep their self-attention mixer,
# which did not change. Each attention head trains like its mean-pool
# counterpart (see model_registry), so this reruns linear and MLP attention for
# the eight global property tasks under both splits and for the real-case tasks.
#
# Runs land in an isolated tree; validate them, then swap the attention run
# directories into results/predictions and recompute metrics, bootstrap and
# ratings. Walltimes come from ankh3_xl, the slowest embedding: the full-batch
# linear head can take up to 200 epochs, so its ligand tasks get one job each.
#   DRYRUN=0 bash <this script>                                    training
#   DRYRUN=0 PHASE=attach AFTER=afterany:<id>:<id> bash <this script> probabilities
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT="${RUN_ROOT:-${REPO}/results/protocol_v3_attention}"
OUT="${RUN_ROOT}/predictions"
LOGS="${RUN_ROOT}/logs"
ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
ARRAY="${ARRAY_SPEC:-0-13%14}"
MAIL="${MAIL_USER:-josechemistry.qb@gmail.com}"
MAILTYPE="${MAIL_TYPE:-FAIL}"
DRYRUN="${DRYRUN:-1}"
PHASE="${PHASE:-train}"
AFTER="${AFTER:-}"
ONLY="${ONLY:-}"
EXCLUDE="${EXCLUDE:-kn001}"
# Resubmitting timed-out runs: they resume from their checkpoint as long as the
# code is unchanged. TIME_OVERRIDE replaces the walltime, AFTER_OVERRIDE chains
# the job (e.g. afternotok:<id>_<task> to resume only if a running task fails).
TIME_OVERRIDE="${TIME_OVERRIDE:-}"
RESIDUE="${REPO}/embeddings_models_output.json"
LAUNCHER="${REPO}/scripts/slurm/run_training_attention_cluster.slurm"

mkdir -p "${LOGS}"
cd "${REPO}"

submit() {
    local name=$1 scope=$2 split=$3 tasks=$4 prefix=$5 mem=$6 time=$7
    [[ -n "${ONLY}" && ! "${name}" =~ ${ONLY} ]] && return 0
    local attach=0 tag=""
    local -a dependency=() exclude=()
    if [[ "${PHASE}" == attach ]]; then
        attach=1 tag="pb_"
        [[ -n "${AFTER}" ]] && dependency=("--dependency=${AFTER}")
    fi
    [[ -n "${EXCLUDE}" ]] && exclude=("--exclude=${EXCLUDE}")
    [[ -n "${TIME_OVERRIDE}" ]] && time="${TIME_OVERRIDE}"
    [[ "${PHASE}" != attach && -n "${AFTER}" ]] && dependency=("--dependency=${AFTER}")
    local -a cmd=(
        env "OUTPUT_DIR=${OUT}" "EMBEDDINGS_JSON=${RESIDUE}"
        "ATTENTION_EMBEDDINGS_JSON=${RESIDUE}"
        "TASK_SCOPES=${scope}" "SPLIT_VARIANT=${split}" "EMBEDDING_MODES=attention"
        "TASK_NAMES=${tasks}" "SKIP_LABEL_KINDS=" "MODEL_PREFIX=${prefix}"
        "EMB_NAME=" "SEED=42" "OVERWRITE_RESULTS=0" "ATTACH_PROBABILITIES=${attach}"
        "COMMON_IDS_PATH=${REPO}/results/embedding_intersection/common_ids.txt"
        sbatch --parsable "--account=${ACCOUNT}" "--array=${ARRAY}"
        --gres=gpu:l40s:1 --cpus-per-task=32 "--mem=${mem}" "--time=${time}"
        "--mail-user=${MAIL}" "--mail-type=${MAILTYPE}"
        "--job-name=attn_${tag}${name}" "--output=${LOGS}/%x_%A_%a.out"
        "${dependency[@]}" "${exclude[@]}" --export=ALL "${LAUNCHER}"
    )
    if [[ "${DRYRUN}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"; printf '\n'
    else
        local id; id=$("${cmd[@]}")
        printf '%-32s %s\n' "attn_${tag}${name}" "${id}"
    fi
}

if [[ "${PHASE}" == attach ]]; then
    # Inference on the test split only, for the classification tasks.
    for split in random similarity; do
        s=${split:0:4}
        submit "cls_${s}" properties "${split}" solubility,binding '' 124G 02:00:00
        submit "ec_${s}"  properties "${split}" ec_number         '' 124G 03:00:00
    done
    exit 0
fi

for split in random similarity; do
    s=${split:0:4}
    submit "ec_lin_${s}"      properties "${split}" ec_number                  attention_linear_ 124G 24:00:00
    submit "ec_mlp_${s}"      properties "${split}" ec_number                  attention_mlp_    124G 12:00:00
    submit "km_lin_${s}"      properties "${split}" km                         attention_linear_ 124G 12:00:00
    submit "binding_lin_${s}" properties "${split}" binding                    attention_linear_ 124G 12:00:00
    submit "kcat_lin_${s}"    properties "${split}" kcat                       attention_linear_ 124G 08:00:00
    submit "lig_mlp_${s}"     properties "${split}" kcat,km,binding            attention_mlp_    124G 08:00:00
    submit "small_lin_${s}"   properties "${split}" ph_opt,solubility,ddg,dtm  attention_linear_ 124G 08:00:00
    submit "small_mlp_${s}"   properties "${split}" ph_opt,solubility,ddg,dtm  attention_mlp_    124G 06:00:00
done
submit "rc_lin" real_case random '' attention_linear_ 124G 12:00:00
submit "rc_mlp" real_case random '' attention_mlp_    124G 06:00:00
