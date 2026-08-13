#!/usr/bin/env bash
# Post-training evaluation: metrics.json per run, then bootstrap CIs.
#
# Submitted one task per job rather than one job for the whole tree, because the
# global form measured 81.8 GB / 10:56h for metrics and 17:48h for bootstrap
# (with timeouts at 4h and 24h respectively). Per task it parallelises and each
# job stays well inside its limit.
#
# Bootstrap reads metrics.json, so each bootstrap job waits on its own metrics
# job via --dependency=afterok.
#
# Note: FILTER_TASKS is comma-separated in the .slurm scripts, but `sbatch
# --export` also splits on commas, so exactly one task is passed per job.
set -euo pipefail

ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MET="${REPO}/scripts/slurm/run_compute_metrics_cluster.slurm"
BOOT="${REPO}/scripts/slurm/run_compute_bootstrap_metrics_cluster.slurm"
REPLICATES="${N_REPLICATES:-1000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOBLOG="${REPO}/logs/submitted_eval_${STAMP}.tsv"
mkdir -p "${REPO}/logs"
printf 'metrics_jobid\tbootstrap_jobid\ttask\tmem\n' > "${JOBLOG}"

# The three residue/multilabel tasks flatten to 10^8 predictions per run; the
# rest are protein-level and comparatively tiny.
HEAVY="ec_number binding_site active_sites"

is_heavy() { case " ${HEAVY} " in *" $1 "*) return 0;; *) return 1;; esac; }

TASKS="solubility ph_opt ec_number kcat km binding dtm ddg active_sites binding_site
align_amylase_activity align_amylase_expression align_amylase_stability
align_glucosidase_activity align_glucosidase_expression align_glucosidase_melting
cleeveland_galox erickson_petase"

for task in ${TASKS}; do
    if is_heavy "${task}"; then
        m_mem=180G; m_time=12:00:00; b_mem=180G; b_time=24:00:00
    else
        m_mem=64G;  m_time=04:00:00; b_mem=64G;  b_time=08:00:00
    fi

    mjid=$(sbatch --parsable \
        --account="${ACCOUNT}" --job-name="met_${task}" \
        --mem="${m_mem}" --time="${m_time}" \
        --output="${REPO}/logs/met_${task}_%j.out" \
        --export="ALL,FILTER_TASKS=${task},SKIP_EXISTING=0" \
        "${MET}")

    bjid=$(sbatch --parsable \
        --account="${ACCOUNT}" --job-name="boot_${task}" \
        --mem="${b_mem}" --time="${b_time}" \
        --dependency="afterok:${mjid}" \
        --output="${REPO}/logs/boot_${task}_%j.out" \
        --export="ALL,FILTER_TASKS=${task},SKIP_EXISTING=0,N_REPLICATES=${REPLICATES},SEED=42" \
        "${BOOT}")

    printf '%s\t%s\t%s\t%s\n' "${mjid}" "${bjid}" "${task}" "${m_mem}" >> "${JOBLOG}"
    echo "  ${task}: metrics=${mjid} bootstrap=${bjid} (${m_mem}/${m_time}, ${b_mem}/${b_time})"
done

echo
echo "Registro: ${JOBLOG}"
echo "Replicas de bootstrap: ${REPLICATES}"
