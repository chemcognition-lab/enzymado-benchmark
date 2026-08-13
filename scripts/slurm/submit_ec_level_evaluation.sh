#!/usr/bin/env bash
# Recompute EC metrics by hierarchy level, then bootstrap threshold metrics.
#
# Metrics use exact threshold-free AUROC/AUPRC on validation and test.  F1,
# precision, recall, and accuracy use one validation-calibrated threshold per
# EC level.  Bootstrap resamples test proteins and intentionally excludes the
# ranking metrics, which are already stored as exact point estimates.
set -euo pipefail

ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METRICS_SCRIPT="${REPO}/scripts/slurm/run_compute_metrics_cluster.slurm"
BOOTSTRAP_SCRIPT="${REPO}/scripts/slurm/run_compute_bootstrap_metrics_cluster.slurm"
SHARDS="${EC_EVAL_SHARDS:-28}"
REPLICATES="${N_REPLICATES:-1000}"

if (( SHARDS <= 0 )); then
    echo "ERROR: EC_EVAL_SHARDS must be positive, got ${SHARDS}." >&2
    exit 1
fi

mkdir -p "${REPO}/logs"

metrics_job=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --job-name="met_ec_level" \
    --array="0-$((SHARDS - 1))" \
    --cpus-per-task=4 \
    --mem=24G \
    --time=02:00:00 \
    --output="${REPO}/logs/met_ec_level_%A_%a.out" \
    --export="ALL,FILTER_TASKS=ec_number,SKIP_EXISTING=0,SHARD_COUNT=${SHARDS}" \
    "${METRICS_SCRIPT}")

bootstrap_job=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --job-name="boot_ec_level" \
    --array="0-$((SHARDS - 1))" \
    --cpus-per-task=4 \
    --mem=20G \
    --time=01:00:00 \
    --dependency="afterok:${metrics_job}" \
    --output="${REPO}/logs/boot_ec_level_%A_%a.out" \
    --export="ALL,FILTER_TASKS=ec_number,SKIP_EXISTING=0,N_REPLICATES=${REPLICATES},SEED=42,SHARD_COUNT=${SHARDS}" \
    "${BOOTSTRAP_SCRIPT}")

printf 'EC metrics array:   %s (0-%s)\n' "${metrics_job}" "$((SHARDS - 1))"
printf 'EC bootstrap array: %s (afterok:%s, 0-%s)\n' \
    "${bootstrap_job}" "${metrics_job}" "$((SHARDS - 1))"
printf 'Bootstrap replicas: %s\n' "${REPLICATES}"
