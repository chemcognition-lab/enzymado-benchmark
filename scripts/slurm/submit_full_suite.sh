#!/usr/bin/env bash
# Submit the full 1960-run benchmark suite.
#
# Sizing comes from sacct accounting of the 2026-07/08 runs rather than from
# round numbers: every --mem/--time below is the measured peak plus margin.
# Two corrections relative to the earlier hand-written plan:
#
#   * binding_site / active_sites no_attention had 6h limits and timed out
#     (site_binding_rand_na: 14/14 TIMEOUT). Bundling all three model families
#     costs ~22h for binding_site, so it is now split by --model_prefix.
#   * binding_site attention hit the 24h wall bundled, so linear/mlp are
#     submitted separately (~12h each).
#
# Heavy jobs go first so they claim GPUs while the cheap ones backfill.
set -euo pipefail

ACCOUNT="${AIP_ACCOUNT:-aip-bensl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The residue XGBoost binned cache is hundreds of GiB (binding_site alone wrote
# 279 GiB of a 301 GiB /tmp before dying with an incomplete FileStream.Write).
# Point it at scratch, which has terabytes; the cache is rebuilt every run, so
# nothing of record lives there.
XGB_CACHE="/scratch/${USER}/xgb_residue_cache"
mkdir -p "${XGB_CACHE}"
NA="${REPO}/scripts/slurm/run_training_cluster.slurm"
AT="${REPO}/scripts/slurm/run_training_attention_cluster.slurm"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOBLOG="${REPO}/logs/submitted_${STAMP}.tsv"
mkdir -p "${REPO}/logs"
printf 'jobid\tname\tscript\tmem\ttime\texports\n' > "${JOBLOG}"

submit() {  # submit <name> <script> <mem> <time> <extra-exports>
    local name="$1" script="$2" mem="$3" tl="$4" exports="$5"
    local jid
    jid=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --job-name="${name}" \
        --array=0-13 \
        --mem="${mem}" \
        --time="${tl}" \
        --output="${REPO}/logs/${name}_%A_%a.out" \
        --export="ALL,OVERWRITE_RESULTS=1,SEED=42,XGB_RESIDUE_CACHE_DIR=${XGB_CACHE},${exports}" \
        "${script}")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${jid}" "${name}" "$(basename "${script}")" "${mem}" "${tl}" "${exports}" >> "${JOBLOG}"
    echo "  ${jid}  ${name}  mem=${mem} time=${tl}"
}

echo "== 1. PESADAS =="
# ec_number no_attention: 2.2h measured + ~3.4h of new per-epoch EC AUPRC
# validation (41s/epoch at 135M label slots). RAM was pinned at 256G, so the
# extra validation arrays get their own headroom.
for split in random similarity; do
    submit "h_ec_${split}_na"  "${NA}" 288G 10:00:00 "TASK_NAMES=ec_number,SPLIT_VARIANT=${split}"
done
# active_sites no_attention: families bundled ~8.5h (lin 1.6 + mlp 2.7 + xgb 4.2).
for split in random similarity; do
    submit "h_as_${split}_na"  "${NA}" 256G 12:00:00 "TASK_NAMES=active_sites,SPLIT_VARIANT=${split}"
done
# binding_site no_attention: split by family; mlp alone measured 16.0h.
for split in random similarity; do
    submit "h_bs_${split}_na_lin" "${NA}" 256G 06:00:00 "TASK_NAMES=binding_site,SPLIT_VARIANT=${split},MODEL_PREFIX=linear_"
    submit "h_bs_${split}_na_xgb" "${NA}" 256G 12:00:00 "TASK_NAMES=binding_site,SPLIT_VARIANT=${split},MODEL_PREFIX=xgb_"
    submit "h_bs_${split}_na_mlp" "${NA}" 256G 24:00:00 "TASK_NAMES=binding_site,SPLIT_VARIANT=${split},MODEL_PREFIX=mlp_"
done
# Attention heavies: split linear/mlp so nothing sits at the 24h wall again.
for split in random similarity; do
    for fam in linear mlp; do
        submit "h_ec_${split}_at_${fam}" "${AT}" 160G 14:00:00 \
            "TASK_NAMES=ec_number,SPLIT_VARIANT=${split},MODEL_PREFIX=attention_${fam}_"
        submit "h_bs_${split}_at_${fam}" "${AT}" 160G 16:00:00 \
            "TASK_NAMES=binding_site,SPLIT_VARIANT=${split},MODEL_PREFIX=attention_${fam}_"
        submit "h_as_${split}_at_${fam}" "${AT}" 160G 08:00:00 \
            "TASK_NAMES=active_sites,SPLIT_VARIANT=${split},MODEL_PREFIX=attention_${fam}_"
    done
done

echo "== 2. MEDIAS =="
# Measured: no_attention <=34G / <=0.09h, attention <=35G / <=2.95h.
for task in kcat km binding ph_opt solubility; do
    for split in random similarity; do
        submit "m_${task}_${split}_na" "${NA}" 48G 01:00:00 "TASK_NAMES=${task},SPLIT_VARIANT=${split}"
        submit "m_${task}_${split}_at" "${AT}" 48G 04:00:00 "TASK_NAMES=${task},SPLIT_VARIANT=${split}"
    done
done

echo "== 3. LIGERAS + REAL_CASE =="
# Measured: no_attention <=7G / 0.02h, attention <=11G / 0.42h.
for task in dtm ddg; do
    for split in random similarity; do
        submit "l_${task}_${split}_na" "${NA}" 24G 01:00:00 "TASK_NAMES=${task},SPLIT_VARIANT=${split}"
        submit "l_${task}_${split}_at" "${AT}" 32G 02:00:00 "TASK_NAMES=${task},SPLIT_VARIANT=${split}"
    done
done
# real_case: 8 tasks bundled, random only. Measured 31G/0.12h and 28G/1.16h.
submit "rc_random_na" "${NA}" 48G 01:00:00 "TASK_SCOPES=real_case,SPLIT_VARIANT=random"
submit "rc_random_at" "${AT}" 48G 03:00:00 "TASK_SCOPES=real_case,SPLIT_VARIANT=random"

echo
echo "Registro de envio: ${JOBLOG}"
wc -l < "${JOBLOG}" | xargs -I{} echo "Jobs enviados (incl. cabecera): {}"
