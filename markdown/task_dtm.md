---
id: task_dtm
type: task
title: "ΔTm (melting-temperature change on mutation)"
task_name: dtm
task_scope: properties
task_type: regression
level: mutation-delta
source_dataset: "FireProtDB"
tags: [dataset, task, properties, regression, mutation-delta]
aliases: ["dtm"]
---

# ΔTm (melting-temperature change on mutation) — dataset / task card

> Regression of the change in thermal melting temperature (ΔTm) caused by a single-point mutation.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `dtm` |
| Scope | properties |
| Prediction level | mutation effect (Δ vs wild-type) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `dtm_standardized` |
| Label units | delta Tm (degC) |
| Primary metric | `mae` |
| Splits available | random, similarity |
| Wild-type sequence column | `wildtype_sequence` |

> Benchmark-card description: Melting temperature shift prediction for point mutations.

## Source & provenance

- **Source dataset:** FireProtDB (benchmark provenance: `FireProtDB`)

ΔTm (°C) of a point mutant relative to wild type, standardized (`dtm_standardized`). This is a mutation-delta task: the feature is the mutant-minus-wild-type embedding.

**Assay / measurement:** Experimentally measured thermostability (Tm) of single-point mutants, manually curated in FireProtDB from literature and lab data.

**Provenance & construction:** FireProtDB split into ΔTm and ΔΔG subsets by non-null field; mutant sequences are built by applying the substitution to the wild-type sequence.

## Featurization

The feature is the mutant-minus-wild-type embedding delta (mutant and wild-type sequences must be equal length).

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 3338
- **Split sizes (train/val/test):** random → train 2670 / val 334 / test 334; similarity → train 2326 / val 679 / test 333
- **Sequence length:** min 96, median 231.0, mean 270.74, max 2527 (3 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 82 clusters from 91 sequences at 0.5 identity (74 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esmc_600m` | 0.3802 | `xgb_regression` |
| 2 | `esm3_open` | 0.3865 | `xgb_regression` |
| 3 | `ankh3_xl` | 0.3909 | `mlp_regressor` |
| 4 | `prot5_xl_half_uniref50` | 0.3985 | `xgb_regression` |
| 5 | `prot5_xl_uniref50` | 0.4029 | `attention_mlp_regressor` |
| 6 | `profluent_e1_300m` | 0.4133 | `xgb_regression` |
| 7 | `esmc_300m` | 0.4237 | `xgb_regression` |
| 8 | `ankh_large` | 0.4263 | `xgb_regression` |
| 9 | `profluent_e1_600m` | 0.4357 | `xgb_regression` |
| 10 | `esm2_3B` | 0.436 | `mlp_regressor` |
| 11 | `prost5` | 0.4374 | `attention_mlp_regressor` |
| 12 | `esm2_650M` | 0.4376 | `xgb_regression` |
| 13 | `ankh_base` | 0.452 | `xgb_regression` |
| 14 | `ankh3_large` | 0.4608 | `mlp_regressor` |

### similarity split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esmc_600m` | 0.5825 | `mlp_regressor` |
| 2 | `ankh_base` | 0.5956 | `mlp_regressor` |
| 3 | `prot5_xl_uniref50` | 0.615 | `linear_regressor` |
| 4 | `esmc_300m` | 0.6164 | `xgb_regression` |
| 5 | `ankh3_xl` | 0.6164 | `mlp_regressor` |
| 6 | `prot5_xl_half_uniref50` | 0.6248 | `xgb_regression` |
| 7 | `esm3_open` | 0.6334 | `xgb_regression` |
| 8 | `ankh_large` | 0.6364 | `xgb_regression` |
| 9 | `ankh3_large` | 0.6464 | `mlp_regressor` |
| 10 | `prost5` | 0.6474 | `mlp_regressor` |
| 11 | `esm2_3B` | 0.652 | `attention_mlp_regressor` |
| 12 | `profluent_e1_600m` | 0.6581 | `attention_linear_regressor` |
| 13 | `profluent_e1_300m` | 0.6801 | `attention_linear_regressor` |
| 14 | `esm2_650M` | 0.7045 | `xgb_regression` |

## When to use

Protein-engineering task for stabilizing mutations — tests whether embedding deltas capture the effect of point mutations on thermal stability.

## Caveats

Small, imbalanced, and family-biased; single-point only. Mutant−WT deltas require equal-length sequences. Standardized target (MAE in standardized units).

## References
- Stourac, J. et al. (2021). *FireProtDB: database of manually curated protein stability data*. Nucleic Acids Research 49(D1):D319–D324. <https://doi.org/10.1093/nar/gkaa981>
