---
id: task_ph_opt
type: task
title: "pH optimum (optimal pH of activity)"
task_name: ph_opt
task_scope: properties
task_type: regression
level: protein-global
source_dataset: "BRENDA"
tags: [dataset, task, properties, regression, protein-global]
aliases: ["ph_opt", "phopt"]
---

# pH optimum (optimal pH of activity) — dataset / task card

> Regression of an enzyme's optimal pH from sequence, aggregated per protein from BRENDA measurements.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `ph_opt` |
| Scope | properties |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `phopt_median_standardized` |
| Label units | pH |
| Primary metric | `mae` |
| Splits available | random, similarity |

> Benchmark-card description: Optimal pH prediction.

## Source & provenance

- **Source dataset:** BRENDA (benchmark provenance: `BRENDA`)

The pH at which an enzyme shows maximal activity. Multiple BRENDA measurements per protein are aggregated to a median (pH ranges are averaged), then z-standardized (`phopt_median_standardized`).

**Assay / measurement:** Literature-curated enzyme activity-vs-pH optima collected in BRENDA, the comprehensive enzyme-function database.

**Provenance & construction:** Parsed from the BRENDA JSON release (`ph_optimum` field), aggregated to a per-protein median with measurement counts recorded.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 11464
- **Split sizes (train/val/test):** random → train 9171 / val 1146 / test 1147; similarity → train 9171 / val 1147 / test 1146
- **Sequence length:** min 5, median 408.0, mean 469.81, max 7096 (411 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 6885 clusters from 11110 sequences at 0.5 identity (5132 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `ankh_base` | 0.5139 | `xgb_regression` |
| 2 | `esm3_open` | 0.5159 | `xgb_regression` |
| 3 | `prot5_xl_half_uniref50` | 0.5182 | `attention_mlp_regressor` |
| 4 | `ankh_large` | 0.5274 | `xgb_regression` |
| 5 | `esm2_3B` | 0.5281 | `xgb_regression` |
| 6 | `ankh3_xl` | 0.5302 | `xgb_regression` |
| 7 | `esm2_650M` | 0.5309 | `xgb_regression` |
| 8 | `prot5_xl_uniref50` | 0.533 | `xgb_regression` |
| 9 | `prost5` | 0.5331 | `xgb_regression` |
| 10 | `ankh3_large` | 0.5418 | `xgb_regression` |
| 11 | `esmc_300m` | 0.5481 | `xgb_regression` |
| 12 | `esmc_600m` | 0.5532 | `xgb_regression` |
| 13 | `profluent_e1_300m` | 0.559 | `xgb_regression` |
| 14 | `profluent_e1_600m` | 0.5602 | `xgb_regression` |

### similarity split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `profluent_e1_300m` | 0.6748 | `attention_mlp_regressor` |
| 2 | `esmc_600m` | 0.6815 | `attention_linear_regressor` |
| 3 | `esm2_3B` | 0.6826 | `mlp_regressor` |
| 4 | `prot5_xl_half_uniref50` | 0.6912 | `xgb_regression` |
| 5 | `esmc_300m` | 0.6933 | `attention_mlp_regressor` |
| 6 | `ankh_large` | 0.6986 | `mlp_regressor` |
| 7 | `prot5_xl_uniref50` | 0.699 | `attention_mlp_regressor` |
| 8 | `prost5` | 0.7006 | `attention_mlp_regressor` |
| 9 | `profluent_e1_600m` | 0.7094 | `linear_regressor` |
| 10 | `esm2_650M` | 0.7104 | `linear_regressor` |
| 11 | `esm3_open` | 0.7159 | `xgb_regression` |
| 12 | `ankh_base` | 0.7261 | `xgb_regression` |
| 13 | `ankh3_xl` | 0.748 | `linear_regressor` |
| 14 | `ankh3_large` | 0.7555 | `xgb_regression` |

## When to use

A protein-level regression testing whether embeddings encode the physicochemical environment an enzyme is adapted to — relevant to enzyme selection for a target pH.

## Caveats

pH optima are condition- and assay-dependent and sparsely measured; aggregating to a median discards spread. Values are standardized, so MAE is in standardized units.

## References
- Chang, A. et al. (2021). *BRENDA, the ELIXIR core data resource in 2021: new developments and updates*. Nucleic Acids Research 49(D1):D498–D508. <https://doi.org/10.1093/nar/gkaa1025>
