---
id: task_ddg
type: task
title: "ΔΔG (folding free-energy change on mutation)"
task_name: ddg
task_scope: properties
task_type: regression
level: mutation-delta
source_dataset: "FireProtDB"
tags: [dataset, task, properties, regression, mutation-delta]
aliases: ["ddg"]
---

# ΔΔG (folding free-energy change on mutation) — dataset / task card

> Regression of the change in folding free energy (ΔΔG) caused by a single-point mutation.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `ddg` |
| Scope | properties |
| Prediction level | mutation effect (Δ vs wild-type) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `ddg_standardized` |
| Label units | delta delta G (kcal/mol) |
| Primary metric | `mae` |
| Splits available | random, similarity |
| Wild-type sequence column | `wildtype_sequence` |

> Benchmark-card description: Gibbs free energy change prediction for point mutations.

## Source & provenance

- **Source dataset:** FireProtDB (benchmark provenance: `FireProtDB`)

ΔΔG (kcal/mol) of a point mutant relative to wild type, standardized (`ddg_standardized`). A mutation-delta task using the mutant-minus-wild-type embedding.

**Assay / measurement:** Experimentally derived stability changes (ΔΔG) for single-point mutants, manually curated in FireProtDB.

**Provenance & construction:** The ΔΔG subset of FireProtDB (data dir `ddg_mut`); mutant sequences built by applying substitutions to the wild type.

## Featurization

The feature is the mutant-minus-wild-type embedding delta (mutant and wild-type sequences must be equal length).

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 3774
- **Split sizes (train/val/test):** random → train 3019 / val 378 / test 377; similarity → train 2741 / val 656 / test 377
- **Sequence length:** min 96, median 164.0, mean 351.55, max 34350 (38 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 62 clusters from 69 sequences at 0.5 identity (57 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esm3_open` | 0.6023 | `mlp_regressor` |
| 2 | `ankh_large` | 0.6101 | `mlp_regressor` |
| 3 | `esmc_600m` | 0.6179 | `mlp_regressor` |
| 4 | `esm2_650M` | 0.6287 | `mlp_regressor` |
| 5 | `prot5_xl_half_uniref50` | 0.6355 | `mlp_regressor` |
| 6 | `ankh3_xl` | 0.6461 | `mlp_regressor` |
| 7 | `prot5_xl_uniref50` | 0.6496 | `mlp_regressor` |
| 8 | `esm2_3B` | 0.6614 | `mlp_regressor` |
| 9 | `profluent_e1_300m` | 0.668 | `mlp_regressor` |
| 10 | `prost5` | 0.6688 | `mlp_regressor` |
| 11 | `esmc_300m` | 0.6832 | `mlp_regressor` |
| 12 | `ankh_base` | 0.6889 | `mlp_regressor` |
| 13 | `profluent_e1_600m` | 0.6941 | `attention_mlp_regressor` |
| 14 | `ankh3_large` | 0.7378 | `xgb_regression` |

### similarity split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `ankh_large` | 0.4491 | `xgb_regression` |
| 2 | `esmc_600m` | 0.4579 | `mlp_regressor` |
| 3 | `ankh_base` | 0.478 | `mlp_regressor` |
| 4 | `esm3_open` | 0.4986 | `xgb_regression` |
| 5 | `prost5` | 0.5069 | `xgb_regression` |
| 6 | `esmc_300m` | 0.5075 | `attention_mlp_regressor` |
| 7 | `profluent_e1_300m` | 0.5127 | `attention_linear_regressor` |
| 8 | `prot5_xl_uniref50` | 0.5214 | `mlp_regressor` |
| 9 | `ankh3_xl` | 0.5236 | `mlp_regressor` |
| 10 | `esm2_650M` | 0.5299 | `mlp_regressor` |
| 11 | `prot5_xl_half_uniref50` | 0.5385 | `xgb_regression` |
| 12 | `esm2_3B` | 0.5479 | `attention_mlp_regressor` |
| 13 | `ankh3_large` | 0.5509 | `xgb_regression` |
| 14 | `profluent_e1_600m` | 0.5574 | `attention_mlp_regressor` |

## When to use

The canonical stability-prediction regression for engineering — tests sensitivity of embedding deltas to destabilizing/stabilizing point mutations.

## Caveats

Sign convention and measurement noise matter; small and family-biased; single-point only. Standardized target (MAE in standardized units).

## References
- Stourac, J. et al. (2021). *FireProtDB: database of manually curated protein stability data*. Nucleic Acids Research 49(D1):D319–D324. <https://doi.org/10.1093/nar/gkaa981>
