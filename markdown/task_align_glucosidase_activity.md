---
id: task_align_glucosidase_activity
type: task
title: "β-Glucosidase activity (Align tournament)"
task_name: align_glucosidase_activity
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Protein Engineering Tournament (Align to Innovate)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["align_glucosidase_activity"]
---

# β-Glucosidase activity (Align tournament) — dataset / task card

> Real-case regression of catalytic activity for engineered β-glucosidase variants.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `align_glucosidase_activity` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `activity_standardized` |
| Label units | dataset-provided value |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: ALiGN b-glucosidase real-case activity measurements.

## Source & provenance

- **Source dataset:** Protein Engineering Tournament (Align to Innovate) (benchmark provenance: `ALiGN`)

Standardized activity (`activity_standardized`) across β-glucosidase B variants.

**Assay / measurement:** Experimental activity measurements on β-glucosidase variants in the Protein Engineering Tournament's supervised event.

**Provenance & construction:** A 'real_case' β-glucosidase dataset from the Align Foundation tournament; random split only.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `ankh_base` | 0.2066 | `attention_linear_regressor` |
| 2 | `esm3_open` | 0.2067 | `attention_mlp_regressor` |
| 3 | `profluent_e1_600m` | 0.2069 | `attention_mlp_regressor` |
| 4 | `esmc_300m` | 0.2078 | `attention_linear_regressor` |
| 5 | `prost5` | 0.2087 | `attention_linear_regressor` |
| 6 | `profluent_e1_300m` | 0.209 | `attention_linear_regressor` |
| 7 | `esm2_650M` | 0.2096 | `attention_mlp_regressor` |
| 8 | `prot5_xl_half_uniref50` | 0.2109 | `attention_linear_regressor` |
| 9 | `ankh_large` | 0.211 | `attention_mlp_regressor` |
| 10 | `ankh3_large` | 0.2118 | `attention_mlp_regressor` |
| 11 | `esmc_600m` | 0.212 | `attention_mlp_regressor` |
| 12 | `prot5_xl_uniref50` | 0.213 | `attention_linear_regressor` |
| 13 | `ankh3_xl` | 90.7729 | `xgb_regression` |
| 14 | `esm2_3B` | 102.9854 | `linear_regressor` |

## When to use

Applied activity ranking within one enzyme family — realistic directed-evolution setting for β-glucosidase.

## Caveats

Small, single-scaffold campaign; random split only. Standardized target (MAE in standardized units).

## References
- The Protein Engineering Tournament (Align to Innovate) (2025). *Results of the Protein Engineering Tournament: An Open Science Benchmark for Protein Modeling and Design*. Proteins: Structure, Function, and Bioinformatics. <https://doi.org/10.1002/prot.70008>
