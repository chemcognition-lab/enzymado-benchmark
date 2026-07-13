---
id: task_align_glucosidase_expression
type: task
title: "β-Glucosidase expression (Align tournament)"
task_name: align_glucosidase_expression
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Protein Engineering Tournament (Align to Innovate)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["align_glucosidase_expression"]
---

# β-Glucosidase expression (Align tournament) — dataset / task card

> Real-case regression of expression level for engineered β-glucosidase variants.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `align_glucosidase_expression` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `expression_standardized` |
| Label units | dataset-provided value |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: ALiGN b-glucosidase real-case expression measurements.

## Source & provenance

- **Source dataset:** Protein Engineering Tournament (Align to Innovate) (benchmark provenance: `ALiGN`)

Standardized expression level (`expression_standardized`) across β-glucosidase B variants.

**Assay / measurement:** Experimental expression measurements on β-glucosidase variants in the Protein Engineering Tournament.

**Provenance & construction:** A 'real_case' β-glucosidase dataset from the Align Foundation tournament; random split only.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esm2_650M` | 0.5083 | `xgb_regression` |
| 2 | `esm3_open` | 0.5184 | `xgb_regression` |
| 3 | `profluent_e1_300m` | 0.5225 | `xgb_regression` |
| 4 | `esmc_300m` | 0.5408 | `xgb_regression` |
| 5 | `esmc_600m` | 0.5436 | `xgb_regression` |
| 6 | `ankh3_xl` | 0.5471 | `xgb_regression` |
| 7 | `ankh_large` | 0.5473 | `attention_linear_regressor` |
| 8 | `prot5_xl_half_uniref50` | 0.548 | `attention_linear_regressor` |
| 9 | `ankh_base` | 0.5483 | `attention_linear_regressor` |
| 10 | `prost5` | 0.549 | `attention_mlp_regressor` |
| 11 | `ankh3_large` | 0.5494 | `attention_linear_regressor` |
| 12 | `profluent_e1_600m` | 0.5501 | `attention_mlp_regressor` |
| 13 | `prot5_xl_uniref50` | 0.5503 | `attention_linear_regressor` |
| 14 | `esm2_3B` | 0.5824 | `linear_regressor` |

## When to use

Applied expressibility ranking within one enzyme family — pre-screening variants for producibility.

## Caveats

Small, single-scaffold campaign; random split only. Standardized target (MAE in standardized units).

## References
- The Protein Engineering Tournament (Align to Innovate) (2025). *Results of the Protein Engineering Tournament: An Open Science Benchmark for Protein Modeling and Design*. Proteins: Structure, Function, and Bioinformatics. <https://doi.org/10.1002/prot.70008>
