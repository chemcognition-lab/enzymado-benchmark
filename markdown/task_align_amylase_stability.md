---
id: task_align_amylase_stability
type: task
title: "α-Amylase thermostability (Align tournament)"
task_name: align_amylase_stability
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Protein Engineering Tournament (Align to Innovate)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["align_amylase_stability"]
---

# α-Amylase thermostability (Align tournament) — dataset / task card

> Real-case regression of thermostability for engineered α-amylase variants.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `align_amylase_stability` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `thermostability_standardized` |
| Label units | dataset-provided value |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: ALiGN alpha-amylase real-case thermostability measurements.

## Source & provenance

- **Source dataset:** Protein Engineering Tournament (Align to Innovate) (benchmark provenance: `ALiGN`)

Standardized thermostability (`thermostability_standardized`) across α-amylase variants.

**Assay / measurement:** Experimental thermostability measurements on α-amylase variants in the Protein Engineering Tournament.

**Provenance & construction:** A 'real_case' α-amylase dataset from the Align Foundation tournament; random split only.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esmc_300m` | 0.117 | `attention_linear_regressor` |
| 2 | `ankh_large` | 0.149 | `attention_linear_regressor` |
| 3 | `ankh_base` | 0.1604 | `attention_mlp_regressor` |
| 4 | `profluent_e1_300m` | 0.2107 | `attention_linear_regressor` |
| 5 | `prot5_xl_uniref50` | 0.2312 | `attention_linear_regressor` |
| 6 | `esm2_650M` | 0.2377 | `attention_linear_regressor` |
| 7 | `profluent_e1_600m` | 0.2418 | `xgb_regression` |
| 8 | `ankh3_large` | 0.2458 | `attention_linear_regressor` |
| 9 | `esm2_3B` | 0.2538 | `xgb_regression` |
| 10 | `esmc_600m` | 0.2581 | `attention_linear_regressor` |
| 11 | `prost5` | 0.2648 | `attention_linear_regressor` |
| 12 | `prot5_xl_half_uniref50` | 0.272 | `attention_linear_regressor` |
| 13 | `esm3_open` | 0.2915 | `xgb_regression` |
| 14 | `ankh3_xl` | 0.3227 | `xgb_regression` |

## When to use

Applied stability ranking within one enzyme family — the practical target when engineering more robust α-amylases.

## Caveats

Small, single-scaffold; random split only (no similarity split). Standardized target (MAE in standardized units).

## References
- The Protein Engineering Tournament (Align to Innovate) (2025). *Results of the Protein Engineering Tournament: An Open Science Benchmark for Protein Modeling and Design*. Proteins: Structure, Function, and Bioinformatics. <https://doi.org/10.1002/prot.70008>
