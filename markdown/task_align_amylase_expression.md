---
id: task_align_amylase_expression
type: task
title: "α-Amylase expression (Align tournament)"
task_name: align_amylase_expression
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Protein Engineering Tournament (Align to Innovate)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["align_amylase_expression"]
---

# α-Amylase expression (Align tournament) — dataset / task card

> Real-case regression of expression level for engineered α-amylase variants.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `align_amylase_expression` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `expression_standardized` |
| Label units | dataset-provided value |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: ALiGN alpha-amylase real-case expression measurements.

## Source & provenance

- **Source dataset:** Protein Engineering Tournament (Align to Innovate) (benchmark provenance: `ALiGN`)

Standardized expression level (`expression_standardized`) across α-amylase variants — how well each variant is produced.

**Assay / measurement:** Experimental expression measurements on α-amylase variants in the Protein Engineering Tournament.

**Provenance & construction:** A 'real_case' α-amylase dataset from the Align Foundation tournament; random split only.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esm2_3B` | 0.1081 | `xgb_regression` |
| 2 | `profluent_e1_300m` | 0.1139 | `xgb_regression` |
| 3 | `profluent_e1_600m` | 0.116 | `xgb_regression` |
| 4 | `ankh_large` | 0.1192 | `xgb_regression` |
| 5 | `esm2_650M` | 0.1193 | `xgb_regression` |
| 6 | `esm3_open` | 0.1251 | `xgb_regression` |
| 7 | `esmc_300m` | 0.1269 | `xgb_regression` |
| 8 | `ankh3_large` | 0.1308 | `xgb_regression` |
| 9 | `esmc_600m` | 0.1313 | `xgb_regression` |
| 10 | `prost5` | 0.1316 | `xgb_regression` |
| 11 | `prot5_xl_half_uniref50` | 0.132 | `xgb_regression` |
| 12 | `ankh3_xl` | 0.1335 | `xgb_regression` |
| 13 | `ankh_base` | 0.1343 | `xgb_regression` |
| 14 | `prot5_xl_uniref50` | 0.1402 | `xgb_regression` |

## When to use

Applied expressibility ranking within one enzyme family — useful for pre-screening variants for producibility.

## Caveats

Small, single-scaffold campaign; the tournament noted expression-control comparability issues. Random split only; standardized target.

## References
- The Protein Engineering Tournament (Align to Innovate) (2025). *Results of the Protein Engineering Tournament: An Open Science Benchmark for Protein Modeling and Design*. Proteins: Structure, Function, and Bioinformatics. <https://doi.org/10.1002/prot.70008>
