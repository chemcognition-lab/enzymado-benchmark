---
id: task_align_glucosidase_melting
type: task
title: "β-Glucosidase melting point (Align tournament)"
task_name: align_glucosidase_melting
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Protein Engineering Tournament (Align to Innovate)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["align_glucosidase_melting"]
---

# β-Glucosidase melting point (Align tournament) — dataset / task card

> Real-case regression of melting temperature for engineered β-glucosidase variants.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `align_glucosidase_melting` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `melting_point_standardized` |
| Label units | dataset-provided value |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: ALiGN b-glucosidase real-case melting-point measurements.

## Source & provenance

- **Source dataset:** Protein Engineering Tournament (Align to Innovate) (benchmark provenance: `ALiGN`)

Standardized melting point (`melting_point_standardized`), a thermostability proxy, across β-glucosidase B variants.

**Assay / measurement:** Experimental melting-temperature measurements on β-glucosidase variants in the Protein Engineering Tournament.

**Provenance & construction:** A 'real_case' β-glucosidase dataset from the Align Foundation tournament; random split only.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `prot5_xl_uniref50` | 0.6244 | `xgb_regression` |
| 2 | `ankh_base` | 0.6616 | `xgb_regression` |
| 3 | `ankh_large` | 0.6682 | `attention_linear_regressor` |
| 4 | `esmc_300m` | 0.6745 | `attention_linear_regressor` |
| 5 | `profluent_e1_600m` | 0.6763 | `attention_mlp_regressor` |
| 6 | `profluent_e1_300m` | 0.6765 | `attention_mlp_regressor` |
| 7 | `esmc_600m` | 0.6769 | `attention_mlp_regressor` |
| 8 | `prot5_xl_half_uniref50` | 0.6779 | `attention_mlp_regressor` |
| 9 | `ankh3_large` | 0.6796 | `attention_mlp_regressor` |
| 10 | `esm2_650M` | 0.6813 | `attention_mlp_regressor` |
| 11 | `esm2_3B` | 0.7016 | `mlp_regressor` |
| 12 | `prost5` | 0.7219 | `attention_mlp_regressor` |
| 13 | `ankh3_xl` | 0.7551 | `xgb_regression` |
| 14 | `esm3_open` | 0.7875 | `linear_regressor` |

## When to use

Applied stability ranking within one enzyme family — the practical target when engineering more thermostable β-glucosidases.

## Caveats

Small, single-scaffold campaign; random split only. Standardized target (MAE in standardized units).

## References
- The Protein Engineering Tournament (Align to Innovate) (2025). *Results of the Protein Engineering Tournament: An Open Science Benchmark for Protein Modeling and Design*. Proteins: Structure, Function, and Bioinformatics. <https://doi.org/10.1002/prot.70008>
