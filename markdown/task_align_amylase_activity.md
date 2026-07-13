---
id: task_align_amylase_activity
type: task
title: "α-Amylase specific activity (Align tournament)"
task_name: align_amylase_activity
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Protein Engineering Tournament (Align to Innovate)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["align_amylase_activity"]
---

# α-Amylase specific activity (Align tournament) — dataset / task card

> Real-case regression of the specific activity of engineered α-amylase variants.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `align_amylase_activity` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `specific_activity_standardized` |
| Label units | dataset-provided value |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: ALiGN alpha-amylase real-case specific-activity measurements.

## Source & provenance

- **Source dataset:** Protein Engineering Tournament (Align to Innovate) (benchmark provenance: `ALiGN`)

Standardized specific activity (`specific_activity_standardized`) across α-amylase sequence variants — a single-family, applied engineering target.

**Assay / measurement:** Experimental enzyme activity measured on α-amylase variants in the Protein Engineering Tournament's supervised α-amylase event.

**Provenance & construction:** One of the 'real_case' applied datasets; α-amylase variant measurements from the Align Foundation tournament. Random split only (no similarity split).

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `profluent_e1_600m` | 0.1967 | `xgb_regression` |
| 2 | `esm2_3B` | 0.2026 | `xgb_regression` |
| 3 | `profluent_e1_300m` | 0.203 | `xgb_regression` |
| 4 | `ankh_large` | 0.205 | `xgb_regression` |
| 5 | `esm2_650M` | 0.2083 | `xgb_regression` |
| 6 | `esmc_300m` | 0.2167 | `xgb_regression` |
| 7 | `esmc_600m` | 0.2198 | `xgb_regression` |
| 8 | `ankh3_large` | 0.2208 | `xgb_regression` |
| 9 | `esm3_open` | 0.2249 | `xgb_regression` |
| 10 | `prot5_xl_uniref50` | 0.2309 | `xgb_regression` |
| 11 | `prot5_xl_half_uniref50` | 0.2337 | `xgb_regression` |
| 12 | `ankh3_xl` | 0.2348 | `xgb_regression` |
| 13 | `prost5` | 0.2406 | `xgb_regression` |
| 14 | `ankh_base` | 0.2463 | `xgb_regression` |

## When to use

A realistic low-n, single-enzyme engineering scenario — tests whether embeddings rank variants of one scaffold, the setting that matters for directed evolution.

## Caveats

Small and single-campaign; variants share a scaffold so generalization is narrow; random split only. Standardized target (MAE in standardized units).

## References
- The Protein Engineering Tournament (Align to Innovate) (2025). *Results of the Protein Engineering Tournament: An Open Science Benchmark for Protein Modeling and Design*. Proteins: Structure, Function, and Bioinformatics. <https://doi.org/10.1002/prot.70008>
