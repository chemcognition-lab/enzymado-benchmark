---
id: task_erickson_petase
type: task
title: "PETase aromatic-product release (Erickson, real-case)"
task_name: erickson_petase
task_scope: real_case
task_type: regression
level: protein-global
source_dataset: "Erickson et al. thermotolerant PET hydrolases (natural diversity)"
tags: [dataset, task, real_case, regression, protein-global]
aliases: ["erickson_petase"]
---

# PETase aromatic-product release (Erickson, real-case) — dataset / task card

> Real-case regression of PET-hydrolysis activity (sum of aromatic products) across natural PET-hydrolase enzymes.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `erickson_petase` |
| Scope | real_case |
| Prediction level | protein-level (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `sum_aromatic_products_standardized` |
| Label units | sum of aromatic products |
| Primary metric | `mae` |
| Splits available | random |

> Benchmark-card description: PETase real-case aromatic product measurements across temperatures and conditions.

## Source & provenance

- **Source dataset:** Erickson et al. thermotolerant PET hydrolases (natural diversity) (benchmark provenance: `Erickson`)

Standardized sum of aromatic products (`sum_aromatic_products_standardized`) released from PET film — total BHET + MHET + TPA quantified by UHPLC — as a measure of depolymerization activity.

**Assay / measurement:** PET-film hydrolysis assays across a panel of natural thermotolerant PET hydrolases over ranges of temperature (30–70 °C) and pH (4.5–9.0), with products quantified by UHPLC.

**Provenance & construction:** A 'real_case' dataset derived from Erickson et al. (2022), which mined and assayed PET hydrolases from natural diversity. Sequence + measured product release; random split only.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `prot5_xl_uniref50` | 1.6181 | `linear_regressor` |
| 2 | `ankh3_xl` | 1.717 | `xgb_regression` |
| 3 | `ankh_base` | 1.7795 | `xgb_regression` |
| 4 | `esmc_600m` | 1.7931 | `xgb_regression` |
| 5 | `ankh_large` | 1.9093 | `mlp_regressor` |
| 6 | `esm2_3B` | 1.9143 | `xgb_regression` |
| 7 | `ankh3_large` | 1.9512 | `attention_mlp_regressor` |
| 8 | `esmc_300m` | 1.9524 | `mlp_regressor` |
| 9 | `esm3_open` | 1.9794 | `linear_regressor` |
| 10 | `prot5_xl_half_uniref50` | 1.994 | `mlp_regressor` |
| 11 | `prost5` | 2.0003 | `mlp_regressor` |
| 12 | `esm2_650M` | 2.1007 | `mlp_regressor` |
| 13 | `profluent_e1_600m` | 2.1145 | `attention_linear_regressor` |
| 14 | `profluent_e1_300m` | 2.131 | `attention_linear_regressor` |

## When to use

A realistic biocatalysis screening target (plastic degradation) — tests whether embeddings rank diverse natural enzymes by a practical activity readout.

## Caveats

Measurements depend strongly on assay conditions (temperature/pH) that may be collapsed into the target; small, specialized panel; random split only. Standardized target (MAE in standardized units).

## References
- Erickson, E. et al. (2022). *Sourcing thermotolerant poly(ethylene terephthalate) hydrolase scaffolds from natural diversity*. Nature Communications 13 (2022). <https://doi.org/10.1038/s41467-022-35237-x>
