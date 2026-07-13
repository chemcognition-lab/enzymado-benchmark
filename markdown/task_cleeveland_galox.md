---
id: task_cleeveland_galox
type: task
title: "Galactose oxidase activity (GalOx, real-case)"
task_name: cleeveland_galox
task_scope: real_case
task_type: regression
level: protein-ligand
source_dataset: "Galactose oxidase engineering campaign (GalOx)"
tags: [dataset, task, real_case, regression, protein-ligand]
aliases: ["cleeveland_galox"]
---

# Galactose oxidase activity (GalOx, real-case) — dataset / task card

> Real-case protein+ligand regression of galactose oxidase (GOase) variant activity toward specific alcohol substrates.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `cleeveland_galox` |
| Scope | real_case |
| Prediction level | protein + ligand (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `specific_activity_standardized` |
| Label units | specific activity (dataset-provided units) |
| Primary metric | `mae` |
| Splits available | random |
| Ligand column | `smiles` |

> Benchmark-card description: Galactose oxidase real-case activity measurements across protein-substrate pairs.

## Source & provenance

- **Source dataset:** Galactose oxidase engineering campaign (GalOx) (benchmark provenance: `Cleeveland`)

Standardized specific activity (`specific_activity_standardized`) for engineered GOase variants paired with a substrate (SMILES) — i.e. variant × substrate catalytic activity.

**Assay / measurement:** Experimental activity of engineered galactose oxidase variants on benzylic/secondary-alcohol substrates, from a machine-learning-guided directed-evolution campaign.

**Provenance & construction:** A 'real_case' dataset combining GOase protein variants with substrate SMILES and measured activity; random split only. Exact source file is a specific engineering campaign — verify provenance before external citation.

## Featurization

The protein embedding is concatenated with an 8-dim RDKit descriptor vector computed from the ligand SMILES (MolWt, MolLogP, TPSA, HBD, HBA, rotatable bonds, rings, heavy atoms).

## Dataset statistics

_No `dataset_card.json` stats available for this task (real_case tasks have none yet)._

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `esm2_650M` | 0.0562 | `xgb_regression` |
| 2 | `esm3_open` | 0.0567 | `xgb_regression` |
| 3 | `ankh3_large` | 0.0576 | `xgb_regression` |
| 4 | `esm2_3B` | 0.0595 | `xgb_regression` |
| 5 | `esmc_600m` | 0.0612 | `xgb_regression` |
| 6 | `ankh_base` | 0.0624 | `xgb_regression` |
| 7 | `esmc_300m` | 0.065 | `xgb_regression` |
| 8 | `prot5_xl_half_uniref50` | 0.0684 | `attention_mlp_regressor` |
| 9 | `prost5` | 0.1084 | `attention_mlp_regressor` |
| 10 | `prot5_xl_uniref50` | 0.1171 | `attention_mlp_regressor` |
| 11 | `profluent_e1_600m` | 0.1178 | `xgb_regression` |
| 12 | `ankh3_xl` | 0.1321 | `linear_regressor` |
| 13 | `profluent_e1_300m` | 0.1448 | `attention_mlp_regressor` |
| 14 | `ankh_large` | 0.3387 | `attention_mlp_regressor` |

## When to use

A realistic enzyme+substrate engineering scenario with a genuine ligand axis — tests joint reasoning over variant sequence and substrate structure.

## Caveats

Small, single-enzyme campaign; random split only; substrate uses the simple 8-dim RDKit featurization. Provenance/attribution should be confirmed with the benchmark authors.

## References
- e.g. Galactose oxidase machine-learning engineering studies (2024). *A Machine Learning-Guided Approach to Navigate the Substrate Activity Scope of Galactose Oxidase*. ACS Catalysis 14. <https://doi.org/10.1021/acscatal.4c04660>
