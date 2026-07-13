---
id: task_kcat
type: task
title: "kcat (turnover number)"
task_name: kcat
task_scope: properties
task_type: regression
level: protein-ligand
source_dataset: "CatPred"
tags: [dataset, task, properties, regression, protein-ligand]
aliases: ["kcat"]
---

# kcat (turnover number) — dataset / task card

> Regression of an enzyme's turnover number (kcat) for a given substrate, from the protein sequence and the substrate SMILES.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `kcat` |
| Scope | properties |
| Prediction level | protein + ligand (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `kcat_standardized` |
| Label units | log10(s^-1) |
| Primary metric | `mae` |
| Splits available | random, similarity |
| Ligand column | `smiles` |

> Benchmark-card description: Turnover number prediction.

## Source & provenance

- **Source dataset:** CatPred (benchmark provenance: `CatPred`)

log10(kcat) in s⁻¹ for an enzyme–substrate pair, standardized (`kcat_standardized`). This is a protein+ligand task: the model sees both the sequence and the reaction's substrate.

**Assay / measurement:** In-vitro steady-state kinetics compiled and cleaned by CatPred (drawing on BRENDA/SABIO-RK-style sources), with careful de-duplication and out-of-distribution splits.

**Provenance & construction:** From the CatPred kcat benchmark (~23k datapoints); substrate SMILES are canonicalized and featurized with RDKit descriptors alongside the protein embedding.

## Featurization

The protein embedding is concatenated with an 8-dim RDKit descriptor vector computed from the ligand SMILES (MolWt, MolLogP, TPSA, HBD, HBA, rotatable bonds, rings, heavy atoms).

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 23151
- **Split sizes (train/val/test):** random → train 18521 / val 2315 / test 2315; similarity → train 18521 / val 2315 / test 2315
- **Sequence length:** min 7, median 377.0, mean 430.46, max 2515 (538 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 4693 clusters from 7177 sequences at 0.5 identity (3570 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `ankh_large` | 0.6482 | `xgb_regression` |
| 2 | `prost5` | 0.6489 | `xgb_regression` |
| 3 | `prot5_xl_half_uniref50` | 0.649 | `xgb_regression` |
| 4 | `prot5_xl_uniref50` | 0.6492 | `xgb_regression` |
| 5 | `esm3_open` | 0.652 | `xgb_regression` |
| 6 | `ankh3_large` | 0.6541 | `xgb_regression` |
| 7 | `esm2_3B` | 0.6545 | `xgb_regression` |
| 8 | `ankh_base` | 0.6548 | `xgb_regression` |
| 9 | `ankh3_xl` | 0.6551 | `xgb_regression` |
| 10 | `profluent_e1_600m` | 0.6571 | `xgb_regression` |
| 11 | `esmc_600m` | 0.6585 | `xgb_regression` |
| 12 | `esmc_300m` | 0.6713 | `xgb_regression` |
| 13 | `profluent_e1_300m` | 0.6743 | `xgb_regression` |
| 14 | `esm2_650M` | 0.676 | `xgb_regression` |

### similarity split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `prot5_xl_half_uniref50` | 0.5819 | `attention_linear_regressor` |
| 2 | `prot5_xl_uniref50` | 0.5906 | `attention_linear_regressor` |
| 3 | `esmc_600m` | 0.5913 | `attention_linear_regressor` |
| 4 | `prost5` | 0.5927 | `attention_linear_regressor` |
| 5 | `esmc_300m` | 0.5951 | `xgb_regression` |
| 6 | `ankh3_xl` | 0.6036 | `xgb_regression` |
| 7 | `profluent_e1_600m` | 0.6086 | `linear_regressor` |
| 8 | `esm2_3B` | 0.609 | `xgb_regression` |
| 9 | `profluent_e1_300m` | 0.6104 | `attention_linear_regressor` |
| 10 | `esm2_650M` | 0.613 | `xgb_regression` |
| 11 | `esm3_open` | 0.617 | `mlp_regressor` |
| 12 | `ankh3_large` | 0.6311 | `linear_regressor` |
| 13 | `ankh_large` | 0.6355 | `xgb_regression` |
| 14 | `ankh_base` | 0.6431 | `attention_mlp_regressor` |

## When to use

A core enzyme-kinetics regression testing whether embeddings (plus ligand descriptors) capture catalytic efficiency — central to enzyme selection and engineering.

## Caveats

kcat spans orders of magnitude and is measurement-noisy; log-standardized targets mean MAE is in standardized units. The 8-dim RDKit ligand featurization is deliberately simple.

## References
- Boorla, V. S. & Maranas, C. D. (2025). *CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters*. Nature Communications 16 (2025). <https://doi.org/10.1038/s41467-025-57215-9>
