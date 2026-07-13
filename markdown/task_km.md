---
id: task_km
type: task
title: "Km (Michaelis constant)"
task_name: km
task_scope: properties
task_type: regression
level: protein-ligand
source_dataset: "CatPred"
tags: [dataset, task, properties, regression, protein-ligand]
aliases: ["km"]
---

# Km (Michaelis constant) — dataset / task card

> Regression of an enzyme's Michaelis constant (Km) for a substrate, from the protein sequence and substrate SMILES.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `km` |
| Scope | properties |
| Prediction level | protein + ligand (global) |
| Task type | regression |
| Label kind | `scalar_regression` |
| Target column | `km_standardized` |
| Label units | log10(M) |
| Primary metric | `mae` |
| Splits available | random, similarity |
| Ligand column | `smiles` |

> Benchmark-card description: Michaelis constant prediction.

## Source & provenance

- **Source dataset:** CatPred (benchmark provenance: `CatPred`)

log10(Km) in M for an enzyme–substrate pair, standardized (`km_standardized`) — the substrate concentration at half-maximal rate, i.e. apparent substrate affinity. A protein+ligand task.

**Assay / measurement:** In-vitro steady-state kinetics compiled and cleaned by CatPred, with de-duplication and sequence-identity-aware evaluation splits.

**Provenance & construction:** From the CatPred Km benchmark (~41k datapoints); substrate SMILES canonicalized and RDKit-featurized alongside the protein embedding.

## Featurization

The protein embedding is concatenated with an 8-dim RDKit descriptor vector computed from the ligand SMILES (MolWt, MolLogP, TPSA, HBD, HBA, rotatable bonds, rings, heavy atoms).

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 41174
- **Split sizes (train/val/test):** random → train 32939 / val 4117 / test 4118; similarity → train 32939 / val 4118 / test 4117
- **Sequence length:** min 5, median 389.0, mean 437.55, max 2590 (996 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 7270 clusters from 12355 sequences at 0.5 identity (5255 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `ankh_base` | 0.6028 | `xgb_regression` |
| 2 | `prost5` | 0.6113 | `xgb_regression` |
| 3 | `ankh_large` | 0.6157 | `xgb_regression` |
| 4 | `esm3_open` | 0.6169 | `xgb_regression` |
| 5 | `prot5_xl_uniref50` | 0.6184 | `xgb_regression` |
| 6 | `ankh3_large` | 0.6189 | `xgb_regression` |
| 7 | `esm2_3B` | 0.6189 | `xgb_regression` |
| 8 | `esm2_650M` | 0.6191 | `xgb_regression` |
| 9 | `prot5_xl_half_uniref50` | 0.624 | `xgb_regression` |
| 10 | `ankh3_xl` | 0.6279 | `xgb_regression` |
| 11 | `profluent_e1_300m` | 0.633 | `xgb_regression` |
| 12 | `profluent_e1_600m` | 0.6341 | `xgb_regression` |
| 13 | `esmc_600m` | 0.6376 | `xgb_regression` |
| 14 | `esmc_300m` | 0.6404 | `xgb_regression` |

### similarity split

| Rank | Model | mae | Winning head |
|---|---|---|---|
| 1 | `ankh3_large` | 0.6337 | `xgb_regression` |
| 2 | `ankh_base` | 0.6456 | `xgb_regression` |
| 3 | `prot5_xl_uniref50` | 0.648 | `xgb_regression` |
| 4 | `prost5` | 0.6495 | `xgb_regression` |
| 5 | `esm2_650M` | 0.6504 | `xgb_regression` |
| 6 | `prot5_xl_half_uniref50` | 0.6518 | `xgb_regression` |
| 7 | `ankh_large` | 0.6527 | `xgb_regression` |
| 8 | `esm3_open` | 0.6545 | `xgb_regression` |
| 9 | `ankh3_xl` | 0.6591 | `xgb_regression` |
| 10 | `esmc_600m` | 0.6598 | `xgb_regression` |
| 11 | `esmc_300m` | 0.6673 | `xgb_regression` |
| 12 | `profluent_e1_300m` | 0.6678 | `xgb_regression` |
| 13 | `esm2_3B` | 0.6688 | `xgb_regression` |
| 14 | `profluent_e1_600m` | 0.6694 | `xgb_regression` |

## When to use

Complements kcat: tests whether representations capture substrate-binding affinity. Useful when ranking models for enzyme–substrate specificity prediction.

## Caveats

Km is noisy and assay-dependent; log-standardized target (MAE in standardized units). Simple ligand featurization limits substrate resolution.

## References
- Boorla, V. S. & Maranas, C. D. (2025). *CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters*. Nature Communications 16 (2025). <https://doi.org/10.1038/s41467-025-57215-9>
