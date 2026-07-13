---
id: task_solubility
type: task
title: "Solubility (protein solubility classification)"
task_name: solubility
task_scope: properties
task_type: classification
level: protein-global
source_dataset: "ProteinEA solubility (DeepSol lineage)"
tags: [dataset, task, properties, classification, protein-global]
aliases: ["solubility"]
---

# Solubility (protein solubility classification) — dataset / task card

> Binary prediction of whether a protein is soluble when expressed, from sequence alone.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `solubility` |
| Scope | properties |
| Prediction level | protein-level (global) |
| Task type | classification |
| Label kind | `binary` |
| Target column | `solubility` |
| Label units | binary label (0/1), unitless |
| Primary metric | `f1` |
| Splits available | random, similarity |

> Benchmark-card description: Protein solubility classification.

## Source & provenance

- **Source dataset:** ProteinEA solubility (DeepSol lineage) (benchmark provenance: `ProteinEA`)

Each protein carries a 0/1 label for insoluble vs. soluble expression. The labels trace to the DeepSol resource (E. coli expression solubility), distributed via the `proteinea/solubility` dataset.

**Assay / measurement:** Experimental solubility upon heterologous (E. coli) expression, aggregated from prior solubility datasets; the original DeepSol split excludes test sequences ≥30% identical to training.

**Provenance & construction:** Downloaded from the ProteinEA Hugging Face dataset; the fluorescence sibling dataset is intentionally excluded from this enzyme benchmark.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 3665
- **Label balance (overall):** `1`=2056 (0.561), `0`=1609 (0.439)
- **Split sizes (train/val/test):** random → train 2932 / val 366 / test 367; similarity → train 2932 / val 366 / test 367
- **Sequence length:** min 61, median 329.0, mean 358.85, max 1645 (32 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 2994 clusters from 3665 sequences at 0.5 identity (2574 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `prot5_xl_half_uniref50` | 0.7828 | `attention_mlp_classification` |
| 2 | `prot5_xl_uniref50` | 0.7754 | `attention_mlp_classification` |
| 3 | `profluent_e1_600m` | 0.7613 | `mlp_classification` |
| 4 | `esm2_3B` | 0.7602 | `xgb_classification` |
| 5 | `ankh_large` | 0.7585 | `mlp_classification` |
| 6 | `esmc_600m` | 0.7564 | `mlp_classification` |
| 7 | `profluent_e1_300m` | 0.7517 | `xgb_classification` |
| 8 | `esmc_300m` | 0.7483 | `xgb_classification` |
| 9 | `esm2_650M` | 0.7438 | `xgb_classification` |
| 10 | `ankh3_xl` | 0.7417 | `xgb_classification` |
| 11 | `ankh3_large` | 0.7391 | `xgb_classification` |
| 12 | `esm3_open` | 0.7228 | `xgb_classification` |
| 13 | `ankh_base` | 0.7186 | `attention_linear_classification` |
| 14 | `prost5` | 0.7064 | `mlp_classification` |

### similarity split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `esm2_3B` | 0.7224 | `attention_mlp_classification` |
| 2 | `ankh3_xl` | 0.7134 | `linear_classification` |
| 3 | `profluent_e1_600m` | 0.7126 | `mlp_classification` |
| 4 | `prot5_xl_uniref50` | 0.7059 | `attention_mlp_classification` |
| 5 | `profluent_e1_300m` | 0.7051 | `linear_classification` |
| 6 | `esmc_600m` | 0.6989 | `xgb_classification` |
| 7 | `prot5_xl_half_uniref50` | 0.6907 | `attention_mlp_classification` |
| 8 | `ankh_large` | 0.6761 | `attention_mlp_classification` |
| 9 | `esm2_650M` | 0.6667 | `attention_mlp_classification` |
| 10 | `esmc_300m` | 0.6667 | `attention_mlp_classification` |
| 11 | `ankh3_large` | 0.6595 | `mlp_classification` |
| 12 | `esm3_open` | 0.6561 | `mlp_classification` |
| 13 | `ankh_base` | 0.6542 | `mlp_classification` |
| 14 | `prost5` | 0.6333 | `xgb_classification` |

## When to use

A protein-level classification probe of how well an embedding captures expressibility/solubility determinants — useful for expression-risk screening.

## Caveats

Solubility labels are noisy and host/condition dependent (largely E. coli); not enzyme-specific, so it tests general representation quality rather than catalysis.

## References
- Khurana, S. et al. (2018). *DeepSol: a deep learning framework for sequence-based protein solubility prediction*. Bioinformatics 34(15):2605–2613. <https://doi.org/10.1093/bioinformatics/bty166>
