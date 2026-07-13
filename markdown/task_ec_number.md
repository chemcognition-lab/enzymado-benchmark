---
id: task_ec_number
type: task
title: "EC number (enzyme commission classification)"
task_name: ec_number
task_scope: properties
task_type: classification
level: protein-global
source_dataset: "UniProt / Swiss-Prot"
tags: [dataset, task, properties, classification, protein-global]
aliases: ["ec_number"]
---

# EC number (enzyme commission classification) — dataset / task card

> Hierarchical multi-label prediction of Enzyme Commission (EC) numbers from sequence — which reaction class(es) an enzyme catalyzes.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `ec_number` |
| Scope | properties |
| Prediction level | protein-level (global) |
| Task type | classification |
| Label kind | `multilabel_ec` |
| Target column | `ec_number` |
| Label units | EC identifier, unitless |
| Primary metric | `f1` |
| Splits available | random, similarity |

> Benchmark-card description: Enzyme Commission number prediction from sequence.

## Source & provenance

- **Source dataset:** UniProt / Swiss-Prot (benchmark provenance: `UniProt/Swiss-Prot`)

Each protein is labeled with one or more EC numbers (four-level hierarchy, e.g. 3.1.1.101). The benchmark treats this as multi-label classification over the EC hierarchy (models use per-level heads or flat blocks).

**Assay / measurement:** Manually curated functional annotation in Swiss-Prot (the reviewed section of UniProtKB), derived from experimental and expert-reviewed evidence.

**Provenance & construction:** Downloaded via the UniProt REST API (`reviewed:true AND ec:*`); EC numbers are split on `;` into multi-labels.

## Featurization

The protein embedding (mean-pooled residues for no-attention heads, or attention-pooled for attention heads) is fed directly to the head.

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 237526
- **Split sizes (train/val/test):** random → train 190021 / val 23752 / test 23753; similarity → train 190021 / val 23753 / test 23752
- **Sequence length:** min 4, median 359.0, mean 428.85, max 35213 (8616 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 46958 clusters from 237474 sequences at 0.5 identity (25629 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `esm2_3B` | 0.9729 | `linear_classification` |
| 2 | `ankh3_xl` | 0.9711 | `linear_classification` |
| 3 | `esm2_650M` | 0.9638 | `linear_classification` |
| 4 | `prost5` | 0.9584 | `linear_classification` |
| 5 | `ankh_base` | 0.9544 | `attention_linear_classification` |
| 6 | `ankh_large` | 0.9528 | `linear_classification` |
| 7 | `prot5_xl_uniref50` | 0.9514 | `linear_classification` |
| 8 | `prot5_xl_half_uniref50` | 0.9513 | `linear_classification` |
| 9 | `ankh3_large` | 0.9451 | `linear_classification` |
| 10 | `profluent_e1_600m` | 0.9252 | `linear_classification` |
| 11 | `profluent_e1_300m` | 0.9249 | `linear_classification` |
| 12 | `esm3_open` | 0.9217 | `linear_classification` |
| 13 | `esmc_600m` | 0.9162 | `linear_classification` |
| 14 | `esmc_300m` | 0.9136 | `linear_classification` |

### similarity split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `esm2_3B` | 0.8185 | `linear_classification` |
| 2 | `ankh_base` | 0.796 | `attention_linear_classification` |
| 3 | `prost5` | 0.7859 | `linear_classification` |
| 4 | `ankh3_xl` | 0.7231 | `linear_classification` |
| 5 | `esm2_650M` | 0.704 | `linear_classification` |
| 6 | `ankh3_large` | 0.6894 | `linear_classification` |
| 7 | `ankh_large` | 0.6516 | `linear_classification` |
| 8 | `prot5_xl_half_uniref50` | 0.6502 | `attention_linear_classification` |
| 9 | `prot5_xl_uniref50` | 0.6305 | `mlp_classification` |
| 10 | `profluent_e1_300m` | 0.5981 | `linear_classification` |
| 11 | `esm3_open` | 0.5875 | `linear_classification` |
| 12 | `profluent_e1_600m` | 0.5691 | `attention_linear_classification` |
| 13 | `esmc_600m` | 0.5671 | `attention_linear_classification` |
| 14 | `esmc_300m` | 0.4927 | `linear_classification` |

## When to use

A protein-level functional-classification task probing whether embeddings organize enzymes by catalyzed reaction — a strong signal for many models.

## Caveats

Label frequencies are highly imbalanced across EC classes; multi-label macro-F1 is sensitive to rare classes. Annotation completeness varies.

## References
- The UniProt Consortium (2025). *UniProt: the Universal Protein Knowledgebase in 2025*. Nucleic Acids Research 53(D1):D609–D617. <https://doi.org/10.1093/nar/gkae1010>
