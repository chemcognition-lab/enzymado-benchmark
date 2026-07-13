---
id: task_active_sites
type: task
title: "Active sites (catalytic-residue prediction)"
task_name: active_sites
task_scope: properties
task_type: classification
level: residue
source_dataset: "UniProt / Swiss-Prot"
tags: [dataset, task, properties, classification, residue]
aliases: ["active_sites"]
---

# Active sites (catalytic-residue prediction) — dataset / task card

> Per-residue binary prediction of which residues are catalytic active-site residues.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `active_sites` |
| Scope | properties |
| Prediction level | per-residue |
| Task type | classification |
| Label kind | `residue_binary_vector` |
| Target column | — (derived from residue positions) |
| Label units | residue indices (1-based) |
| Primary metric | `f1` |
| Splits available | random, similarity |
| Residue-positions column | `active_site` |

> Benchmark-card description: Active site residue annotation from sequence.

## Source & provenance

- **Source dataset:** UniProt / Swiss-Prot (benchmark provenance: `UniProt/Swiss-Prot`)

For each residue in a protein, a 0/1 label for whether it is an annotated active site. Labels come from Swiss-Prot 'Active site' sequence features (1-based positions expanded to a per-residue binary vector).

**Assay / measurement:** Expert-curated active-site annotations in Swiss-Prot, based on experimental and homology evidence.

**Provenance & construction:** Parsed from the Swiss-Prot `active_site` feature column into residue-position vectors; the model keeps the full per-residue embedding tensor.

## Featurization

The full per-residue embedding tensor (length × dim) is retained; labels are a binary vector over residue positions.

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 90436
- **Split sizes (train/val/test):** random → train 72349 / val 9044 / test 9043; similarity → train 72349 / val 9043 / test 9044
- **Sequence length:** min 8, median 352.0, mean 439.37, max 35213 (3462 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 17764 clusters from 89725 sequences at 0.5 identity (9347 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `prot5_xl_half_uniref50` | 0.9894 | `mlp_classification` |
| 2 | `esm2_3B` | 0.9881 | `mlp_classification` |
| 3 | `prost5` | 0.988 | `mlp_classification` |
| 4 | `esm2_650M` | 0.9875 | `mlp_classification` |
| 5 | `esm3_open` | 0.9874 | `mlp_classification` |
| 6 | `profluent_e1_600m` | 0.9864 | `mlp_classification` |
| 7 | `ankh3_xl` | 0.9862 | `mlp_classification` |
| 8 | `prot5_xl_uniref50` | 0.9857 | `mlp_classification` |
| 9 | `profluent_e1_300m` | 0.9842 | `mlp_classification` |
| 10 | `ankh_base` | 0.9819 | `mlp_classification` |
| 11 | `ankh_large` | 0.9797 | `mlp_classification` |
| 12 | `ankh3_large` | 0.9764 | `mlp_classification` |
| 13 | `esmc_600m` | 0.9732 | `mlp_classification` |
| 14 | `esmc_300m` | 0.9676 | `mlp_classification` |

### similarity split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `esm3_open` | 0.9194 | `mlp_classification` |
| 2 | `ankh3_xl` | 0.9181 | `mlp_classification` |
| 3 | `ankh_base` | 0.9146 | `mlp_classification` |
| 4 | `profluent_e1_600m` | 0.9006 | `mlp_classification` |
| 5 | `esm2_3B` | 0.89 | `mlp_classification` |
| 6 | `ankh_large` | 0.888 | `mlp_classification` |
| 7 | `prost5` | 0.8858 | `mlp_classification` |
| 8 | `esm2_650M` | 0.8794 | `mlp_classification` |
| 9 | `ankh3_large` | 0.867 | `mlp_classification` |
| 10 | `profluent_e1_300m` | 0.8648 | `mlp_classification` |
| 11 | `prot5_xl_half_uniref50` | 0.8571 | `mlp_classification` |
| 12 | `prot5_xl_uniref50` | 0.8422 | `mlp_classification` |
| 13 | `esmc_300m` | 0.7808 | `mlp_classification` |
| 14 | `esmc_600m` | 0.7557 | `mlp_classification` |

## When to use

A residue-level localization task testing whether embeddings pinpoint catalytic machinery — relevant to mechanism annotation and active-site design.

## Caveats

Extremely imbalanced (few positive residues per protein), so F1 is dominated by rare positives. Annotation coverage is incomplete; the residue target mask is derived from the padding mask.

## References
- The UniProt Consortium (2025). *UniProt: the Universal Protein Knowledgebase in 2025*. Nucleic Acids Research 53(D1):D609–D617. <https://doi.org/10.1093/nar/gkae1010>
