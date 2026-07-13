---
id: task_binding_site
type: task
title: "Binding site (ligand-binding-residue prediction)"
task_name: binding_site
task_scope: properties
task_type: classification
level: residue
source_dataset: "UniProt / Swiss-Prot"
tags: [dataset, task, properties, classification, residue]
aliases: ["binding_site"]
---

# Binding site (ligand-binding-residue prediction) — dataset / task card

> Per-residue binary prediction of which residues form ligand/cofactor binding sites.

## Task definition (auto)

| Field | Value |
|---|---|
| Registry name | `binding_site` |
| Scope | properties |
| Prediction level | per-residue |
| Task type | classification |
| Label kind | `residue_binary_vector` |
| Target column | — (derived from residue positions) |
| Label units | residue indices (1-based) |
| Primary metric | `f1` |
| Splits available | random, similarity |
| Residue-positions column | `binding_positions` |

> Benchmark-card description: Binding site residue annotation from sequence.

## Source & provenance

- **Source dataset:** UniProt / Swiss-Prot (benchmark provenance: `UniProt/Swiss-Prot`)

For each residue, a 0/1 label for whether it is an annotated binding-site residue. Labels come from Swiss-Prot 'Binding site' sequence features (positions expanded to a per-residue binary vector).

**Assay / measurement:** Expert-curated binding-site annotations in Swiss-Prot (substrate/cofactor/metal contacts), from experimental and homology evidence.

**Provenance & construction:** Parsed from the Swiss-Prot `binding_positions` feature column into residue-position vectors; the full per-residue embedding tensor is retained by the model.

## Featurization

The full per-residue embedding tensor (length × dim) is retained; labels are a binary vector over residue positions.

## Dataset statistics (auto — from `dataset_card.json`)

- **Datapoints after cleaning:** 158681
- **Split sizes (train/val/test):** random → train 126945 / val 15868 / test 15868; similarity → train 126945 / val 15868 / test 15868
- **Sequence length:** min 8, median 369.0, mean 437.49, max 35213 (5541 > 1024 residues)
- **Protein-pool MMseqs2 clustering (diagnostic — not the split threshold):** 27759 clusters from 157433 sequences at 0.5 identity (14338 singletons)

## Leaderboard (auto — best test score per model)

### random split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `prot5_xl_uniref50` | 0.941 | `mlp_classification` |
| 2 | `profluent_e1_300m` | 0.9165 | `mlp_classification` |
| 3 | `ankh_base` | 0.9025 | `mlp_classification` |
| 4 | `esmc_300m` | 0.7958 | `mlp_classification` |
| 5 | `esm2_3B` | 0.5214 | `linear_classification` |
| 6 | `ankh_large` | 0.4638 | `linear_classification` |
| 7 | `esm2_650M` | 0.4625 | `linear_classification` |
| 8 | `ankh3_xl` | 0.4531 | `linear_classification` |
| 9 | `prot5_xl_half_uniref50` | 0.4417 | `linear_classification` |
| 10 | `profluent_e1_600m` | 0.4279 | `linear_classification` |
| 11 | `prost5` | 0.4272 | `linear_classification` |
| 12 | `esm3_open` | 0.4107 | `linear_classification` |
| 13 | `esmc_600m` | 0.3711 | `linear_classification` |
| 14 | `ankh3_large` | 0.367 | `linear_classification` |

### similarity split

| Rank | Model | f1 | Winning head |
|---|---|---|---|
| 1 | `prost5` | 0.7028 | `mlp_classification` |
| 2 | `esm3_open` | 0.6925 | `mlp_classification` |
| 3 | `prot5_xl_half_uniref50` | 0.6862 | `mlp_classification` |
| 4 | `esm2_650M` | 0.6763 | `mlp_classification` |
| 5 | `prot5_xl_uniref50` | 0.6757 | `mlp_classification` |
| 6 | `ankh_base` | 0.6668 | `mlp_classification` |
| 7 | `profluent_e1_300m` | 0.6482 | `mlp_classification` |
| 8 | `esmc_300m` | 0.5374 | `mlp_classification` |
| 9 | `esm2_3B` | 0.4964 | `linear_classification` |
| 10 | `esmc_600m` | 0.4574 | `mlp_classification` |
| 11 | `ankh_large` | 0.4573 | `linear_classification` |
| 12 | `profluent_e1_600m` | 0.4345 | `linear_classification` |
| 13 | `ankh3_xl` | 0.4311 | `linear_classification` |
| 14 | `ankh3_large` | 0.3953 | `linear_classification` |

## When to use

A residue-level localization task probing whether embeddings identify binding pockets — useful for functional-site annotation and engineering.

## Caveats

Highly imbalanced per protein; F1 is driven by rare positives. Distinct from BioLiP binding (which is a protein+ligand classification, not residue labels). Annotation coverage is incomplete.

## References
- The UniProt Consortium (2025). *UniProt: the Universal Protein Knowledgebase in 2025*. Nucleic Acids Research 53(D1):D609–D617. <https://doi.org/10.1093/nar/gkae1010>
