---
id: split_similarity
type: split
title: "Similarity (MMseqs2) split"
split_family: loco
tags: [data-split, similarity, generalization]
aliases: ["similarity", "loco"]
---

# Similarity (MMseqs2) split — data-split card

> Homology-aware 80/10/10 split: sequences are clustered by identity with MMseqs2 and whole clusters go to a single split, so no test sequence is homologous to any training sequence. Estimates generalization to novel proteins (leave-one-cluster-out).

- **Split id (this benchmark):** `similarity` (recorded as `loco` in results)

## How this split is constructed

Sequences are deduplicated, clustered with MMseqs2 at a minimum sequence identity of 0.3 (coverage 0.8), and entire clusters are assigned to one split to approximate 80/10/10. As with the random split, targets are standardized on train only. Available for the 10 `properties` tasks (real_case is random-only) and recorded as the `loco` (leave-one-cluster-out) family in the results tree.

## Generalization gap (auto — random vs similarity, `properties` tasks)

Mean held-out primary metric across all foundation models and heads. The
similarity (MMseqs2) split holds out homologous clusters, so it is the harder,
more realistic estimate of generalization to novel sequences.

| Task | Primary metric | Random | Similarity | Similarity harder by |
|---|---|---|---|---|
| `active_sites` | f1 | 0.8171 | 0.6586 | 0.1586 |
| `binding` | f1 | 0.5896 | 0.6186 | -0.029 |
| `binding_site` | f1 | 0.5304 | 0.507 | 0.0235 |
| `ddg` | mae | 0.7522 | 0.5516 | -0.2006 |
| `dtm` | mae | 0.7353 | 1.0406 | 0.3053 |
| `ec_number` | f1 | 0.7818 | 0.4819 | 0.2999 |
| `kcat` | mae | 0.7104 | 0.635 | -0.0753 |
| `km` | mae | 0.679 | 0.693 | 0.0139 |
| `ph_opt` | mae | 0.5771 | 0.7436 | 0.1666 |
| `solubility` | f1 | 0.7067 | 0.6538 | 0.0529 |

_"Harder by" is sign-adjusted: positive means the similarity split scores worse.
real_case tasks are random-only and excluded._

## When to use

Use as the realistic estimate of performance on sequences dissimilar to the training data — the setting that matters for discovering or engineering novel enzymes. Prefer it when ranking foundation models for out-of-distribution generalization.

## Caveats

Harder than the random split (see the generalization-gap table): absolute scores are lower and, for tasks with strong distribution shift, model ranking can differ from the random split. The 0.3 identity threshold is a design choice; stricter thresholds would push toward even harder generalization.

## References
- Steinegger, M. & Söding, J. (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets*. Nature Biotechnology 35:1026–1028. <https://doi.org/10.1038/nbt.3988>
