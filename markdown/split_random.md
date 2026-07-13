---
id: split_random
type: split
title: "Random split (80/10/10)"
split_family: random
tags: [data-split, random, generalization]
aliases: ["random"]
---

# Random split (80/10/10) — data-split card

> Group-aware random partition of each dataset into 80% train / 10% validation / 10% test, keeping all rows of the same protein together. Measures in-distribution performance.

- **Split id (this benchmark):** `random`

## How this split is constructed

Rows are grouped by sequence (or a configured group column) and whole groups are greedily assigned to splits to approximate the 80/10/10 target (seed 42), so identical sequences never straddle train and test. Regression targets are z-standardized with statistics fit on the training split only (a leakage guard), producing the `*_standardized` columns. This is the default split and the only one available for the 8 real_case tasks.

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

Use for an optimistic, in-distribution estimate and for the applied real_case engineering campaigns, where held-out variants closely resemble training variants. Appropriate for model comparison when the deployment distribution matches the training data.

## Caveats

Because homologous sequences can land in both train and test, scores are inflated relative to novel-sequence generalization. Always compare against the similarity split to read the generalization gap.

## References
No external reference — this split is a methodology defined in the benchmark's `src/dataset_splitting.py`.
