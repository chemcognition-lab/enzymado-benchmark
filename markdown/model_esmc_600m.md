---
id: model_esmc_600m
type: foundation_model
title: "ESM Cambrian (ESM-C 600M)"
family: "ESM-C"
embedding_dim: 1152
tags: [foundation-model, protein-language-model, esm-c]
aliases: ["esmc_600m"]
---

# ESM Cambrian (ESM-C 600M) — foundation model card

> The 600M ESM Cambrian encoder — reportedly rivals ESM-2 3B while being far smaller, producing 1152-dim embeddings.
>
> ⚠️ **Needs verification:** some specifics below are best-effort for this recent / thin-source model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `esmc_600m` |
| Family | ESM-C |
| Developer | EvolutionaryScale |
| Parameters | 600M |
| Embedding dimension | **1152** |
| Context length | Up to ~2048 residues (benchmark truncates longer sequences) |
| HuggingFace | `EvolutionaryScale/esmc-600m-2024-12` |

## Architecture & pretraining

Encoder-only masked transformer in the ESM-C representation line; 1152-dim per-residue embeddings.

- **Pretraining data:** Large-scale natural protein sequence corpora (EvolutionaryScale; see the ESM-C release).
- **Pretraining objective:** Masked language modeling.

**Key ideas:**
- 600M reportedly matches ESM-2 3B and approaches the 15B model at a fraction of the size
- Open-weight, representation-focused counterpart to ESM3

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/esmc_600m_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1152**. No-attention baselines mean-pool the
residues into one vector; attention baselines learn the pooling instead (see the
baseline cards).

## Benchmark standing (auto-generated — refreshes when results change)

Mean rank among the 14 foundation models (1 = best), averaged over
heads and tasks:

| Track | Mean rank (of 14) | Groups averaged |
|---|---|---|
| No-attention · Random | 7.9 | 51 |
| No-attention · Similarity | 6.82 | 28 |
| Attention · Random | 6.0 | 31 |
| Attention · Similarity | 3.25 | 16 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9732 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2198 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1313 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2581 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.212 | `attention_mlp_regressor` |
| `align_glucosidase_expression` | mae | 0.5436 | `xgb_regression` |
| `align_glucosidase_melting` | mae | 0.6769 | `attention_mlp_regressor` |
| `binding` | f1 | 0.8068 | `xgb_classification` |
| `binding_site` | f1 | 0.3711 | `linear_classification` |
| `cleeveland_galox` | mae | 0.0612 | `xgb_regression` |
| `ddg` | mae | 0.6179 | `mlp_regressor` |
| `dtm` | mae | 0.3802 | `xgb_regression` |
| `ec_number` | f1 | 0.9162 | `linear_classification` |
| `erickson_petase` | mae | 1.7931 | `xgb_regression` |
| `kcat` | mae | 0.6585 | `xgb_regression` |
| `km` | mae | 0.6376 | `xgb_regression` |
| `ph_opt` | mae | 0.5532 | `xgb_regression` |
| `solubility` | f1 | 0.7564 | `mlp_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.7557 | `mlp_classification` |
| `binding` | f1 | 0.8111 | `xgb_classification` |
| `binding_site` | f1 | 0.4574 | `mlp_classification` |
| `ddg` | mae | 0.4579 | `mlp_regressor` |
| `dtm` | mae | 0.5825 | `mlp_regressor` |
| `ec_number` | f1 | 0.5671 | `attention_linear_classification` |
| `kcat` | mae | 0.5913 | `attention_linear_regressor` |
| `km` | mae | 0.6598 | `xgb_regression` |
| `ph_opt` | mae | 0.6815 | `attention_linear_regressor` |
| `solubility` | f1 | 0.6989 | `xgb_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When you want near-ESM-2-3B quality without the 3B footprint; a strong modern default in the mid-size range (1152-dim).

## Caveats

Documented mainly via the EvolutionaryScale release/blog and independent evaluations rather than a peer-reviewed paper — verify specifics before relying on exact numbers.

## References
- EvolutionaryScale Team (2024). *ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning*. EvolutionaryScale (release note). <https://www.evolutionaryscale.ai/blog/esm-cambrian>
