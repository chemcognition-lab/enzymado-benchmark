---
id: model_ankh_large
type: foundation_model
title: "Ankh (large)"
family: "Ankh"
embedding_dim: 1536
tags: [foundation-model, protein-language-model, ankh]
aliases: ["ankh_large"]
---

# Ankh (large) — foundation model card

> The full-size Ankh model — 1.15B-parameter T5 with 1536-dim embeddings that reached general-purpose state of the art with far fewer parameters than contemporaries.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `ankh_large` |
| Family | Ankh |
| Developer | ElnaggarLab (Proteinea / TU Munich) |
| Parameters | ≈1.15B (T5 encoder–decoder) |
| Embedding dimension | **1536** |
| Context length | T5 relative position bias (memory-limited; long sequences truncated in this benchmark) |
| HuggingFace | `ElnaggarLab/ankh-large` |

## Architecture & pretraining

T5 encoder–decoder with relative position bias (larger embedding dim, more attention heads and feed-forward layers than Ankh-base); 1536-dim per-residue encoder embeddings.

- **Pretraining data:** UniRef50 (~45M sequences).
- **Pretraining objective:** T5-style masked-span denoising (1-gram masking ~20%, full reconstruction).

**Key ideas:**
- State-of-the-art general-purpose PLM performance at <10% of prior training params
- Ablation-optimized recipe; TPU-v4 pretraining

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/ankh_large_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1536**. No-attention baselines mean-pool the
residues into one vector; attention baselines learn the pooling instead (see the
baseline cards).

## Benchmark standing (auto-generated — refreshes when results change)

Mean rank among the 14 foundation models (1 = best), averaged over
heads and tasks:

| Track | Mean rank (of 14) | Groups averaged |
|---|---|---|
| No-attention · Random | 6.16 | 51 |
| No-attention · Similarity | 7.52 | 27 |
| Attention · Random | 6.57 | 30 |
| Attention · Similarity | 10.0 | 14 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9797 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.205 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1192 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.149 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.211 | `attention_mlp_regressor` |
| `align_glucosidase_expression` | mae | 0.5473 | `attention_linear_regressor` |
| `align_glucosidase_melting` | mae | 0.6682 | `attention_linear_regressor` |
| `binding` | f1 | 0.8076 | `xgb_classification` |
| `binding_site` | f1 | 0.4638 | `linear_classification` |
| `cleeveland_galox` | mae | 0.3387 | `attention_mlp_regressor` |
| `ddg` | mae | 0.6101 | `mlp_regressor` |
| `dtm` | mae | 0.4263 | `xgb_regression` |
| `ec_number` | f1 | 0.9528 | `linear_classification` |
| `erickson_petase` | mae | 1.9093 | `mlp_regressor` |
| `kcat` | mae | 0.6482 | `xgb_regression` |
| `km` | mae | 0.6157 | `xgb_regression` |
| `ph_opt` | mae | 0.5274 | `xgb_regression` |
| `solubility` | f1 | 0.7585 | `mlp_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.888 | `mlp_classification` |
| `binding` | f1 | 0.8229 | `xgb_classification` |
| `binding_site` | f1 | 0.4573 | `linear_classification` |
| `ddg` | mae | 0.4491 | `xgb_regression` |
| `dtm` | mae | 0.6364 | `xgb_regression` |
| `ec_number` | f1 | 0.6516 | `linear_classification` |
| `kcat` | mae | 0.6355 | `xgb_regression` |
| `km` | mae | 0.6527 | `xgb_regression` |
| `ph_opt` | mae | 0.6986 | `mlp_regressor` |
| `solubility` | f1 | 0.6761 | `attention_mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

A strong, efficient alternative to ProtT5/ESM-2 for enzyme property prediction; 1536-dim embeddings balance quality and cost.

## Caveats

Encoder–decoder checkpoint carries unused decoder weights for embedding use; single-sequence, sequence-only.

## References
- Elnaggar, A. et al. (2023). *Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling*. bioRxiv 2023.01.16.524265. <https://doi.org/10.1101/2023.01.16.524265>
