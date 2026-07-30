---
id: model_ankh_base
type: foundation_model
title: "Ankh (base)"
family: "Ankh"
embedding_dim: 768
tags: [foundation-model, protein-language-model, ankh]
aliases: ["ankh_base"]
---

# Ankh (base) — foundation model card

> The compact Ankh model — a deliberately small, ablation-optimized T5 protein LM whose 768-dim embeddings rival much larger models at a fraction of the cost.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `ankh_base` |
| Family | Ankh |
| Developer | ElnaggarLab (Proteinea / TU Munich) |
| Parameters | ≈450M (T5 encoder–decoder) |
| Embedding dimension | **768** |
| Context length | T5 relative position bias (memory-limited; long sequences truncated in this benchmark) |
| HuggingFace | `ElnaggarLab/ankh-base` |

## Architecture & pretraining

T5 encoder–decoder with relative position bias, tuned through >20 architecture/pretraining ablations; 768-dim per-residue encoder embeddings used here.

- **Pretraining data:** UniRef50 (~45M sequences).
- **Pretraining objective:** T5-style masked-span denoising with 1-gram token masking (~20% probability) and full de-masking reconstruction.

**Key ideas:**
- Efficiency-first design: strong general-purpose performance with far fewer params/embedding dims than ProtT5/ESM
- Systematic ablation of masking, depth, and width to optimize the recipe

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/ankh_base_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **768**. No-attention baselines mean-pool the
residues into one vector; attention baselines learn the pooling instead (see the
baseline cards).

## Benchmark standing (auto-generated — refreshes when results change)

Competition score against the other foundation models, averaged over heads and
tasks. Each task/head group is scored `(n - rank) / (n - 1)` — **1.0 = beat every
competitor, 0.0 = beaten by all** — so groups where only a few of the
14 models ran count on the same scale as full 14-way groups. The raw
mean rank is shown alongside it, but compare on the score: it is the rank-based
number that survives the uneven group sizes.

| Track | Score (1 = best) | Mean rank | Groups averaged |
|---|---|---|---|
| No-attention · Random | 0.518 | 7.13 | 52 |
| No-attention · Similarity | 0.566 | 6.54 | 28 |
| Attention · Random | 0.651 | 5.19 | 31 |
| Attention · Similarity | 0.447 | 8.19 | 16 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9819 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2463 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1343 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.1604 | `attention_mlp_regressor` |
| `align_glucosidase_activity` | mae | 0.2066 | `attention_linear_regressor` |
| `align_glucosidase_expression` | mae | 0.5483 | `attention_linear_regressor` |
| `align_glucosidase_melting` | mae | 0.6616 | `xgb_regression` |
| `binding` | f1 | 0.8178 | `xgb_classification` |
| `binding_site` | f1 | 0.9025 | `mlp_classification` |
| `cleeveland_galox` | mae | 0.0624 | `xgb_regression` |
| `ddg` | mae | 0.6889 | `mlp_regressor` |
| `dtm` | mae | 0.452 | `xgb_regression` |
| `ec_number` | f1 | 0.9544 | `attention_linear_classification` |
| `erickson_petase` | mae | 1.7795 | `xgb_regression` |
| `kcat` | mae | 0.6548 | `xgb_regression` |
| `km` | mae | 0.6028 | `xgb_regression` |
| `ph_opt` | mae | 0.5139 | `xgb_regression` |
| `solubility` | f1 | 0.7186 | `attention_linear_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9146 | `mlp_classification` |
| `binding` | f1 | 0.836 | `xgb_classification` |
| `binding_site` | f1 | 0.6668 | `mlp_classification` |
| `ddg` | mae | 0.478 | `mlp_regressor` |
| `dtm` | mae | 0.5956 | `mlp_regressor` |
| `ec_number` | f1 | 0.796 | `attention_linear_classification` |
| `kcat` | mae | 0.6431 | `attention_mlp_regressor` |
| `km` | mae | 0.6456 | `xgb_regression` |
| `ph_opt` | mae | 0.7261 | `xgb_regression` |
| `solubility` | f1 | 0.6542 | `mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When you want a cheap, low-dimensional (768) embedding that is still competitive — good for large-scale embedding or memory-constrained heads.

## Caveats

Smallest Ankh; may trail larger models on the hardest tasks. Single-sequence, sequence-only.

## References
- Elnaggar, A. et al. (2023). *Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling*. bioRxiv 2023.01.16.524265. <https://doi.org/10.1101/2023.01.16.524265>
