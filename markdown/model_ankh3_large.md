---
id: model_ankh3_large
type: foundation_model
title: "Ankh3 (large)"
family: "Ankh3"
embedding_dim: 1536
tags: [foundation-model, protein-language-model, ankh3]
aliases: ["ankh3_large"]
---

# Ankh3 (large) — foundation model card

> The large Ankh3 model — a next-generation Ankh trained with multi-task denoising plus sequence completion, showing that sequence-only multi-task pretraining can rival multimodal models.
>
> ⚠️ **Needs verification:** some specifics below are best-effort for this recent / thin-source model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `ankh3_large` |
| Family | Ankh3 |
| Developer | ElnaggarLab (Proteinea / TU Munich) |
| Parameters | Not officially disclosed per size at time of writing (Ankh3-Large; 1536-dim embeddings) |
| Embedding dimension | **1536** |
| Context length | T5 relative position bias (memory-limited; long sequences truncated in this benchmark) |
| HuggingFace | `ElnaggarLab/ankh3-large` |

## Architecture & pretraining

T5-style encoder–decoder (Ankh line); 1536-dim per-residue encoder embeddings. Uses task-prefix conditioning for representation vs. generation.

- **Pretraining data:** ≈59M sequences from UniRef50 (JAX/T5x on TPU-v4).
- **Pretraining objective:** Joint multi-task pretraining: masked language modeling at multiple masking probabilities plus protein sequence completion (sequence-only).

**Key ideas:**
- Combines MLM (multi-probability) with sequence-completion objectives
- Shows unimodal, sequence-only multi-task design can match or surpass multimodal PLMs on downstream tasks

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/ankh3_large_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1536**. No-attention baselines mean-pool the
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
| No-attention · Random | 0.339 | 9.59 | 51 |
| No-attention · Similarity | 0.447 | 8.19 | 27 |
| Attention · Random | 0.461 | 7.70 | 30 |
| Attention · Similarity | 0.209 | 11.29 | 14 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9764 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2208 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1308 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2458 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.2118 | `attention_mlp_regressor` |
| `align_glucosidase_expression` | mae | 0.5494 | `attention_linear_regressor` |
| `align_glucosidase_melting` | mae | 0.6796 | `attention_mlp_regressor` |
| `binding` | f1 | 0.8017 | `xgb_classification` |
| `binding_site` | f1 | 0.367 | `linear_classification` |
| `cleeveland_galox` | mae | 0.0576 | `xgb_regression` |
| `ddg` | mae | 0.7378 | `xgb_regression` |
| `dtm` | mae | 0.4608 | `mlp_regressor` |
| `ec_number` | f1 | 0.9451 | `linear_classification` |
| `erickson_petase` | mae | 1.9512 | `attention_mlp_regressor` |
| `kcat` | mae | 0.6541 | `xgb_regression` |
| `km` | mae | 0.6189 | `xgb_regression` |
| `ph_opt` | mae | 0.5418 | `xgb_regression` |
| `solubility` | f1 | 0.7391 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.867 | `mlp_classification` |
| `binding` | f1 | 0.8193 | `xgb_classification` |
| `binding_site` | f1 | 0.3953 | `linear_classification` |
| `ddg` | mae | 0.5509 | `xgb_regression` |
| `dtm` | mae | 0.6464 | `mlp_regressor` |
| `ec_number` | f1 | 0.6894 | `linear_classification` |
| `kcat` | mae | 0.6311 | `linear_regressor` |
| `km` | mae | 0.6337 | `xgb_regression` |
| `ph_opt` | mae | 0.7555 | `xgb_regression` |
| `solubility` | f1 | 0.6595 | `mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When you want a recent, sequence-only model with strong reported downstream performance at a 1536-dim embedding size.

## Caveats

Very recent (2025); exact parameter counts and training details are thin in public sources — verify before quoting specifics.

## References
- Elnaggar, A. et al. (2025). *Ankh3: Multi-Task Pretraining with Sequence Denoising and Completion Enhances Protein Representations*. arXiv:2505.20052. <https://arxiv.org/abs/2505.20052>
