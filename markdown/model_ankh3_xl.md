---
id: model_ankh3_xl
type: foundation_model
title: "Ankh3 (XL)"
family: "Ankh3"
embedding_dim: 2560
tags: [foundation-model, protein-language-model, ankh3]
aliases: ["ankh3_xl"]
---

# Ankh3 (XL) — foundation model card

> The XL Ankh3 model — the largest Ankh3 variant, with 2560-dim embeddings, trained with the same multi-task denoising + sequence-completion recipe.
>
> ⚠️ **Needs verification:** some specifics below are best-effort for this recent / thin-source model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `ankh3_xl` |
| Family | Ankh3 |
| Developer | ElnaggarLab (Proteinea / TU Munich) |
| Parameters | Not officially disclosed per size at time of writing (Ankh3-XL; 2560-dim embeddings) |
| Embedding dimension | **2560** |
| Context length | T5 relative position bias (memory-limited; long sequences truncated in this benchmark) |
| HuggingFace | `ElnaggarLab/ankh3-xl` |

## Architecture & pretraining

T5-style encoder–decoder (Ankh line); 2560-dim per-residue encoder embeddings; task-prefix conditioning for representation vs. generation.

- **Pretraining data:** ≈59M sequences from UniRef50 (JAX/T5x on TPU-v4).
- **Pretraining objective:** Multi-task: masked language modeling at multiple masking probabilities plus protein sequence completion (sequence-only).

**Key ideas:**
- Largest Ankh3 variant; highest embedding dimension (2560) in this benchmark's Ankh line
- Multi-task sequence-only pretraining aimed at richer, more generalizable representations

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/ankh3_xl_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **2560**. No-attention baselines mean-pool the
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
| No-attention · Random | 0.478 | 7.78 | 51 |
| No-attention · Similarity | 0.544 | 6.93 | 27 |
| Attention · Random | 0.271 | 10.47 | 19 |
| Attention · Similarity | 0.258 | 10.64 | 14 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9862 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2348 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1335 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.3227 | `xgb_regression` |
| `align_glucosidase_activity` | mae | 90.7729 | `xgb_regression` |
| `align_glucosidase_expression` | mae | 0.5471 | `xgb_regression` |
| `align_glucosidase_melting` | mae | 0.7551 | `xgb_regression` |
| `binding` | f1 | 0.8063 | `xgb_classification` |
| `binding_site` | f1 | 0.4531 | `linear_classification` |
| `cleeveland_galox` | mae | 0.1321 | `linear_regressor` |
| `ddg` | mae | 0.6461 | `mlp_regressor` |
| `dtm` | mae | 0.3909 | `mlp_regressor` |
| `ec_number` | f1 | 0.9711 | `linear_classification` |
| `erickson_petase` | mae | 1.717 | `xgb_regression` |
| `kcat` | mae | 0.6551 | `xgb_regression` |
| `km` | mae | 0.6279 | `xgb_regression` |
| `ph_opt` | mae | 0.5302 | `xgb_regression` |
| `solubility` | f1 | 0.7417 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9181 | `mlp_classification` |
| `binding` | f1 | 0.8255 | `xgb_classification` |
| `binding_site` | f1 | 0.4311 | `linear_classification` |
| `ddg` | mae | 0.5236 | `mlp_regressor` |
| `dtm` | mae | 0.6164 | `mlp_regressor` |
| `ec_number` | f1 | 0.7231 | `linear_classification` |
| `kcat` | mae | 0.6036 | `xgb_regression` |
| `km` | mae | 0.6591 | `xgb_regression` |
| `ph_opt` | mae | 0.748 | `linear_regressor` |
| `solubility` | f1 | 0.7134 | `linear_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When maximum Ankh3 capacity / high-dimensional embeddings are desired and the extra compute is acceptable.

## Caveats

Very recent (2025); parameter counts and full training details are not clearly published — verify before quoting. High embedding dim increases downstream head cost.

## References
- Elnaggar, A. et al. (2025). *Ankh3: Multi-Task Pretraining with Sequence Denoising and Completion Enhances Protein Representations*. arXiv:2505.20052. <https://arxiv.org/abs/2505.20052>
