---
id: model_prot5_xl_half_uniref50
type: foundation_model
title: "ProtT5-XL-UniRef50 (half precision, encoder-only)"
family: "ProtT5 (ProtTrans)"
embedding_dim: 1024
tags: [foundation-model, protein-language-model, prott5-(prottrans)]
aliases: ["prot5_xl_half_uniref50"]
---

# ProtT5-XL-UniRef50 (half precision, encoder-only) — foundation model card

> A half-precision, encoder-only distribution of ProtT5-XL-UniRef50 — the same 1024-dim representations at roughly half the memory/compute of the full model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `prot5_xl_half_uniref50` |
| Family | ProtT5 (ProtTrans) |
| Developer | Rostlab (TU Munich) et al. |
| Parameters | ~3B (T5-3B encoder, fp16 weights) |
| Embedding dimension | **1024** |
| Context length | Effectively unbounded via T5 relative position bias (memory-limited; benchmark truncates very long sequences) |
| HuggingFace | `Rostlab/prot_t5_xl_half_uniref50-enc` |

## Architecture & pretraining

Identical T5-3B encoder to ProtT5-XL-UniRef50, shipped as encoder-only float16 weights; produces 1024-dim per-residue embeddings.

- **Pretraining data:** BFD then UniRef50 (same as the full ProtT5-XL-UniRef50).
- **Pretraining objective:** T5-style span-corruption denoising.

**Key ideas:**
- fp16 encoder-only packaging removes the unused decoder and halves memory
- Near-identical embeddings to the full-precision model at lower cost

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/prot5_xl_half_uniref50_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1024**. No-attention baselines mean-pool the
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
| No-attention · Random | 0.563 | 6.69 | 51 |
| No-attention · Similarity | 0.546 | 6.86 | 28 |
| Attention · Random | 0.645 | 5.10 | 31 |
| Attention · Similarity | 0.713 | 4.67 | 15 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9894 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2337 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.132 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.272 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.2109 | `attention_linear_regressor` |
| `align_glucosidase_expression` | mae | 0.548 | `attention_linear_regressor` |
| `align_glucosidase_melting` | mae | 0.6779 | `attention_mlp_regressor` |
| `binding` | f1 | 0.8069 | `xgb_classification` |
| `binding_site` | f1 | 0.4417 | `linear_classification` |
| `cleeveland_galox` | mae | 0.0684 | `attention_mlp_regressor` |
| `ddg` | mae | 0.6355 | `mlp_regressor` |
| `dtm` | mae | 0.3985 | `xgb_regression` |
| `ec_number` | f1 | 0.9513 | `linear_classification` |
| `erickson_petase` | mae | 1.994 | `mlp_regressor` |
| `kcat` | mae | 0.649 | `xgb_regression` |
| `km` | mae | 0.624 | `xgb_regression` |
| `ph_opt` | mae | 0.5182 | `attention_mlp_regressor` |
| `solubility` | f1 | 0.7828 | `attention_mlp_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.8571 | `mlp_classification` |
| `binding` | f1 | 0.8292 | `xgb_classification` |
| `binding_site` | f1 | 0.6862 | `mlp_classification` |
| `ddg` | mae | 0.5385 | `xgb_regression` |
| `dtm` | mae | 0.6248 | `xgb_regression` |
| `ec_number` | f1 | 0.6502 | `attention_linear_classification` |
| `kcat` | mae | 0.5819 | `attention_linear_regressor` |
| `km` | mae | 0.6518 | `xgb_regression` |
| `ph_opt` | mae | 0.6912 | `xgb_regression` |
| `solubility` | f1 | 0.6907 | `attention_mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

Prefer over the full ProtT5-XL when memory/throughput matters; expect essentially the same embedding quality. Useful for large-scale embedding of the protein pool.

## Caveats

fp16 can introduce tiny numerical differences vs full precision; still a single-sequence model with the older ProtTrans recipe.

## References
- Elnaggar, A. et al. (2022). *ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning*. IEEE TPAMI 44(10):7112–7127. <https://doi.org/10.1109/TPAMI.2021.3095381>
