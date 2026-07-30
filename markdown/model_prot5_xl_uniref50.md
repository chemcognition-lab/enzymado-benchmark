---
id: model_prot5_xl_uniref50
type: foundation_model
title: "ProtT5-XL-UniRef50"
family: "ProtT5 (ProtTrans)"
embedding_dim: 1024
tags: [foundation-model, protein-language-model, prott5-(prottrans)]
aliases: ["prot5_xl_uniref50"]
---

# ProtT5-XL-UniRef50 — foundation model card

> A T5-3B protein language model from ProtTrans whose 1024-dim encoder embeddings are a long-standing strong baseline for per-residue and per-protein tasks.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `prot5_xl_uniref50` |
| Family | ProtT5 (ProtTrans) |
| Developer | Rostlab (TU Munich) et al. |
| Parameters | ~3B (T5-3B encoder–decoder; encoder used for embeddings) |
| Embedding dimension | **1024** |
| Context length | Effectively unbounded via T5 relative position bias (memory-limited; benchmark truncates very long sequences) |
| HuggingFace | `Rostlab/prot_t5_xl_uniref50` |

## Architecture & pretraining

T5 encoder–decoder (T5-3B) with relative position bias; only the encoder is used to produce 1024-dim per-residue embeddings, mean/attention-pooled here.

- **Pretraining data:** Pretrained on BFD (Big Fantastic Database) then fine-tuned on UniRef50 (~45M sequences).
- **Pretraining objective:** Span-corruption / masked span reconstruction (T5-style denoising, adapted for proteins).

**Key ideas:**
- Encoder embeddings capture secondary structure, localization and membrane signal without supervision
- Established that large PLMs rival evolutionary (MSA) features for many tasks

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/prot5_xl_uniref50_shards`.
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
| No-attention · Random | 0.590 | 6.33 | 52 |
| No-attention · Similarity | 0.548 | 6.79 | 28 |
| Attention · Random | 0.637 | 4.94 | 31 |
| Attention · Similarity | 0.752 | 4.13 | 15 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9857 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2309 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1402 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2312 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.213 | `attention_linear_regressor` |
| `align_glucosidase_expression` | mae | 0.5503 | `attention_linear_regressor` |
| `align_glucosidase_melting` | mae | 0.6244 | `xgb_regression` |
| `binding` | f1 | 0.8133 | `xgb_classification` |
| `binding_site` | f1 | 0.941 | `mlp_classification` |
| `cleeveland_galox` | mae | 0.1171 | `attention_mlp_regressor` |
| `ddg` | mae | 0.6496 | `mlp_regressor` |
| `dtm` | mae | 0.4029 | `attention_mlp_regressor` |
| `ec_number` | f1 | 0.9514 | `linear_classification` |
| `erickson_petase` | mae | 1.6181 | `linear_regressor` |
| `kcat` | mae | 0.6492 | `xgb_regression` |
| `km` | mae | 0.6184 | `xgb_regression` |
| `ph_opt` | mae | 0.533 | `xgb_regression` |
| `solubility` | f1 | 0.7754 | `attention_mlp_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.8422 | `mlp_classification` |
| `binding` | f1 | 0.8239 | `xgb_classification` |
| `binding_site` | f1 | 0.6757 | `mlp_classification` |
| `ddg` | mae | 0.5214 | `mlp_regressor` |
| `dtm` | mae | 0.615 | `linear_regressor` |
| `ec_number` | f1 | 0.6305 | `mlp_classification` |
| `kcat` | mae | 0.5906 | `attention_linear_regressor` |
| `km` | mae | 0.648 | `xgb_regression` |
| `ph_opt` | mae | 0.699 | `attention_mlp_regressor` |
| `solubility` | f1 | 0.7059 | `attention_mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

A dependable, well-studied 1024-dim baseline; frequently competitive on the enzyme regression tasks in this benchmark. Encoder-only inference keeps it practical.

## Caveats

Large model; the full checkpoint stores decoder weights that are unused for embedding. Single-sequence; older recipe than ESM-2/ESM-C.

## References
- Elnaggar, A. et al. (2022). *ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning*. IEEE TPAMI 44(10):7112–7127. <https://doi.org/10.1109/TPAMI.2021.3095381>
