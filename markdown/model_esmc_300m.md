---
id: model_esmc_300m
type: foundation_model
title: "ESM Cambrian (ESM-C 300M)"
family: "ESM-C"
embedding_dim: 960
tags: [foundation-model, protein-language-model, esm-c]
aliases: ["esmc_300m"]
---

# ESM Cambrian (ESM-C 300M) — foundation model card

> The 300M ESM Cambrian encoder — a representation-focused masked PLM that reportedly matches ESM-2 650M at lower cost, producing 960-dim embeddings.
>
> ⚠️ **Needs verification:** some specifics below are best-effort for this recent / thin-source model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `esmc_300m` |
| Family | ESM-C |
| Developer | EvolutionaryScale |
| Parameters | 300M |
| Embedding dimension | **960** |
| Context length | Up to ~2048 residues (benchmark truncates longer sequences) |
| HuggingFace | `EvolutionaryScale/esmc-300m-2024-12` |

## Architecture & pretraining

Encoder-only masked transformer in the ESM-C representation line (parallel to the generative ESM3); 960-dim per-residue embeddings.

- **Pretraining data:** Large-scale natural protein sequence corpora (EvolutionaryScale; see the ESM-C release).
- **Pretraining objective:** Masked language modeling.

**Key ideas:**
- Representation-optimized companion to ESM3, released as open weights
- 300M reportedly rivals ESM-2 650M with reduced memory and faster inference

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/esmc_300m_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **960**. No-attention baselines mean-pool the
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
| No-attention · Random | 0.385 | 8.81 | 52 |
| No-attention · Similarity | 0.373 | 9.00 | 28 |
| Attention · Random | 0.612 | 5.84 | 31 |
| Attention · Similarity | 0.672 | 5.03 | 15 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9676 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2167 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1269 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.117 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.2078 | `attention_linear_regressor` |
| `align_glucosidase_expression` | mae | 0.5408 | `xgb_regression` |
| `align_glucosidase_melting` | mae | 0.6745 | `attention_linear_regressor` |
| `binding` | f1 | 0.7997 | `xgb_classification` |
| `binding_site` | f1 | 0.7958 | `mlp_classification` |
| `cleeveland_galox` | mae | 0.065 | `xgb_regression` |
| `ddg` | mae | 0.6832 | `mlp_regressor` |
| `dtm` | mae | 0.4237 | `xgb_regression` |
| `ec_number` | f1 | 0.9136 | `linear_classification` |
| `erickson_petase` | mae | 1.9524 | `mlp_regressor` |
| `kcat` | mae | 0.6713 | `xgb_regression` |
| `km` | mae | 0.6404 | `xgb_regression` |
| `ph_opt` | mae | 0.5481 | `xgb_regression` |
| `solubility` | f1 | 0.7483 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.7808 | `mlp_classification` |
| `binding` | f1 | 0.8165 | `xgb_classification` |
| `binding_site` | f1 | 0.5374 | `mlp_classification` |
| `ddg` | mae | 0.5075 | `attention_mlp_regressor` |
| `dtm` | mae | 0.6164 | `xgb_regression` |
| `ec_number` | f1 | 0.4927 | `linear_classification` |
| `kcat` | mae | 0.5951 | `xgb_regression` |
| `km` | mae | 0.6673 | `xgb_regression` |
| `ph_opt` | mae | 0.6933 | `attention_mlp_regressor` |
| `solubility` | f1 | 0.6667 | `attention_mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

A cost-efficient modern encoder: good when you want ESM-2-650M-level quality at smaller size / 960-dim embeddings.

## Caveats

Primarily documented via the EvolutionaryScale release/blog rather than a peer-reviewed paper — verify exact architecture and training details before citing specifics.

## References
- EvolutionaryScale Team (2024). *ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning*. EvolutionaryScale (release note). <https://www.evolutionaryscale.ai/blog/esm-cambrian>
