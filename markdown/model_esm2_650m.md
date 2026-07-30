---
id: model_esm2_650m
type: foundation_model
title: "ESM-2 (650M)"
family: "ESM-2"
embedding_dim: 1280
tags: [foundation-model, protein-language-model, esm-2]
aliases: ["esm2_650M"]
---

# ESM-2 (650M) — foundation model card

> Meta's 650M-parameter masked protein language model — the most widely used general-purpose PLM and a strong, well-validated baseline for embedding quality.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `esm2_650M` |
| Family | ESM-2 |
| Developer | Meta AI (FAIR) |
| Parameters | 650M (33 transformer layers) |
| Embedding dimension | **1280** |
| Context length | ~1024 residues (rotary positional embeddings; long sequences are truncated in this benchmark) |
| HuggingFace | `facebook/esm2_t33_650M_UR50D` |

## Architecture & pretraining

Encoder-only BERT-style transformer (33 layers, 1280 hidden, 20 attention heads) with rotary position embeddings. Per-residue outputs are the final hidden states; this benchmark mean- or attention-pools them.

- **Pretraining data:** UniRef50 representative sequences (~65M clusters), sampled during training from the broader UniRef90.
- **Pretraining objective:** Masked language modeling (BERT-style, ~15% token masking).

**Key ideas:**
- Evolutionary-scale masked LM whose attention maps recover residue–residue contacts
- Backbone of ESMFold for single-sequence structure prediction
- Family spans 8M–15B params; 650M is the standard quality/cost sweet spot

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/esm2_650M_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1280**. No-attention baselines mean-pool the
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
| No-attention · Random | 0.498 | 7.53 | 51 |
| No-attention · Similarity | 0.492 | 7.54 | 28 |
| Attention · Random | 0.365 | 8.52 | 31 |
| Attention · Similarity | 0.336 | 9.16 | 16 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9875 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2083 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1193 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2377 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.2096 | `attention_mlp_regressor` |
| `align_glucosidase_expression` | mae | 0.5083 | `xgb_regression` |
| `align_glucosidase_melting` | mae | 0.6813 | `attention_mlp_regressor` |
| `binding` | f1 | 0.8064 | `xgb_classification` |
| `binding_site` | f1 | 0.4625 | `linear_classification` |
| `cleeveland_galox` | mae | 0.0562 | `xgb_regression` |
| `ddg` | mae | 0.6287 | `mlp_regressor` |
| `dtm` | mae | 0.4376 | `xgb_regression` |
| `ec_number` | f1 | 0.9638 | `linear_classification` |
| `erickson_petase` | mae | 2.1007 | `mlp_regressor` |
| `kcat` | mae | 0.676 | `xgb_regression` |
| `km` | mae | 0.6191 | `xgb_regression` |
| `ph_opt` | mae | 0.5309 | `xgb_regression` |
| `solubility` | f1 | 0.7438 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.8794 | `mlp_classification` |
| `binding` | f1 | 0.8213 | `xgb_classification` |
| `binding_site` | f1 | 0.6763 | `mlp_classification` |
| `ddg` | mae | 0.5299 | `mlp_regressor` |
| `dtm` | mae | 0.7045 | `xgb_regression` |
| `ec_number` | f1 | 0.704 | `linear_classification` |
| `kcat` | mae | 0.613 | `xgb_regression` |
| `km` | mae | 0.6504 | `xgb_regression` |
| `ph_opt` | mae | 0.7104 | `linear_regressor` |
| `solubility` | f1 | 0.6667 | `attention_mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

A reliable default and comparison anchor: strong single-sequence embeddings at moderate cost, extensively benchmarked across function and structure tasks.

## Caveats

Single-sequence only (no MSA/retrieval), no explicit structure or function conditioning; sequences beyond the training context are truncated.

## References
- Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science 379:1123–1130. <https://doi.org/10.1126/science.ade2574>
