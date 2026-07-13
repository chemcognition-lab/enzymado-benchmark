---
id: model_esm3_open
type: foundation_model
title: "ESM3 (open, small)"
family: "ESM3"
embedding_dim: 1536
tags: [foundation-model, protein-language-model, esm3]
aliases: ["esm3_open"]
---

# ESM3 (open, small) — foundation model card

> The open 1.4B-parameter ESM3 — a multimodal generative model over sequence, structure, and function. Here it is used purely as a sequence encoder (1536-dim per-residue embeddings).

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `esm3_open` |
| Family | ESM3 |
| Developer | EvolutionaryScale |
| Parameters | 1.4B (esm3-sm-open-v1) |
| Embedding dimension | **1536** |
| Context length | Up to a few thousand residues (benchmark truncates very long sequences) |
| HuggingFace | `EvolutionaryScale/esm3-sm-open-v1` |

## Architecture & pretraining

Multimodal transformer that jointly tokenizes sequence, 3D structure, and function into discrete tracks; the smallest 'open' variant. This benchmark consumes only its per-residue sequence-track representations (1536-dim).

- **Pretraining data:** ≈2.78B natural proteins (≈3.15B with synthetic augmentation; 236M structures and 539M function-annotated proteins), ≈771B tokens.
- **Pretraining objective:** Masked generative modeling across the sequence, structure, and function tracks.

**Key ideas:**
- First natively multimodal, generative ESM (sequence + structure + function)
- Demonstrated de novo design (esmGFP), an order of magnitude larger training than ESM-2

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/esm3_open_shards`.
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
| No-attention · Random | 6.1 | 51 |
| No-attention · Similarity | 7.02 | 28 |
| Attention · Random | 9.87 | 30 |
| Attention · Similarity | 10.33 | 15 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9874 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2249 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1251 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2915 | `xgb_regression` |
| `align_glucosidase_activity` | mae | 0.2067 | `attention_mlp_regressor` |
| `align_glucosidase_expression` | mae | 0.5184 | `xgb_regression` |
| `align_glucosidase_melting` | mae | 0.7875 | `linear_regressor` |
| `binding` | f1 | 0.815 | `xgb_classification` |
| `binding_site` | f1 | 0.4107 | `linear_classification` |
| `cleeveland_galox` | mae | 0.0567 | `xgb_regression` |
| `ddg` | mae | 0.6023 | `mlp_regressor` |
| `dtm` | mae | 0.3865 | `xgb_regression` |
| `ec_number` | f1 | 0.9217 | `linear_classification` |
| `erickson_petase` | mae | 1.9794 | `linear_regressor` |
| `kcat` | mae | 0.652 | `xgb_regression` |
| `km` | mae | 0.6169 | `xgb_regression` |
| `ph_opt` | mae | 0.5159 | `xgb_regression` |
| `solubility` | f1 | 0.7228 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9194 | `mlp_classification` |
| `binding` | f1 | 0.828 | `xgb_classification` |
| `binding_site` | f1 | 0.6925 | `mlp_classification` |
| `ddg` | mae | 0.4986 | `xgb_regression` |
| `dtm` | mae | 0.6334 | `xgb_regression` |
| `ec_number` | f1 | 0.5875 | `linear_classification` |
| `kcat` | mae | 0.617 | `mlp_regressor` |
| `km` | mae | 0.6545 | `xgb_regression` |
| `ph_opt` | mae | 0.7159 | `xgb_regression` |
| `solubility` | f1 | 0.6561 | `mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When structure/function-aware pretraining might benefit a property, or to compare a generative multimodal model against masked encoders — while still using single-sequence embeddings.

## Caveats

Designed as a generative multimodal model; using it only as an embedding encoder ignores its structure/function inputs. Heavier tokenization pipeline than plain PLMs.

## References
- Hayes, T. et al. (2025). *Simulating 500 million years of evolution with a language model (ESM3)*. Science 387:850–858. <https://doi.org/10.1126/science.ads0018>
