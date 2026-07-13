---
id: model_profluent_e1_600m
type: foundation_model
title: "Profluent E1 (600M)"
family: "Profluent E1"
embedding_dim: 1280
tags: [foundation-model, protein-language-model, profluent-e1]
aliases: ["profluent_e1_600m"]
---

# Profluent E1 (600M) — foundation model card

> The 600M Profluent E1 encoder — the larger retrieval-augmented E1 model, used single-sequence here to produce 1280-dim embeddings.
>
> ⚠️ **Needs verification:** some specifics below are best-effort for this recent / thin-source model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `profluent_e1_600m` |
| Family | Profluent E1 |
| Developer | Profluent Bio |
| Parameters | 600M |
| Embedding dimension | **1280** |
| Context length | Single-sequence in this benchmark (E1 also supports retrieval-augmented multi-sequence context) |
| HuggingFace | `Profluent-Bio/E1-600m` |

## Architecture & pretraining

Retrieval-augmented protein encoder (block-causal multi-sequence attention over retrieved homologs); run single-sequence in this benchmark for 1280-dim per-residue embeddings.

- **Pretraining data:** ≈4 trillion tokens from the Profluent Protein Atlas.
- **Pretraining objective:** Encoder pretraining with retrieval augmentation (homolog conditioning).

**Key ideas:**
- Largest released E1 encoder; retrieval-augmented evolutionary conditioning
- Reported SOTA zero-shot fitness / contact prediction; permissive license for research and commercial use

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/profluent_e1_600m_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1280**. No-attention baselines mean-pool the
residues into one vector; attention baselines learn the pooling instead (see the
baseline cards).

## Benchmark standing (auto-generated — refreshes when results change)

Mean rank among the 14 foundation models (1 = best), averaged over
heads and tasks:

| Track | Mean rank (of 14) | Groups averaged |
|---|---|---|
| No-attention · Random | 7.78 | 51 |
| No-attention · Similarity | 9.3 | 27 |
| Attention · Random | 7.23 | 30 |
| Attention · Similarity | 6.81 | 16 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9864 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.1967 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.116 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2418 | `xgb_regression` |
| `align_glucosidase_activity` | mae | 0.2069 | `attention_mlp_regressor` |
| `align_glucosidase_expression` | mae | 0.5501 | `attention_mlp_regressor` |
| `align_glucosidase_melting` | mae | 0.6763 | `attention_mlp_regressor` |
| `binding` | f1 | 0.807 | `xgb_classification` |
| `binding_site` | f1 | 0.4279 | `linear_classification` |
| `cleeveland_galox` | mae | 0.1178 | `xgb_regression` |
| `ddg` | mae | 0.6941 | `attention_mlp_regressor` |
| `dtm` | mae | 0.4357 | `xgb_regression` |
| `ec_number` | f1 | 0.9252 | `linear_classification` |
| `erickson_petase` | mae | 2.1145 | `attention_linear_regressor` |
| `kcat` | mae | 0.6571 | `xgb_regression` |
| `km` | mae | 0.6341 | `xgb_regression` |
| `ph_opt` | mae | 0.5602 | `xgb_regression` |
| `solubility` | f1 | 0.7613 | `mlp_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9006 | `mlp_classification` |
| `binding` | f1 | 0.8161 | `xgb_classification` |
| `binding_site` | f1 | 0.4345 | `linear_classification` |
| `ddg` | mae | 0.5574 | `attention_mlp_regressor` |
| `dtm` | mae | 0.6581 | `attention_linear_regressor` |
| `ec_number` | f1 | 0.5691 | `attention_linear_classification` |
| `kcat` | mae | 0.6086 | `linear_regressor` |
| `km` | mae | 0.6694 | `xgb_regression` |
| `ph_opt` | mae | 0.7094 | `linear_regressor` |
| `solubility` | f1 | 0.7126 | `mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

As a larger modern ESM-style encoder (1280-dim); again, its retrieval advantage is not used in this single-sequence embedding setup.

## Caveats

Very recent (2025 preprint); benefit is partly tied to retrieval context absent here. Verify per-size specifics before quoting.

## References
- Profluent Bio (2025). *E1: Retrieval-Augmented Protein Encoder Models*. bioRxiv 2025.11.12.688125. <https://doi.org/10.1101/2025.11.12.688125>
