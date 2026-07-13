---
id: model_profluent_e1_300m
type: foundation_model
title: "Profluent E1 (300M)"
family: "Profluent E1"
embedding_dim: 1024
tags: [foundation-model, protein-language-model, profluent-e1]
aliases: ["profluent_e1_300m"]
---

# Profluent E1 (300M) — foundation model card

> The 300M Profluent E1 encoder — a retrieval-augmented protein LM positioned as a drop-in ESM replacement. Here it is used single-sequence, producing 1024-dim embeddings.
>
> ⚠️ **Needs verification:** some specifics below are best-effort for this recent / thin-source model.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `profluent_e1_300m` |
| Family | Profluent E1 |
| Developer | Profluent Bio |
| Parameters | 300M |
| Embedding dimension | **1024** |
| Context length | Single-sequence in this benchmark (E1 also supports retrieval-augmented multi-sequence context) |
| HuggingFace | `Profluent-Bio/E1-300m` |

## Architecture & pretraining

Protein encoder that can condition on retrieved homologous sequences via block-causal multi-sequence attention; in this benchmark it is run single-sequence to produce 1024-dim per-residue embeddings.

- **Pretraining data:** ≈4 trillion tokens from the Profluent Protein Atlas.
- **Pretraining objective:** Encoder pretraining with retrieval augmentation (conditioning on homologous sequences).

**Key ideas:**
- Explicit retrieval augmentation: integrates evolutionary context at inference via multi-sequence attention
- Reported state-of-the-art zero-shot fitness and unsupervised contact prediction; drop-in ESM replacement

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/profluent_e1_300m_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **1024**. No-attention baselines mean-pool the
residues into one vector; attention baselines learn the pooling instead (see the
baseline cards).

## Benchmark standing (auto-generated — refreshes when results change)

Mean rank among the 14 foundation models (1 = best), averaged over
heads and tasks:

| Track | Mean rank (of 14) | Groups averaged |
|---|---|---|
| No-attention · Random | 8.5 | 52 |
| No-attention · Similarity | 8.0 | 28 |
| Attention · Random | 8.3 | 30 |
| Attention · Similarity | 4.33 | 15 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9842 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.203 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1139 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2107 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.209 | `attention_linear_regressor` |
| `align_glucosidase_expression` | mae | 0.5225 | `xgb_regression` |
| `align_glucosidase_melting` | mae | 0.6765 | `attention_mlp_regressor` |
| `binding` | f1 | 0.8087 | `xgb_classification` |
| `binding_site` | f1 | 0.9165 | `mlp_classification` |
| `cleeveland_galox` | mae | 0.1448 | `attention_mlp_regressor` |
| `ddg` | mae | 0.668 | `mlp_regressor` |
| `dtm` | mae | 0.4133 | `xgb_regression` |
| `ec_number` | f1 | 0.9249 | `linear_classification` |
| `erickson_petase` | mae | 2.131 | `attention_linear_regressor` |
| `kcat` | mae | 0.6743 | `xgb_regression` |
| `km` | mae | 0.633 | `xgb_regression` |
| `ph_opt` | mae | 0.559 | `xgb_regression` |
| `solubility` | f1 | 0.7517 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.8648 | `mlp_classification` |
| `binding` | f1 | 0.8282 | `xgb_classification` |
| `binding_site` | f1 | 0.6482 | `mlp_classification` |
| `ddg` | mae | 0.5127 | `attention_linear_regressor` |
| `dtm` | mae | 0.6801 | `attention_linear_regressor` |
| `ec_number` | f1 | 0.5981 | `linear_classification` |
| `kcat` | mae | 0.6104 | `attention_linear_regressor` |
| `km` | mae | 0.6678 | `xgb_regression` |
| `ph_opt` | mae | 0.6748 | `attention_mlp_regressor` |
| `solubility` | f1 | 0.7051 | `linear_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

As a modern ESM-style encoder; note that its headline retrieval-augmentation advantage is not exercised here (single-sequence embeddings only).

## Caveats

Very recent (2025 preprint); much of its benefit comes from retrieval context that this benchmark does not supply. Verify per-size architecture details before quoting.

## References
- Profluent Bio (2025). *E1: Retrieval-Augmented Protein Encoder Models*. bioRxiv 2025.11.12.688125. <https://doi.org/10.1101/2025.11.12.688125>
