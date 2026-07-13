---
id: model_esm2_3b
type: foundation_model
title: "ESM-2 (3B)"
family: "ESM-2"
embedding_dim: 2560
tags: [foundation-model, protein-language-model, esm-2]
aliases: ["esm2_3B"]
---

# ESM-2 (3B) — foundation model card

> The 3B-parameter ESM-2 variant — larger capacity than the 650M model, with 2560-dim embeddings that often improve structure- and function-related tasks at higher compute cost.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `esm2_3B` |
| Family | ESM-2 |
| Developer | Meta AI (FAIR) |
| Parameters | 3B (36 transformer layers) |
| Embedding dimension | **2560** |
| Context length | ~1024 residues (rotary positional embeddings; long sequences are truncated in this benchmark) |
| HuggingFace | `facebook/esm2_t36_3B_UR50D` |

## Architecture & pretraining

Encoder-only BERT-style transformer (36 layers, 2560 hidden) with rotary position embeddings. Per-residue final hidden states are pooled by this benchmark.

- **Pretraining data:** UniRef50 representative sequences, sampled from UniRef90 during training.
- **Pretraining objective:** Masked language modeling (~15% token masking).

**Key ideas:**
- Scaling ESM-2 to 3B improves contact/structure recovery over 650M
- Same MLM recipe as the family; larger hidden size (2560-dim embeddings)

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/esm2_3B_shards`.
Per-residue representations are the model's `last_hidden_state` with special
tokens stripped and non-standard residues (`U`, `Z`, `O`) remapped to `X`; each
residue vector has dimension **2560**. No-attention baselines mean-pool the
residues into one vector; attention baselines learn the pooling instead (see the
baseline cards).

## Benchmark standing (auto-generated — refreshes when results change)

Mean rank among the 14 foundation models (1 = best), averaged over
heads and tasks:

| Track | Mean rank (of 14) | Groups averaged |
|---|---|---|
| No-attention · Random | 6.27 | 51 |
| No-attention · Similarity | 6.41 | 27 |
| Attention · Random | 9.26 | 19 |
| Attention · Similarity | 8.14 | 14 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.9881 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2026 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1081 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2538 | `xgb_regression` |
| `align_glucosidase_activity` | mae | 102.9854 | `linear_regressor` |
| `align_glucosidase_expression` | mae | 0.5824 | `linear_regressor` |
| `align_glucosidase_melting` | mae | 0.7016 | `mlp_regressor` |
| `binding` | f1 | 0.8102 | `xgb_classification` |
| `binding_site` | f1 | 0.5214 | `linear_classification` |
| `cleeveland_galox` | mae | 0.0595 | `xgb_regression` |
| `ddg` | mae | 0.6614 | `mlp_regressor` |
| `dtm` | mae | 0.436 | `mlp_regressor` |
| `ec_number` | f1 | 0.9729 | `linear_classification` |
| `erickson_petase` | mae | 1.9143 | `xgb_regression` |
| `kcat` | mae | 0.6545 | `xgb_regression` |
| `km` | mae | 0.6189 | `xgb_regression` |
| `ph_opt` | mae | 0.5281 | `xgb_regression` |
| `solubility` | f1 | 0.7602 | `xgb_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.89 | `mlp_classification` |
| `binding` | f1 | 0.8134 | `xgb_classification` |
| `binding_site` | f1 | 0.4964 | `linear_classification` |
| `ddg` | mae | 0.5479 | `attention_mlp_regressor` |
| `dtm` | mae | 0.652 | `attention_mlp_regressor` |
| `ec_number` | f1 | 0.8185 | `linear_classification` |
| `kcat` | mae | 0.609 | `xgb_regression` |
| `km` | mae | 0.6688 | `xgb_regression` |
| `ph_opt` | mae | 0.6826 | `mlp_regressor` |
| `solubility` | f1 | 0.7224 | `attention_mlp_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When extra representational capacity helps and the ~4× cost over the 650M model is acceptable; a strong upper-mid baseline in the ESM-2 line.

## Caveats

Substantially heavier than 650M for often-marginal gains on easy tasks; single-sequence only; long sequences truncated.

## References
- Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science 379:1123–1130. <https://doi.org/10.1126/science.ade2574>
