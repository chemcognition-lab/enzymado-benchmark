---
id: model_prost5
type: foundation_model
title: "ProstT5"
family: "ProstT5 (ProtT5-based)"
embedding_dim: 1024
tags: [foundation-model, protein-language-model, prostt5-(prott5-based)]
aliases: ["prost5"]
---

# ProstT5 — foundation model card

> A bilingual sequence-and-structure T5 that fine-tunes ProtT5-XL on translating between amino acids and Foldseek 3Di structure tokens — its 1024-dim encoder embeddings carry structure-aware signal.

## At a glance

| Field | Value |
|---|---|
| Benchmark model key | `prost5` |
| Family | ProstT5 (ProtT5-based) |
| Developer | Rostlab (Heinzinger et al.) |
| Parameters | ~3B (T5-3B; fine-tuned from ProtT5-XL-U50) |
| Embedding dimension | **1024** |
| Context length | T5 relative position bias (memory-limited; long sequences truncated in this benchmark) |
| HuggingFace | `Rostlab/ProstT5` |

## Architecture & pretraining

T5 encoder–decoder initialized from ProtT5-XL-UniRef50 and trained to translate between amino-acid sequences and 3Di structural tokens; the encoder yields 1024-dim per-residue embeddings used here.

- **Pretraining data:** ~17M proteins with high-quality predicted structures from the AlphaFold Protein Structure Database (3Di tokens via Foldseek).
- **Pretraining objective:** Bidirectional translation between amino-acid and 3Di (Foldseek) alphabets ('folding' and 'inverse folding'), on top of the ProtT5 base.

**Key ideas:**
- Injects structural information into a sequence PLM via the 3Di structural alphabet
- Enables fast structure-aware remote-homology search and structure/inverse-folding generation

## How its embeddings are used in this benchmark

Embeddings are precomputed once over the deduplicated protein pool and stored as
bfloat16 safetensors shards (1000 proteins each) under `data/embeddings/protein_sequences_unique/prost5_shards`.
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
| No-attention · Random | 8.02 | 51 |
| No-attention · Similarity | 7.34 | 28 |
| Attention · Random | 6.6 | 30 |
| Attention · Similarity | 7.06 | 16 |

### Best test score per task — random split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.988 | `mlp_classification` |
| `align_amylase_activity` | mae | 0.2406 | `xgb_regression` |
| `align_amylase_expression` | mae | 0.1316 | `xgb_regression` |
| `align_amylase_stability` | mae | 0.2648 | `attention_linear_regressor` |
| `align_glucosidase_activity` | mae | 0.2087 | `attention_linear_regressor` |
| `align_glucosidase_expression` | mae | 0.549 | `attention_mlp_regressor` |
| `align_glucosidase_melting` | mae | 0.7219 | `attention_mlp_regressor` |
| `binding` | f1 | 0.8184 | `xgb_classification` |
| `binding_site` | f1 | 0.4272 | `linear_classification` |
| `cleeveland_galox` | mae | 0.1084 | `attention_mlp_regressor` |
| `ddg` | mae | 0.6688 | `mlp_regressor` |
| `dtm` | mae | 0.4374 | `attention_mlp_regressor` |
| `ec_number` | f1 | 0.9584 | `linear_classification` |
| `erickson_petase` | mae | 2.0003 | `mlp_regressor` |
| `kcat` | mae | 0.6489 | `xgb_regression` |
| `km` | mae | 0.6113 | `xgb_regression` |
| `ph_opt` | mae | 0.5331 | `xgb_regression` |
| `solubility` | f1 | 0.7064 | `mlp_classification` |

### Best test score per task — similarity split

| Task | Primary metric | Best score | Winning head |
|---|---|---|---|
| `active_sites` | f1 | 0.8858 | `mlp_classification` |
| `binding` | f1 | 0.8045 | `xgb_classification` |
| `binding_site` | f1 | 0.7028 | `mlp_classification` |
| `ddg` | mae | 0.5069 | `xgb_regression` |
| `dtm` | mae | 0.6474 | `mlp_regressor` |
| `ec_number` | f1 | 0.7859 | `linear_classification` |
| `kcat` | mae | 0.5927 | `attention_linear_regressor` |
| `km` | mae | 0.6495 | `xgb_regression` |
| `ph_opt` | mae | 0.7006 | `attention_mlp_regressor` |
| `solubility` | f1 | 0.6333 | `xgb_classification` |

> Scores are point estimates. Paired-bootstrap CIs and Bradley-Terry Elo ratings
> are produced on the cluster (`compute_bootstrap_metrics.py` →
> `compute_ratings.py`); regenerate this card once those artifacts are available
> locally.

## When to use

When structure-aware sequence representations may help (e.g. binding/active-site or fold-dependent properties) while keeping sequence-only inference.

## Caveats

Structure signal is distilled from predicted (AlphaFold) structures, not experimental; for embedding we use only the amino-acid encoder path.

## References
- Heinzinger, M. et al. (2024). *Bilingual language model for protein sequence and structure (ProstT5)*. NAR Genomics and Bioinformatics 6(4):lqae150. <https://doi.org/10.1093/nargab/lqae150>
