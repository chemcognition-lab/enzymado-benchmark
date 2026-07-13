---
id: baseline_attention_linear
type: baseline
title: "Attention-pooled linear head"
embedding_mode: attention
model_family: linear
tags: [baseline, prediction-head, attention, linear]
aliases: ["attention_linear"]
---

# Attention-pooled linear head — baseline / prediction-head card

> Replaces fixed mean pooling with a single-head cross-attention pooler, then applies a linear head — tests whether learned pooling over residues beats the mean for a linear predictor.

- **Embedding mode:** `attention` (learned attention pooling)
- **Model family:** `linear`

## What it is

`AttentionAggregation` pools residues with `nn.MultiheadAttention` (1 head): the query is the masked mean of the residues and the keys/values are the residue embeddings, producing a learned weighted sum. The pooled vector (then concatenated with ligand/extra features) goes to a linear head. For per-residue tasks it instead applies residue self-attention (LayerNorm + residual, no pooling) followed by a per-residue linear head.

- **Loss:** Same as the linear head — MSE / BCE-with-logits / cross-entropy; EC multi-head BCE; residue BCE.
- **Pooling of residue embeddings:** Learned single-head cross-attention pooling (query = masked mean, keys/values = residues); `attention_dropout = 0.0`.

## Hyperparameters (auto-generated from `results_card.json`)

These are the exact parameters recorded for the runs of this head. Structured
tasks (EC multi-label, per-residue) use their own variants.

**`attention_linear_regressor`** — standard (global) variant _(e.g. align_amylase_activity)_

```json
{
  "hidden_dims": [],
  "lr": 0.001,
  "epochs": 200,
  "optimizer_name": "adamw",
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

**`attention_linear_classification`** — EC multi-label (multi-head) variant _(e.g. binding)_

```json
{
  "hidden_dims": [],
  "lr": 0.001,
  "epochs": 200,
  "optimizer_name": "adamw",
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

Attention runs under bf16 autocast; per-residue embeddings are streamed via a lazy DataLoader (batch size 16). Head hyperparameters match the fixed-pool linear head (lr 1e-3/1e-2, up to 100–200 epochs, AdamW, early stopping).

## Documentation references

- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

## Benchmark standing (auto — mean/best test score across the 14 models)

### Random split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `align_amylase_activity` | mae | 0.5074 | 0.4132 | 14 |
| `align_amylase_expression` | mae | 0.7452 | 0.568 | 14 |
| `align_amylase_stability` | mae | 0.2369 | 0.117 | 12 |
| `align_glucosidase_activity` | mae | 0.3227 | 0.2066 | 12 |
| `align_glucosidase_expression` | mae | 0.5498 | 0.5473 | 12 |
| `align_glucosidase_melting` | mae | 0.7895 | 0.6682 | 12 |
| `binding` | f1 | 0.5439 | 0.5693 | 14 |
| `cleeveland_galox` | mae | 0.4207 | 0.1055 | 12 |
| `ddg` | mae | 0.8881 | 0.8249 | 14 |
| `dtm` | mae | 0.5286 | 0.4675 | 14 |
| `ec_number` | f1 | 0.9443 | 0.9544 | 2 |
| `erickson_petase` | mae | 2.3171 | 2.0698 | 12 |
| `kcat` | mae | 0.7827 | 0.707 | 14 |
| `km` | mae | 0.7396 | 0.6761 | 14 |
| `ph_opt` | mae | 0.641 | 0.5659 | 14 |
| `solubility` | f1 | 0.673 | 0.7613 | 14 |

### Similarity split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `binding` | f1 | 0.5254 | 0.5798 | 14 |
| `ddg` | mae | 0.5424 | 0.4925 | 14 |
| `dtm` | mae | 0.7037 | 0.6236 | 14 |
| `ec_number` | f1 | 0.5536 | 0.796 | 10 |
| `kcat` | mae | 0.6477 | 0.5819 | 14 |
| `km` | mae | 0.7112 | 0.6796 | 14 |
| `ph_opt` | mae | 0.7894 | 0.6815 | 14 |
| `solubility` | f1 | 0.6175 | 0.6932 | 14 |

## When to use

When you suspect a few residues (e.g. active-site or binding residues) carry the signal and mean pooling dilutes them, but you still want a linear final map for regularization on smaller datasets.

## Caveats

More parameters and compute than the fixed-pool linear head; the benefit over mean pooling is task-dependent and sometimes negligible. No XGBoost analog exists for the attention track.

## References
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017. <https://arxiv.org/abs/1706.03762>
