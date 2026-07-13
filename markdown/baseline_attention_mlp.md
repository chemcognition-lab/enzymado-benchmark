---
id: baseline_attention_mlp
type: baseline
title: "Attention-pooled MLP head"
embedding_mode: attention
model_family: mlp
tags: [baseline, prediction-head, attention, mlp]
aliases: ["attention_mlp"]
---

# Attention-pooled MLP head — baseline / prediction-head card

> Single-head cross-attention pooling followed by a 2048-unit SiLU MLP — the most expressive baseline, combining learned residue pooling with a nonlinear head.

- **Embedding mode:** `attention` (learned attention pooling)
- **Model family:** `mlp`

## What it is

`AttentionAggregation` (1-head `nn.MultiheadAttention`, query = masked mean, keys/values = residue embeddings) produces a learned pooled vector, which is concatenated with any ligand/extra features and passed to an MLP (`Linear → SiLU → Dropout` → `Linear`, hidden `(2048,)`; `(256,)` for residue tasks). For per-residue tasks it applies residue self-attention then a per-residue MLP head.

- **Loss:** MSE (regression) / BCE-with-logits (binary) / cross-entropy (multiclass); EC multi-head BCE; residue BCE.
- **Pooling of residue embeddings:** Learned single-head cross-attention pooling (query = masked mean, keys/values = residues); `attention_dropout = 0.0`.

## Hyperparameters (auto-generated from `results_card.json`)

These are the exact parameters recorded for the runs of this head. Structured
tasks (EC multi-label, per-residue) use their own variants.

**`attention_mlp_classification`** — standard (global) variant _(e.g. binding)_

```json
{
  "hidden_dims": [
    2048
  ],
  "lr": 0.001,
  "epochs": 100,
  "optimizer_name": "adamw",
  "weight_decay": 0.1,
  "dropout": 0.1,
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

**`attention_mlp_regressor`** — standard (global) variant _(e.g. align_amylase_activity)_

```json
{
  "hidden_dims": [
    2048
  ],
  "lr": 0.001,
  "epochs": 100,
  "optimizer_name": "adamw",
  "weight_decay": 0.1,
  "dropout": 0.1,
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

**`attention_mlp_classification`** — EC multi-label (multi-head) variant _(e.g. ec_number)_

```json
{
  "hidden_dims": [
    2048
  ],
  "lr": 0.001,
  "epochs": 100,
  "optimizer_name": "adamw",
  "weight_decay": 0.1,
  "dropout": 0.1,
  "batch_size": 256,
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

AdamW (`weight_decay=0.1`), `dropout=0.1`, `lr=1e-3`, up to 100 epochs with early stopping; bf16 autocast; residue embeddings streamed via a lazy DataLoader (batch size 16).

## Documentation references

- [PyTorch nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)

## Benchmark standing (auto — mean/best test score across the 14 models)

### Random split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `align_amylase_activity` | mae | 0.5748 | 0.5104 | 14 |
| `align_amylase_expression` | mae | 0.8673 | 0.6733 | 14 |
| `align_amylase_stability` | mae | 0.3111 | 0.1604 | 14 |
| `align_glucosidase_activity` | mae | 0.2113 | 0.2067 | 12 |
| `align_glucosidase_expression` | mae | 0.5499 | 0.5481 | 12 |
| `align_glucosidase_melting` | mae | 0.8231 | 0.6763 | 12 |
| `binding` | f1 | 0.3209 | 0.7583 | 14 |
| `cleeveland_galox` | mae | 0.473 | 0.0684 | 12 |
| `ddg` | mae | 0.7962 | 0.6653 | 14 |
| `dtm` | mae | 1.824 | 0.4029 | 14 |
| `ec_number` | f1 | 0.8363 | 0.8848 | 4 |
| `erickson_petase` | mae | 2.3228 | 1.9512 | 12 |
| `kcat` | mae | 0.7065 | 0.6809 | 14 |
| `km` | mae | 0.6696 | 0.6445 | 14 |
| `ph_opt` | mae | 0.584 | 0.5182 | 14 |
| `solubility` | f1 | 0.683 | 0.7828 | 14 |

### Similarity split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `binding` | f1 | 0.4422 | 0.7815 | 14 |
| `ddg` | mae | 0.5487 | 0.4721 | 14 |
| `dtm` | mae | 2.4647 | 0.626 | 14 |
| `ec_number` | f1 | 0.5053 | 0.6515 | 5 |
| `kcat` | mae | 0.6332 | 0.5959 | 14 |
| `km` | mae | 0.693 | 0.6752 | 14 |
| `ph_opt` | mae | 0.7494 | 0.6748 | 14 |
| `solubility` | f1 | 0.6633 | 0.7224 | 14 |

## When to use

When both learned pooling and a nonlinear head are expected to help — e.g. residue-localized signals combined with feature interactions, and enough data to fit the extra capacity.

## Caveats

The highest-capacity, most compute-heavy baseline; easiest to overfit on small datasets, and gains over the no-attention MLP are inconsistent across tasks.

## References
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017. <https://arxiv.org/abs/1706.03762>- Elfwing, S., Uchibe, E. & Doya, K. (2018). *Sigmoid-weighted linear units (SiLU)*. Neural Networks 107:3–11. <https://doi.org/10.1016/j.neunet.2017.12.012>
