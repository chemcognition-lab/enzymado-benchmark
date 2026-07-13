---
id: baseline_mlp
type: baseline
title: "MLP head (no attention)"
embedding_mode: no_attention
model_family: mlp
tags: [baseline, prediction-head, no-attention, mlp]
aliases: ["mlp"]
---

# MLP head (no attention) — baseline / prediction-head card

> A multilayer perceptron (one hidden layer of 2048 units with SiLU) on the mean-pooled embedding — adds nonlinearity over the linear head while remaining a fixed-pooling baseline.

- **Embedding mode:** `no_attention` (fixed masked mean pooling)
- **Model family:** `mlp`

## What it is

Fixed masked mean pooling → MLP: `Linear → SiLU → Dropout` for each hidden width (default `(2048,)`), then a final `Linear`. Ligand or mutation-delta features are concatenated into the pooled vector before the MLP. For EC it uses a shared backbone with one linear head per EC level; for per-residue tasks it uses a smaller `(256,)` hidden width. Implemented in PyTorch.

- **Loss:** MSE (regression) / BCE-with-logits (binary) / cross-entropy (multiclass); EC uses masked multi-head BCE; residue tasks use per-residue BCE.
- **Pooling of residue embeddings:** Fixed masked mean over residues.

## Hyperparameters (auto-generated from `results_card.json`)

These are the exact parameters recorded for the runs of this head. Structured
tasks (EC multi-label, per-residue) use their own variants.

**`mlp_regressor`** — standard (global) variant _(e.g. align_amylase_activity)_

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

**`mlp_classification`** — EC multi-label (multi-head) variant _(e.g. binding)_

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

**`mlp_classification`** — per-residue variant _(e.g. active_sites)_

```json
{
  "hidden_dims": [
    256
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

**`mlp_classification`** — per-residue variant _(e.g. binding_site)_

```json
{
  "hidden_dims": [
    256
  ],
  "lr": 0.001,
  "epochs": 40,
  "optimizer_name": "adam",
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.001
}
```

AdamW with `weight_decay=0.1` and `dropout=0.1`, `lr=1e-3`, up to 100 epochs with early stopping (patience 20). Embedding features are standardized on train. `hidden_dims=(2048,)` for global tasks, `(256,)` for per-residue tasks.

## Documentation references

- [scikit-learn — Multi-layer Perceptron (conceptual reference)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [PyTorch nn.SiLU activation](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)

## Benchmark standing (auto — mean/best test score across the 14 models)

### Random split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `active_sites` | f1 | 0.983 | 0.9894 | 14 |
| `align_amylase_activity` | mae | 0.3439 | 0.2446 | 14 |
| `align_amylase_expression` | mae | 0.2772 | 0.2151 | 14 |
| `align_amylase_stability` | mae | 0.832 | 0.5276 | 14 |
| `align_glucosidase_activity` | mae | 253.7538 | 161.2593 | 14 |
| `align_glucosidase_expression` | mae | 0.7557 | 0.6173 | 14 |
| `align_glucosidase_melting` | mae | 0.9765 | 0.7016 | 14 |
| `binding` | f1 | 0.7563 | 0.7843 | 14 |
| `binding_site` | f1 | 0.8889 | 0.941 | 4 |
| `cleeveland_galox` | mae | 3.2149 | 0.1602 | 14 |
| `ddg` | mae | 0.6578 | 0.6023 | 14 |
| `dtm` | mae | 0.4378 | 0.3909 | 14 |
| `ec_number` | f1 | 0.857 | 0.9601 | 14 |
| `erickson_petase` | mae | 2.1959 | 1.9093 | 14 |
| `kcat` | mae | 0.6982 | 0.6588 | 14 |
| `km` | mae | 0.6716 | 0.6434 | 14 |
| `ph_opt` | mae | 0.5532 | 0.5326 | 14 |
| `solubility` | f1 | 0.7378 | 0.7752 | 14 |

### Similarity split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `active_sites` | f1 | 0.8688 | 0.9194 | 14 |
| `binding` | f1 | 0.7727 | 0.7952 | 14 |
| `binding_site` | f1 | 0.6381 | 0.7028 | 9 |
| `ddg` | mae | 0.5454 | 0.4579 | 14 |
| `dtm` | mae | 0.6617 | 0.5825 | 14 |
| `ec_number` | f1 | 0.5535 | 0.7508 | 14 |
| `kcat` | mae | 0.6447 | 0.6051 | 14 |
| `km` | mae | 0.6989 | 0.6726 | 14 |
| `ph_opt` | mae | 0.7271 | 0.6826 | 14 |
| `solubility` | f1 | 0.6685 | 0.7126 | 14 |

## When to use

When the property is nonlinearly encoded and there is enough data to fit a hidden layer — a good middle ground between the linear probe and XGBoost. Prefer the attention-MLP variant when per-residue detail matters.

## Caveats

More prone to overfitting small real_case sets than the linear head, and it still discards residue-level structure because of mean pooling.

## References
- Elfwing, S., Uchibe, E. & Doya, K. (2018). *Sigmoid-weighted linear units for neural network function approximation in reinforcement learning (SiLU)*. Neural Networks 107:3–11. <https://doi.org/10.1016/j.neunet.2017.12.012>
