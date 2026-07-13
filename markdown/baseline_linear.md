---
id: baseline_linear
type: baseline
title: "Linear head (no attention)"
embedding_mode: no_attention
model_family: linear
tags: [baseline, prediction-head, no-attention, linear]
aliases: ["linear"]
---

# Linear head (no attention) — baseline / prediction-head card

> A single linear layer on top of the mean-pooled protein embedding — the simplest baseline, probing how linearly predictive each foundation model's representation already is.

- **Embedding mode:** `no_attention` (fixed masked mean pooling)
- **Model family:** `linear`

## What it is

Residue embeddings are collapsed to one vector by a fixed masked mean, then a single `nn.Linear` maps that vector (optionally concatenated with ligand or mutation-delta features) to the target. It is implemented in PyTorch rather than scikit-learn so it shares the same training loop, feature standardization, and early stopping as the MLP head; conceptually it is ordinary least squares (regression) or logistic regression (classification).

- **Loss:** MSE for regression; binary cross-entropy with logits for binary classification; cross-entropy for multiclass. EC uses masked multi-head BCE; per-residue tasks use per-residue BCE.
- **Pooling of residue embeddings:** Fixed masked mean over residues (no learned pooling).

## Hyperparameters (auto-generated from `results_card.json`)

These are the exact parameters recorded for the runs of this head. Structured
tasks (EC multi-label, per-residue) use their own variants.

**`linear_classification`** — standard (global) variant _(e.g. binding)_

```json
{
  "lr": 0.001,
  "epochs": 200,
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

**`linear_regressor`** — standard (global) variant _(e.g. align_amylase_activity)_

```json
{
  "lr": 0.001,
  "epochs": 200,
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

**`linear_classification`** — EC multi-label (multi-head) variant _(e.g. ec_number)_

```json
{
  "hidden_dims": [],
  "lr": 0.01,
  "epochs": 200,
  "optimizer_name": "adamw",
  "early_stopping_patience": 20,
  "early_stopping_min_delta": 0.0001
}
```

**`linear_classification`** — per-residue variant _(e.g. active_sites)_

```json
{
  "hidden_dims": [],
  "lr": 0.01,
  "epochs": 30,
  "optimizer_name": "adamw",
  "early_stopping_patience": 5,
  "early_stopping_min_delta": 0.001
}
```

Per-feature embedding standardization is fit on the training split only. The optimizer is AdamW; training runs up to `epochs` with early stopping on validation loss (patience shown per variant). `hidden_dims: []` marks the pure-linear configuration; the EC and residue variants raise the learning rate and shorten training.

## Documentation references

- [scikit-learn — Linear models (conceptual reference)](https://scikit-learn.org/stable/modules/linear_model.html)
- [PyTorch nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

## Benchmark standing (auto — mean/best test score across the 14 models)

### Random split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `active_sites` | f1 | 0.6513 | 0.8069 | 14 |
| `align_amylase_activity` | mae | 0.9213 | 0.8617 | 14 |
| `align_amylase_expression` | mae | 0.4739 | 0.4457 | 14 |
| `align_amylase_stability` | mae | 3.9338 | 3.1684 | 14 |
| `align_glucosidase_activity` | mae | 97.4913 | 92.0788 | 14 |
| `align_glucosidase_expression` | mae | 0.6621 | 0.575 | 14 |
| `align_glucosidase_melting` | mae | 0.9552 | 0.7563 | 14 |
| `binding` | f1 | 0.5181 | 0.5287 | 14 |
| `binding_site` | f1 | 0.428 | 0.5214 | 14 |
| `cleeveland_galox` | mae | 0.8158 | 0.1321 | 14 |
| `ddg` | mae | 0.7058 | 0.6379 | 14 |
| `dtm` | mae | 0.4573 | 0.4062 | 14 |
| `ec_number` | f1 | 0.9445 | 0.9729 | 14 |
| `erickson_petase` | mae | 2.4077 | 1.6181 | 14 |
| `kcat` | mae | 0.7072 | 0.6825 | 14 |
| `km` | mae | 0.6916 | 0.6781 | 14 |
| `ph_opt` | mae | 0.571 | 0.5542 | 14 |
| `solubility` | f1 | 0.6982 | 0.7422 | 14 |

### Similarity split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `active_sites` | f1 | 0.4483 | 0.5884 | 14 |
| `binding` | f1 | 0.5316 | 0.5477 | 14 |
| `binding_site` | f1 | 0.4226 | 0.4964 | 14 |
| `ddg` | mae | 0.5822 | 0.5128 | 14 |
| `dtm` | mae | 0.7092 | 0.5962 | 14 |
| `ec_number` | f1 | 0.6469 | 0.8185 | 14 |
| `kcat` | mae | 0.6307 | 0.6016 | 14 |
| `km` | mae | 0.7062 | 0.6719 | 14 |
| `ph_opt` | mae | 0.726 | 0.6882 | 14 |
| `solubility` | f1 | 0.6599 | 0.7134 | 14 |

## When to use

Use as a capacity-controlled probe of embedding quality and as a fast, low-variance baseline — especially on the small real_case datasets where MLP/XGBoost overfit. Strong linear performance means the foundation model already encodes the property near-linearly.

## Caveats

Cannot model feature interactions, so it underperforms on tasks needing nonlinearity. Being a torch reimplementation, results can differ slightly from scikit-learn's solvers.

## References
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12:2825–2830. <https://jmlr.org/papers/v12/pedregosa11a.html>
