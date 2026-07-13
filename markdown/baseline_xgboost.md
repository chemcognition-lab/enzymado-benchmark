---
id: baseline_xgboost
type: baseline
title: "XGBoost head (no attention)"
embedding_mode: no_attention
model_family: xgboost
tags: [baseline, prediction-head, no-attention, xgboost]
aliases: ["xgboost"]
---

# XGBoost head (no attention) — baseline / prediction-head card

> Gradient-boosted decision trees (XGBoost) on the mean-pooled protein embedding — a strong nonlinear tabular baseline that frequently wins the global regression and classification tasks.

- **Embedding mode:** `no_attention` (fixed masked mean pooling)
- **Model family:** `xgboost`

## What it is

Residue embeddings are mean-pooled to a fixed vector (optionally concatenated with the 8-dim RDKit ligand descriptor or the mutation delta), then fed to `xgboost.XGBRegressor` / `XGBClassifier`. Trees are built by second-order gradient boosting with histogram binning. EC multi-label classification trains over 256-column label blocks; per-residue tasks stream residues into a `QuantileDMatrix`. There is no attention variant for XGBoost.

- **Loss:** `reg:squarederror` for regression and `binary:logistic` for binary classification; multi-label EC uses per-column logistic objectives with masking of unknown labels.
- **Pooling of residue embeddings:** Fixed masked mean over residues.

## Hyperparameters (auto-generated from `results_card.json`)

These are the exact parameters recorded for the runs of this head. Structured
tasks (EC multi-label, per-residue) use their own variants.

**`xgb_classification`** — standard (global) variant _(e.g. binding)_

```json
{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "n_jobs": -1,
  "device": "cuda"
}
```

**`xgb_regression`** — standard (global) variant _(e.g. align_amylase_activity)_

```json
{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "n_jobs": -1,
  "device": "cuda"
}
```

**`xgb_classification`** — EC multi-label (multi-head) variant _(e.g. ec_number)_

```json
{
  "n_estimators": 50,
  "max_depth": 4,
  "learning_rate": 0.1,
  "subsample": 0.8,
  "colsample_bytree": 0.1,
  "tree_method": "hist",
  "max_bin": 64,
  "block_size": 256,
  "n_jobs": -1,
  "device": "cuda",
  "random_state": 0,
  "eval_metric": "logloss",
  "early_stopping_rounds": 20
}
```

**`xgb_classification`** — per-residue variant _(e.g. active_sites)_

```json
{
  "n_estimators": 50,
  "max_depth": 4,
  "learning_rate": 0.1,
  "subsample": 0.8,
  "colsample_bytree": 0.5,
  "max_bin": 128,
  "n_jobs": -1,
  "device": "cuda",
  "random_state": 0,
  "eval_metric": "logloss",
  "early_stopping_rounds": 10
}
```

`n_estimators` = number of boosting rounds (trees); `max_depth` = maximum tree depth (capacity / overfitting control); `learning_rate` = shrinkage per round; `subsample` / `colsample_bytree` = row/feature subsampling for regularization; `tree_method=hist` with `max_bin` controls histogram binning; `early_stopping_rounds` halts on the validation metric. Runs on GPU (`device=cuda`) with a CPU/OOM fallback.

## Documentation references

- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [XGBoost parameters reference](https://xgboost.readthedocs.io/en/stable/parameter.html)

## Benchmark standing (auto — mean/best test score across the 14 models)

### Random split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `align_amylase_activity` | mae | 0.2203 | 0.1967 | 14 |
| `align_amylase_expression` | mae | 0.1259 | 0.1081 | 14 |
| `align_amylase_stability` | mae | 0.3005 | 0.2418 | 14 |
| `align_glucosidase_activity` | mae | 175.0786 | 62.9715 | 14 |
| `align_glucosidase_expression` | mae | 0.8143 | 0.5083 | 14 |
| `align_glucosidase_melting` | mae | 0.8035 | 0.6244 | 14 |
| `binding` | f1 | 0.809 | 0.8184 | 14 |
| `cleeveland_galox` | mae | 1.0264 | 0.0562 | 14 |
| `ddg` | mae | 0.7132 | 0.6404 | 14 |
| `dtm` | mae | 0.4288 | 0.3802 | 14 |
| `ec_number` | f1 | 0.5051 | 0.7111 | 14 |
| `erickson_petase` | mae | 2.3373 | 1.717 | 14 |
| `kcat` | mae | 0.6574 | 0.6482 | 14 |
| `km` | mae | 0.6228 | 0.6028 | 14 |
| `ph_opt` | mae | 0.536 | 0.5139 | 14 |
| `solubility` | f1 | 0.7417 | 0.7668 | 14 |

### Similarity split

| Task | Primary metric | Mean over models | Best | Models |
|---|---|---|---|---|
| `binding` | f1 | 0.8211 | 0.836 | 14 |
| `ddg` | mae | 0.5393 | 0.4491 | 14 |
| `dtm` | mae | 0.6637 | 0.5987 | 14 |
| `ec_number` | f1 | 0.1858 | 0.3289 | 14 |
| `kcat` | mae | 0.6189 | 0.595 | 14 |
| `km` | mae | 0.6556 | 0.6337 | 14 |
| `ph_opt` | mae | 0.7264 | 0.6912 | 14 |
| `solubility` | f1 | 0.66 | 0.7003 | 14 |

## When to use

A strong default for the global protein and protein+ligand tasks: robust to feature scaling, captures nonlinear interactions, and effective on medium-sized tabular data (notably `kcat`, `km`, `binding`).

## Caveats

Operates only on the pooled vector, so it discards per-residue structure (no attention/residue-aware pooling). It can overfit tiny real_case datasets and, being tree-based, extrapolates poorly outside the training target range (see the anomalous `align_glucosidase_activity` MAE).

## References
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016. <https://doi.org/10.1145/2939672.2939785>
