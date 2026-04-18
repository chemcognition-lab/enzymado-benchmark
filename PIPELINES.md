# Training Pipelines

This document describes how training pipelines are selected in this repository and which models run for each property.

## Source Of Truth

The training flow is split across four places:

1. `benchmark_card.json`
   Defines the benchmark properties and their high-level `task_type` (`classification` or `regression`).
2. `models/train.py`
   Orchestrates the end-to-end training loop, maps card names to registry names, and selects the training route.
3. `src/data_loading/registry.py`
   Defines the dataset-level configuration for each task, especially `batch_kind` and `label_kind`.
4. `models/models.py`
   Defines the available model families through `MODEL_MAP` and `STRUCTURED_MODEL_MAP`.

## How A Pipeline Is Resolved

For each property in `benchmark_card.json`, training follows this order:

1. The property name is resolved to a registry task name.
   Example: `phopt -> ph_opt`, `binding_site -> binding_sites`.
2. The task config is loaded from `TASK_REGISTRY`.
3. The task's `batch_kind` and `label_kind` determine which data-building path is used.
4. The task's `label_kind` and card `task_type` determine which model map is used.
5. The selected model is trained through either:
   - direct pooled arrays with `fit(...)`, or
   - residue-aware `DataLoader` batches with `fit_loader(...)`.

## Main Training Routes

There are five training routes:

1. `protein_global` + `binary` or `scalar_regression`
   Mean-pool protein embeddings first, then train on 2D arrays.
2. `protein_global` + `multilabel_ec`
   Mean-pool protein embeddings first, then train either:
   - a hierarchical multi-head EC classifier, or
   - a flat concatenated EC XGBoost classifier with per-column masking for unknown descendants.
3. `protein_ligand_global`
   Mean-pool protein embeddings, concatenate ligand features, then train on 2D arrays.
4. `mutation_delta_global`
   Build mutant minus wildtype pooled embeddings, then train on 2D arrays.
5. `protein_residue` + `residue_binary_vector`
   Keep residue-level structure and train through `DataLoader` batches.

## Model Selection Rules

- Standard tasks use `MODEL_MAP[task_type]`.
- Structured tasks use `STRUCTURED_MODEL_MAP[label_kind]`.
- `multilabel_ec` uses a dedicated structured classifier map.
- EC XGBoost uses one concatenated label vector across all EC hierarchy levels.
- Unknown descendants from incomplete EC annotations are masked out per column for EC XGBoost.
- `residue_binary_vector` uses dedicated residue-aware classifiers.
- XGBoost is part of the standard model coverage across the registered properties.

## Property Matrix

| Card property | Registry task | Task type | Batch kind | Label kind | Training route | Models that run |
| --- | --- | --- | --- | --- | --- | --- |
| `solubility` | `solubility` | classification | `protein_global` | `binary` | pooled protein-global | `mlp_classification`, `sgd_classification`, `xgb_classification` |
| `phopt` | `ph_opt` | regression | `protein_global` | `scalar_regression` | pooled protein-global | `mlp_regressor`, `sgd_regressor`, `xgb_regression` |
| `binding` | `binding` | classification | `protein_ligand_global` | `binary` | pooled protein + ligand | `mlp_classification`, `sgd_classification`, `xgb_classification` |
| `kcat` | `kcat` | regression | `protein_ligand_global` | `scalar_regression` | pooled protein + ligand | `mlp_regressor`, `sgd_regressor`, `xgb_regression` |
| `km` | `km` | regression | `protein_ligand_global` | `scalar_regression` | pooled protein + ligand | `mlp_regressor`, `sgd_regressor`, `xgb_regression` |
| `dtm` | `dtm` | regression | `mutation_delta_global` | `scalar_regression` | pooled mutation-delta | `mlp_regressor`, `sgd_regressor`, `xgb_regression` |
| `ddg` | `ddg` | regression | `mutation_delta_global` | `scalar_regression` | pooled mutation-delta | `mlp_regressor`, `sgd_regressor`, `xgb_regression` |
| `ec_number` | `ec_number` | classification | `protein_global` | `multilabel_ec` | pooled hierarchical EC or pooled flat EC for XGBoost | `mlp_classification`, `sgd_classification`, `xgb_classification` |
| `active_sites` | `active_sites` | classification | `protein_residue` | `residue_binary_vector` | residue-aware loader pipeline | `mlp_classification`, `sgd_classification`, residue `xgb_classification` |
| `binding_site` | `binding_sites` | classification | `protein_residue` | `residue_binary_vector` | residue-aware loader pipeline | `mlp_classification`, `sgd_classification`, residue `xgb_classification` |

## Important Clarifications

- Every registered property now has a training path for the repository's model families:
  `MLP`, `SGD`, and `XGBoost`.
- For attention models, the preferred baseline name is `attention_linear_*`.
  The older `attention_sgd_*` names are still accepted as compatibility aliases.
- `ec_number` supports XGBoost through a flat concatenated EC target vector.
- `active_sites` and `binding_site` do use a dedicated structured model map.
- `fluorescence` is present in the benchmark card but is skipped because it is does not contain enzymes.

## Reading Guide

If you want to inspect the exact code path for a task:

1. Start in `src/data_loading/registry.py` to find `batch_kind` and `label_kind`.
2. Open `models/train.py` and look at the branch conditions in `main(...)`.
3. Open `src/data_loading/datasets.py` to see how the dataset and builders are assembled.
4. Open `models/models.py` to see which wrapper class each model name actually instantiates.

## Short Answer To A Common Question

"Do all tasks work the same across models?"

Almost.

- All registered properties have `MLP`, `XGBoost` and `SGD` coverage.
- `ec_number` uses a dedicated structured EC pipeline, and XGBoost follows a separate flat-label variant of that pipeline.
- Residue-level tasks use a different structured pipeline.
- Unregistered properties are skipped entirely, which currently means `fluorescence`.
