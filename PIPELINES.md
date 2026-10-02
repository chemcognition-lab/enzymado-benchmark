# Training Pipelines

This document describes how training pipelines are selected in this repository and which models run for each property.

## Source Of Truth

The active training flow is defined by:

1. `benchmark_card.json`: benchmark metadata and rating eligibility rules.
2. `scripts/run_benchmark.py` and `src/enzymado/pipeline/`: CLI selection, data
   preparation, and execution of each model/task/embedding combination.
3. `src/enzymado/data_loading/data_registry.py`: task configuration, including
   `batch_kind`, `label_kind`, target columns, and available dataset splits.
4. `src/enzymado/training/model_registry.py` and `training/train.py`: model
   families and pooled, residue, or attention training routes.
5. `src/enzymado/prediction/`: prediction, model persistence, and optional
   attachment of probabilities from restored checkpoints.

## How A Pipeline Is Resolved

For each selected task in `TASK_REGISTRY`, training follows this order:

1. The CLI selects registered tasks and validates requested task names.
2. The task config is loaded from `TASK_REGISTRY`.
3. The task's `batch_kind` and `label_kind` determine which data-building path is used.
4. The task's `label_kind` determines the task type and model map.
5. The selected route trains the model through either:
   - direct pooled arrays with `fit(...)`, or
   - residue-aware `DataLoader` batches with `fit_loader(...)`.
6. Each run writes prediction files plus experiment-style artifacts:
   compressed `*.safetensors.xz` predictions, `hparams.json`,
   `training_log.csv`, and the saved model. Torch models also emit safetensors
   weights. The evaluation CLI writes `metrics.json`. `hparams.json` is the
   provenance record: it carries the resolved task config, dataset paths,
   hyperparameters, and the random seed. `run_manifest.json` is also emitted by
   `artifacts.reproducibility`, but no published run under `results/` carries
   one, so nothing downstream may depend on it.

## CLI Filters For SLURM

`scripts/run_benchmark.py` accepts an embeddings manifest followed by filters:

```bash
python scripts/run_benchmark.py embeddings_models_output.json \
  --output_dir results/predictions \
  --emb_name ankh_base \
  --model_prefix mlp_ \
  --task_name ph_opt \
  --split_variant random
```

- `--emb_name` selects one foundation model / embedding entry.
- `--model_prefix` selects a model family: `mlp_`, `linear_`, or `xgb_`.
- `--task_name` selects a task. Repeat it to train multiple tasks.
- `--task_scope` selects a benchmark-card section, `properties` or `real_case`.
  It defaults to `properties`, so real-case tasks need it explicitly.
- `--split_variant` selects `random` or `similarity`. If omitted, the CLI uses
  `random`. A task that registers no similarity split is skipped when
  `similarity` is requested.
- `--embedding_mode` selects `attention` or `no_attention` runs.
- `--attention_embeddings_json` supplies the per-residue manifest. Attention
  models do not train at all without it.
- `--allowed_ids` optionally restricts rows to embeddings present in an ID
  allowlist. The published runs pass
  `results/embedding_intersection/common_ids.txt` so every embedding model is
  scored on the same proteins.
- `--residue_store_mode` forces `materialized` or `lazy` residue access instead
  of the `auto` size heuristic.
- `--overwrite_results` replaces existing artifacts instead of skipping
  completed runs; `--deterministic` requires deterministic PyTorch kernels.

`python scripts/run_benchmark.py --help` lists the remaining attention
learning-rate and batch-diagnostic overrides.

Similarity splits were generated with an MMseqs2 minimum sequence identity of
`0.3`. The `0.5` threshold recorded in dataset cards belongs to a separate
whole-dataset diversity analysis and does not define the train/validation/test
splits.

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
   Ligand features are the 200 descriptastorus `RDKit2DNormalized` descriptors of the
   substrate SMILES, read from `data/processed/ligand_features/descriptastorus.npz`.
   Build that cache once with `scripts/data_prep/build_ligand_features.py`; training
   stops if a SMILES is missing from it. These are 2D descriptors, so stereoisomers
   (for example L- and D-alanine) share one vector.
4. `mutation_delta_global`
   Build mutant minus wildtype pooled embeddings, then train on 2D arrays.
5. `protein_residue` + `residue_binary_vector`
   Keep residue-level structure and train through `DataLoader` batches.

## Computational Policy

Training protocol v3 selects only trained neural epochs. Epoch 0 is saved as
`initial_model.pt` and recorded as a reference, never as the final trained model.
The history explicitly records whether the selected model improved that reference.

Pooled neural tasks select on AUPRC for classification and Kendall tau for
regression, evaluated on the whole validation split and negated for minimization.
EC and per-residue networks retain validation loss. These are protocol choices;
none guarantees agreement with every metric in the leaderboard. Standard XGBoost
uses `aucpr` for classification and `rmse` for regression.

Non-finite losses, gradients, weights and monitored predictions fail the run.
Undefined Kendall from constant predictions makes an epoch ineligible; AUPRC with
constant probabilities remains defined when both classes are present. A run with
no eligible trained epoch fails explicitly. Old checkpoints remain loadable for
inference, but resumption requires a matching v3 protocol and provenance contract.

Evaluation differs from selection on one point. A reported Kendall tau, including
each bootstrap replicate, is 0 when all predictions or all labels are tied: a full
tie orders nothing, so it scores as no skill instead of undefined. This matters for
XGBoost, which stops on `rmse` and can end up almost constant on a small task.

The v3 runbook under `docs/` carries the reporting fields, migration rules and
isolated launch/evaluation commands; it is kept locally and not published. These
changes require a new benchmark version.

Classification accuracy, precision, recall, and F1 use saved hard labels.
AUROC and AUPRC use score histograms when the artifact includes probabilities;
otherwise they use the hard-label fallback. `ranking_source` records that choice.
`--attach_probabilities` restores completed classification runs and adds test
probabilities while preserving their hard-label tensors. Metrics and bootstrap
must then be recomputed to reflect the new ranking scores.

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
| `solubility` | `solubility` | classification | `protein_global` | `binary` | pooled protein-global | `mlp_classification`, `linear_classification`, `xgb_classification` |
| `ph_opt` | `ph_opt` | regression | `protein_global` | `scalar_regression` | pooled protein-global | `mlp_regressor`, `linear_regressor`, `xgb_regression` |
| `binding` | `binding` | classification | `protein_ligand_global` | `binary` | pooled protein + ligand | `mlp_classification`, `linear_classification`, `xgb_classification` |
| `kcat` | `kcat` | regression | `protein_ligand_global` | `scalar_regression` | pooled protein + ligand | `mlp_regressor`, `linear_regressor`, `xgb_regression` |
| `km` | `km` | regression | `protein_ligand_global` | `scalar_regression` | pooled protein + ligand | `mlp_regressor`, `linear_regressor`, `xgb_regression` |
| `dtm` | `dtm` | regression | `mutation_delta_global` | `scalar_regression` | pooled mutation-delta | `mlp_regressor`, `linear_regressor`, `xgb_regression` |
| `ddg` | `ddg` | regression | `mutation_delta_global` | `scalar_regression` | pooled mutation-delta | `mlp_regressor`, `linear_regressor`, `xgb_regression` |
| `ec_number` | `ec_number` | classification | `protein_global` | `multilabel_ec` | pooled hierarchical EC or pooled flat EC for XGBoost | `mlp_classification`, `linear_classification`, `xgb_classification` |
| `active_sites` | `active_sites` | classification | `protein_residue` | `residue_binary_vector` | residue-aware loader pipeline | `mlp_classification`, `linear_classification`, residue `xgb_classification` |
| `binding_site` | `binding_site` | classification | `protein_residue` | `residue_binary_vector` | residue-aware loader pipeline | `mlp_classification`, `linear_classification`, residue `xgb_classification` |

## Real-Case Matrix

`--task_scope real_case` selects the eight applied datasets. All are regression,
all register only a `random` split, and all key rows by `dataset_id` alone.

| Registry task | Batch kind | Training route | Extra features |
| --- | --- | --- | --- |
| `align_amylase_activity` | `protein_global` | pooled protein-global | - |
| `align_amylase_expression` | `protein_global` | pooled protein-global | - |
| `align_amylase_stability` | `protein_global` | pooled protein-global | - |
| `align_glucosidase_activity` | `protein_global` | pooled protein-global | - |
| `align_glucosidase_expression` | `protein_global` | pooled protein-global | - |
| `align_glucosidase_melting` | `protein_global` | pooled protein-global | - |
| `cleeveland_galox` | `protein_ligand_global` | pooled protein + ligand | `smiles`, `substrate_concentration` |
| `erickson_petase` | `protein_global` | pooled protein-global | `temperature` |

Because they are regression only, the `real_case` leaderboard scores 24 metric
cells: 8 tasks x 1 split x 3 rating-eligible metrics (`mae`, `rmse`,
`kendall_tau`; `r2` carries weight zero).

## Important Clarifications

- Every registered property has a training path for the repository's model families:
  `MLP`, linear/SGD, and `XGBoost`.
- For attention models, the linear baseline name is `attention_linear_*`.
- `ec_number` supports XGBoost through a flat concatenated EC target vector.
- `active_sites` and `binding_site` do use a dedicated structured model map.
- `fluorescence` was dropped from the benchmark card because it does not contain
  enzymes. It is neither registered nor scored.

## Reading Guide

If you want to inspect the exact code path for a task:

1. Start in `src/enzymado/data_loading/data_registry.py` to find `batch_kind` and `label_kind`.
2. Open `src/enzymado/pipeline/run_data.py` and inspect `prepare(...)` and its selected training route.
3. Open `src/enzymado/data_loading/datasets.py` to see how the dataset and builders are assembled.
4. Open `src/enzymado/training/model_registry.py` to see which wrapper class each model name actually instantiates.

## Short Answer To A Common Question

"Do all tasks work the same across models?"

Almost.

- All registered properties have `MLP`, `XGBoost` and linear coverage.
- `ec_number` uses a dedicated structured EC pipeline, and XGBoost follows a separate flat-label variant of that pipeline.
- Residue-level tasks use a different structured pipeline.
- Tasks absent from `TASK_REGISTRY` are skipped entirely.

## Post-training evaluation

Primary metrics are recomputed directly from the prediction artifacts; no
separate results manifest is required. The benchmark card supplies the task
type when a run no longer contains `hparams.json`:

```bash
python scripts/results/compute_metrics.py --filter-task ec_number
```

Classification tasks are reduced to per-row TP, FP, FN, and TN counts under
`results/evaluation_statistics/`. Each cache records the SHA-256 digest of its
source prediction. Multi-head EC tensors are processed level by level, avoiding
their dense concatenation. Accuracy, precision, recall, and F1 use global micro aggregation. AUROC and AUPRC
use probability histograms when available and confusion counts otherwise.
The bootstrap pass reuses these statistics:

```bash
python scripts/results/compute_bootstrap_metrics.py \
    --filter-task ec_number --n-replicates 200
```

Bootstrap units are proteins for EC and residue tasks and rows for other tasks.
Experiments share resampling groups by `(split, task)` across model families and
pooling modes. Stored sample identity is checked before sharing indices.

Every registered metric is computed and bootstrapped. Rating eligibility is a
separate step: `r2` has weight zero; `benchmark_card.json:rating_exclusions` can
withhold accuracy for low-prevalence splits, or Kendall tau/AUROC cells whose best
competitor does not exceed the configured ceiling. These exclusions do not remove
metrics from per-model result files.
