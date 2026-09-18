# Training protocol v3

This version preserves initialization as a reference and selects only trained
neural checkpoints. It changes the benchmark protocol; keep published v2
artifacts intact. The new full benchmark covers 1,960 runs, including EC
attention, whose selection policy also changes.

## Selection and reporting

- Save initial neural weights as `initial_model.pt` and record their validation
  objective at epoch 0. Epoch 0 is never eligible as the final trained model.
- Select the first eligible trained epoch, then replace it only for improvements
  larger than `min_delta`. Ties retain the earlier trained checkpoint.
- Pooled neural classification selects on negative AUPRC; pooled regression
  selects on negative Kendall tau. EC and residue networks keep validation loss.
  These are selection objectives, not equivalent versions of all reported metrics.
- Constant predictions have defined AUPRC when both classes exist. Constant
  predictions have undefined Kendall and make an epoch ineligible. A validation
  target without variation fails before training under a ranking criterion.
- If no trained epoch has a defined metric, the run fails. An undefined baseline
  leaves `improved_over_initialization` empty, rather than claiming improvement.
- `training_log.csv` records `selection_objective`, `validation_score`,
  `best_validation_score`, `baseline_validation_score`, `best_epoch`, and
  `improved_over_initialization`. Objective values are minimized: ranking values
  are negated. Internal `best_val_loss_` remains a compatibility attribute.
- A trained model that does not improve its initial reference remains a valid
  reported result, with the flag set to false. Do not drop it from evaluation.

## Numerical failures and resumption

Non-finite losses, gradients, weights or monitored predictions abort training.
Only explicit annotation masks exclude observations; invalid predictions cannot
silently disappear from a metric. Pipeline failures produce
`training_failure.json` and require review before retrying. Retain failed-run
counts in benchmark reporting; missing runs must not silently improve rankings.

Checkpoint schema 3 records the selection protocol, constructor configuration,
backend, loader settings for Torch loaders and pipeline provenance. Reusing a
training checkpoint requires a matching contract. The pipeline compares dataset,
embedding and hparams fingerprints, split identities, source hash, package
versions and accelerator information before resuming. It preserves the original
manifest when these match. Low-level calls without a pipeline manifest cannot
verify the identity of externally supplied datasets.

Older checkpoints remain usable through `restore_fitted_wrapper` for inference.
They are rejected for continuation under this protocol. Changing epochs,
patience, metric or inputs requires a new run directory, not a silent overwrite
of the loaded configuration. Failed runs should be investigated before retrying;
a fresh directory makes changes to the failed configuration auditable.

## Launch preparation

`bash scripts/slurm/plan_training_v3.sh` prints the commands; it does not submit
jobs. Review and freeze the code and data before submission. The plan uses
`results/protocol_v3/predictions` and separate logs, and keeps seed 42 and the
existing common-ID filter. Eight arrays separate scopes, splits, pooling and
residue inputs. No-attention site tasks require residue embeddings, so they must
not be submitted with the mean-pooled embedding manifest.

First run a pilot with one embedding (array index 2, `ankh_base`) and a small
selection of tasks. Include at least one pooled classifier, one pooled regressor,
one attention route and one residue route. Tests verify numerical contracts but
do not establish convergence or runtime on the full datasets. Review the pilot's
failure files, baseline comparisons and resource usage before submitting all
arrays. The historical 1,049-hour sum is not a forecast for this protocol.

After all training jobs complete, verify completeness and explicitly account for
failures. Attach classification probabilities using the same launch commands
with `ATTACH_PROBABILITIES=1` and the same output root. This loads existing weights
rather than retraining. Do not run it concurrently with training.

Then run evaluation on a compute node, with every output root isolated:

```bash
.venv/bin/python scripts/results/compute_metrics.py \
  --predictions-root results/protocol_v3/predictions \
  --statistics-root results/protocol_v3/evaluation_statistics
.venv/bin/python scripts/results/compute_bootstrap_metrics.py \
  --predictions-root results/protocol_v3/predictions \
  --statistics-root results/protocol_v3/evaluation_statistics \
  --bootstrap-root results/protocol_v3/bootstrap --n-replicates 1000
.venv/bin/python scripts/results/compute_ratings.py \
  --properties-predictions-root results/protocol_v3/predictions \
  --properties-bootstrap-root results/protocol_v3/bootstrap \
  --real-case-no-attention-predictions-root results/protocol_v3/predictions \
  --real-case-no-attention-bootstrap-root results/protocol_v3/bootstrap \
  --real-case-attention-predictions-root results/protocol_v3/predictions \
  --real-case-attention-bootstrap-root results/protocol_v3/bootstrap \
  --ratings-root results/protocol_v3/ratings
```

These commands do not change the task weighting or split algorithms. Changing
those is a separate protocol decision.
