Reproduce the Appendix B MLP search from the repository root. The commands use
the existing training environment, current K_M task loader and mean-pooled
`esmc_300m` embeddings. Prepare those inputs with the normal benchmark workflow
before submitting jobs.

The split is **similarity_mmseqs2_80_10_10**, not the benchmark runner's current
random default. On `origin/hpo`, `scripts/run_training_cluster.slurm` delegated
split selection to `models/train.py`, which read K_M's `default_split` from
`benchmark_card.json`. That card selected the MMseqs2 similarity split.
The runner explicitly selects its current registry equivalent, `similarity`.
The original SLURM script had no embedding-intersection allowlist by default,
so this search uses all available rows from that split.

Generate the original 400 Latin hypercube configurations:

```bash
uv run --extra train python scripts/mlp_hpo/generate_configs.py \
  --n-configs 400 --sampling lhs --seed 42 --batch-size 256 --num-epochs 20 \
  --output scripts/mlp_hpo/configs_400_lhs.json
```

The search-space order, random draws, deduplication and refill order match
`origin/hpo`. The committed JSON was downloaded from that branch with Git LFS,
not generated as a replacement. The test compares the regenerated list and
bytes to that original artifact. If a fresh checkout contains an LFS pointer,
download this file with:

```bash
git lfs pull --include=scripts/mlp_hpo/configs_400_lhs.json --exclude=''
```

Run a single configuration locally, optionally adding `--device cpu`:

```bash
uv run --extra train python scripts/mlp_hpo/run_hpo.py \
  --config-index 0 \
  --embeddings-json embeddings_models_output_mean_pooled.json \
  --output-dir results/mlp_hpo
```

Submit the full search as 400 array tasks, with at most ten running at once:

```bash
uv run --extra train python scripts/mlp_hpo/hpo_job_submitter.py \
  --account aip-bensl --n-configs 400 --max-concurrent 10 \
  --gpus l40s:1 --exclude kn051,kn092,kn116,kn117 \
  --embeddings-json embeddings_models_output_mean_pooled.json \
  --output-dir results/mlp_hpo
```

Add `--dry-run` to inspect the exact `sbatch` command without submitting or
creating directories. The default cluster executable is `.venv/bin/python`;
override it with `--python /path/to/environment/bin/python`. The SLURM launcher
loads Python 3.13.2 when environment modules are available. GPU, CPU, memory and
time allocations are CLI options; the default GPU is one L40S. Pass
`--exclude` with the nodes currently known to fall back to CPU. Submission itself needs only Python's
standard library, so it can also run with `python` from the cluster login node.
To resubmit a subset, use `--start-index 100 --n-configs 25`, for example.
There is no hidden last-submitted-index file. A rerun replaces that config's
metrics after successful completion.

Every configuration uses AdamW, batch size 256 and the final model after all
20 epochs. There is no early stopping or checkpoint selection within a run.
The sampled `num_layers` is translated into repeated `hidden_dims`. Optional
layer normalization precedes each hidden projection. SwiGLU uses two thirds
of the requested width rounded up to a multiple of 32, and a doubled linear
projection, matching the original HPO implementation.

The runner disables the feature scaling that main added after the original
search. Other benchmark runs retain their existing scaling, SiLU activation,
architecture, optimization, checkpoint names and selection defaults. K_M
targets and ligand features are supplied by the current loader, including the
`km_standardized` target column; no data preparation or evaluation code is
changed. Metrics are therefore expressed in the current target's units.
The sampling seed is 42. New runs also record an explicit training seed
(default 42); the historical submitter did not fix the training RNG seed.
This reproduces the reported search design and configuration list, rather than
promising identical floating-point scores across software and data revisions.

Aggregate the completed runs and select the winner by validation R²:

```bash
uv run --extra train python scripts/mlp_hpo/aggregate_hpo_results.py \
  --hpo-dir results/mlp_hpo/hpo_metrics \
  --output results/mlp_hpo/hpo_final_results.csv
```

The CSV is sorted by decreasing `val_R2`, and also includes `val_MAE`,
`val_Kendall`, every sampled parameter and training metadata. Undefined
Kendall tau is stored as JSON null. Aggregation requires 400 distinct completed
runs by default. Use `--expected-count N` only when intentionally reviewing a
partial search. No test-set predictions enter selection. Output consists of
isolated metrics JSON files and SLURM logs, without full model checkpoints or
benchmark prediction artifacts. Default outputs go to `results/mlp_hpo/`, which git ignores like other reruns.

Run the laptop checks without embeddings or a GPU:

```bash
uv run --extra train pytest tests/test_mlp_hpo.py
```

The exploratory batch-size study is omitted. Its separate bfloat16 model,
multi-GPU timing, plots, notebook and cluster shell launcher are outside the
fixed-batch Appendix B search; preserving that study would exceed a small
training script. Also omitted are committed `__pycache__` files and the old
branch's changes to `benchmark_card.json`, `.gitignore` and
`create_dataset_cards.py`. No environment-triggered registry injection or hooks
are needed in the regular benchmark runner, training routes or SLURM script.
Existing dependencies already supply the required libraries, so no
`pyproject.toml` changes are needed.
