# Reproducing the published results

This guide reruns the published benchmark as it is. To add your own dataset,
or to train your own model and get its weights, see [add_data.md](add_data.md).

| Level | You reproduce | You need | Time |
|---|---|---|---|
| [1. Leaderboards](#level-1-recompute-the-leaderboards) | the two global ratings, from the published files | a laptop | minutes |
| [2. One task](#level-2-rerun-one-task) | any task × embedding model × head you choose, retrained | one GPU, or a CPU for small tasks | minutes to hours |
| [3. Everything](#level-3-rerun-everything-on-a-slurm-cluster) | all 2,100 runs (1,960 on the 14 PLMs, 140 on `noplm`), from embeddings up | a Slurm GPU cluster and about 5 TB of storage | days to weeks |

## Setup

Requires Python ≥3.12,<3.14, [uv](https://docs.astral.sh/uv/) and **Git LFS**.
Without Git LFS the datasets arrive as pointer files and every run fails.

```bash
git lfs install
git clone <repo-url> && cd enzymado-benchmark
git checkout main
uv sync --extra train
source .venv/bin/activate
```

Run every command from the repository root. Use the `main` branch: other
branches may register extra tasks, which changes the leaderboards.

**In the repository:** the datasets and splits, the sequence index, the
shared-ID list (`results/embedding_intersection/common_ids.txt`), and every
published `metrics.json`, `hparams.json`, bootstrap file and leaderboard.
**Not in the repository, so you rebuild them:** the embeddings (about 4.6 TB
at residue level and 11 GB mean-pooled for the 14 PLMs), prediction files and
model weights.

**How exact is a rerun?** Every published run used seed 42. GPU kernels were
not forced to be deterministic, so retrained neural heads come out close to
the published numbers but not identical. Compare against the published 95%
confidence intervals, not the last digit.

---

## The tasks

Use these names with `--task_name`.

**Properties** (the default; each has a `random` and a `similarity` split):

| Task | Predicts | One row is | Problem |
|---|---|---|---|
| `solubility` | soluble or not | a protein | yes/no |
| `ph_opt` | optimal pH | a protein | number |
| `ec_number` | EC number (enzyme class) | a protein | 4-level class |
| `binding` | binds the ligand or not | a protein + a ligand | yes/no |
| `kcat` | turnover number | a protein + a substrate | number |
| `km` | Michaelis constant | a protein + a substrate | number |
| `dtm` | change in melting temperature caused by a mutation | a mutant + its wild type | number |
| `ddg` | change in folding stability (ΔΔG) caused by a mutation | a mutant + its wild type | number |
| `active_sites` | which residues are catalytic | a protein, labelled per residue | yes/no per residue |
| `binding_site` | which residues bind a ligand | a protein, labelled per residue | yes/no per residue |

`dtm` and `ddg` are the **mutation tasks**. Each row is one mutant (for example
`P21S`) of a known wild-type protein. The model sees the mutant's embedding
minus the wild type's, so it has to learn what the mutation changed, not which
protein it is.

**Real cases** (need `--task_scope real_case`; regression, `random` split
only): `align_amylase_activity`, `align_amylase_expression`,
`align_amylase_stability`, `align_glucosidase_activity`,
`align_glucosidase_expression`, `align_glucosidase_melting`,
`cleeveland_galox` (protein + substrate) and `erickson_petase`. The six
`align_*` datasets are libraries of variants of one enzyme, differing by a few
residues.

## What one run trains

A run trains one **head** (a small model) on frozen embeddings from one
**embedding model**, for one task and one split.

- **Heads.** `linear_`, `mlp_` and `xgb_` are trained on mean-pooled
  embeddings. `attention_linear_` and `attention_mlp_` learn their own pooling
  over residues, and only run when you pass a residue-level manifest
  (`--attention_embeddings_json`). Each head uses the classification,
  regression, EC or per-residue version that matches the task.
- **Inputs.** Protein tasks use the mean-pooled embedding. Ligand tasks add
  the 200 descriptastorus `RDKit2DNormalized` descriptors of the `smiles`. Mutation tasks use mutant minus wild
  type. Per-residue tasks keep one embedding per residue.
- **Training.** Settings are fixed in
  `src/enzymado/training/model_registry.py`. The MLP settings come from one
  early search on `km` (`scripts/mlp_hpo/`) and are reused everywhere. Neural
  heads keep the epoch with the best validation Kendall τ (regression) or
  AUPRC (yes/no and EC number), and per-residue heads the one with the lowest
  validation loss. XGBoost stops early on validation AUPRC (yes/no), RMSE
  (regression) or log loss (EC number and per-residue tasks).
- **Scoring.** Every metric is computed on the test split, with 95% bootstrap
  intervals. See the README for how the leaderboards are built.

---

## Level 1: Recompute the leaderboards

No GPU or embeddings needed:

```bash
python scripts/results/compute_ratings.py --ratings-root results/ratings_rerun
diff results/ratings/seed_42_two_leaderboards/properties.csv results/ratings_rerun/properties.csv
diff results/ratings/seed_42_two_leaderboards/real_case.csv  results/ratings_rerun/real_case.csv
```

Always pass `--ratings-root`. Its default is the published folder, which a
rerun would overwrite.

---

## Level 2: Rerun one task

The example retrains `ph_opt` with `esm2_650M`, the cheapest useful case
(10,794 sequences).

### 1. Embed the task's sequences

On a Slurm cluster, from a login node:

```bash
TASK=ph_opt EMBED_MODELS="esm2_650M" bash scripts/slurm/submit_task_embeddings.sh
```

When the job finishes, it has written the manifest
`embeddings_ph_opt_pooled.json`.

On any other machine, first install the ESM-2 environment from
`envs_files/venv-esm2.txt`. A GPU is much faster but not required. Then run:

```bash
python scripts/data_prep/subset_sequence_index_for_task.py \
  -t ph_opt -o data/processed/ph_opt_sequences.csv
python scripts/data_prep/make_fasta_from_unique_sequences.py \
  --input data/processed/ph_opt_sequences.csv --output data/fastas/ph_opt_sequences.fasta
python embedding_scripts/esm2_embedding_script.py \
  -i data/fastas/ph_opt_sequences.fasta \
  -o embeddings/ph_opt_sequences/esm2_650M_shards \
  --model facebook/esm2_t33_650M_UR50D --per_protein 0 \
  --mean_pooled_output data/embeddings_mean_pooled/ph_opt_sequences/esm2_650M.safetensors
```

and write `embeddings_ph_opt_pooled.json`:

```json
{
  "esm2_650M": {
    "path": "data/embeddings_mean_pooled/ph_opt_sequences/esm2_650M.safetensors",
    "dim": 1280
  }
}
```

Every embedding model's key and `dim` is in the README under *What it
evaluates*. Each model family has its own script in `embedding_scripts/` and
environment in `envs_files/`.

### 2. Train

`scripts/run_benchmark.py` trains every task, embedding model and head unless
you narrow it down:

| Option | Selects | Default |
|---|---|---|
| `--task_name ph_opt` | one task; repeat for several | all tasks in the scope |
| `--task_scope real_case` | real-case tasks | `properties` only |
| `--split_variant similarity` | `random` or `similarity` | `random` |
| `--emb_name esm2_650M` | one embedding model from the manifest | all in the manifest |
| `--model_prefix mlp_` | one head: `linear_`, `mlp_`, `xgb_`, `attention_linear_`, `attention_mlp_` | all |
| `--attention_embeddings_json <manifest>` | also train the attention heads, on this residue-level manifest | attention heads do not run |
| `--embedding_mode attention` | only the attention heads (`no_attention`: only the others) | both |
| `--output_dir <folder>` | where results go | `results/predictions` (the published folder) |
| `--allowed_ids results/embedding_intersection/common_ids.txt` | only the proteins every embedding model covers, as in the published runs | all rows |
| `--overwrite_results` | retrain runs that already finished | finished runs are skipped |

To match the published run, train both splits with the shared-ID list:

```bash
for split in random similarity; do
  python scripts/run_benchmark.py embeddings_ph_opt_pooled.json \
    --output_dir results_rerun/predictions \
    --task_name ph_opt --emb_name esm2_650M \
    --split_variant "$split" \
    --allowed_ids results/embedding_intersection/common_ids.txt --seed 42
done
```

Common mistakes:

- **No `--output_dir`:** your runs are written into the published `results/`
  folder.
- **A real-case task without `--task_scope real_case`:** the task is skipped
  with no error, and nothing trains.
- **`train_task_pooled.slurm`** always passes `--task_scope properties`, so it
  cannot run real-case tasks. Use the command above in a job instead.

### 3. Score and compare

The scoring scripts take the same kind of filters, and their defaults also
point at the published folders, so pass your own:

```bash
python scripts/results/compute_metrics.py \
  --predictions-root results_rerun/predictions \
  --statistics-root results_rerun/statistics --filter-task ph_opt
python scripts/results/compute_bootstrap_metrics.py \
  --predictions-root results_rerun/predictions \
  --statistics-root results_rerun/statistics \
  --bootstrap-root results_rerun/bootstrap \
  --filter-task ph_opt --n-replicates 1000
```

Run them in that order: the second reuses a cache the first writes. Each run's
scores are in a `metrics.json` at
`results_rerun/predictions/<scope>/<attention|no_attention>/<split>/<head>/<task>/<embedding>/`,
the same layout as `results/predictions/`. To compare every rerun with the
published number and its interval:

```python
import json
from pathlib import Path

published, rerun = Path("results/predictions"), Path("results_rerun/predictions")
for path in sorted(rerun.rglob("metrics.json")):
    run = path.relative_to(rerun)
    ref = json.loads((published / run).read_text())
    new = json.loads(path.read_text())["test"]
    interval = ref["test_bootstrap"]["metrics"]
    for name, value in new.items():
        if name in interval:
            lo, hi = interval[name]["ci_lo"], interval[name]["ci_hi"]
            print(f"{run.parent}  {name:12s} published={ref['test'][name]:.4f} "
                  f"rerun={value:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
```

---

## Level 3: Rerun everything on a Slurm cluster

One pass of training took about 1,700 GPU-hours on L40S GPUs, plus one GPU
job of up to 24 h per embedding model.

**Cluster assumptions.** `scripts/slurm/` was written for the Killarney
cluster (Alliance Canada). It expects `module load python/3.13.2` and
`python/3.12.4`, GPUs requested as `--gres=gpu:l40s:1`, large data under
`/scratch/$USER`, compute nodes without internet, and the repository outside
`/scratch`. On another cluster, change the `module load` lines, `--gres` and
paths. The script headers hard-code a maintainer's account and e-mail, so
always pass your own (`AIP_ACCOUNT` and `MAIL_USER`, or `--account` and
`--mail-type=NONE` for a direct `sbatch`).

**1. Environments and weights.** On a login node, build the training
environment with `uv sync --extra train`, then the five embedding
environments that `run_embeddings_cluster.slurm` expects (`.venv-prot5`,
`.venv-esm2`, `.venv-esm3-esmc`, `.venv-ankh` and `.venv-profluent`) from the
pinned files in `envs_files/`. Compute nodes have no internet, so download
every model's weights into the cache the job reads (`HF_HOME`, set at the top
of `run_embeddings_cluster.slurm`) beforehand. ESM3 is gated on Hugging Face,
so you need a token with access to it.

**2. Embeddings.** One array task per model, at most two at a time:

```bash
sbatch --account=<your-account> --mail-type=NONE --array=0-13%2 \
  --export=ALL,SAVE_MEAN_POOLED=1 scripts/slurm/run_embeddings_cluster.slurm
```

Residue-level files go to `/scratch/$USER/enzymado/data/embeddings/`. Make
sure your cluster's scratch purge does not delete them before training ends.
Do not rebuild `data/processed/protein_pool.csv`: the published IDs depend on
it.

**3. Manifests.**

```bash
cp embeddings_models_output.example.json embeddings_models_output.json
cp embeddings_models_output_mean_pooled.example.json embeddings_models_output_mean_pooled.json
```

Point each `"path"` in `embeddings_models_output.json` at its folder under
`/scratch`. The mean-pooled template is already correct.

**4. Training.** `run_training_cluster.slurm` trains the mean-pooled heads and
`run_training_attention_cluster.slurm` the attention heads, one array task per
embedding model; their headers list the variables they read. Always set
`OUTPUT_DIR`, whose default is the published `results/predictions`:

```bash
sbatch --account=<your-account> --mail-type=NONE --array=0-13%2 \
  --export=ALL,OUTPUT_DIR=$PWD/results_rerun/predictions \
  scripts/slurm/run_training_cluster.slurm
```

Add `TASK_SCOPES=real_case` for the real-case datasets. The EC and residue
tasks need far more memory and time than the rest (up to 256 GB and several
hours per run), so submit each as its own array with `TASK_NAMES` and a larger
`--mem`, one task per array: `--export` splits its values at commas. Finished runs
are skipped, so after a `TIMEOUT` or `OUT_OF_MEMORY`, submit the same command
again to continue. A failed run leaves `training_failure.json` in its folder;
delete it before resubmitting, or the run is not retried.

**5. Score and rate.**

```bash
RUN=results_rerun
python scripts/results/compute_metrics.py \
  --predictions-root $RUN/predictions --statistics-root $RUN/evaluation_statistics
python scripts/results/compute_bootstrap_metrics.py \
  --predictions-root $RUN/predictions --statistics-root $RUN/evaluation_statistics \
  --bootstrap-root $RUN/bootstrap --n-replicates 1000 --seed 0
python scripts/results/compute_ratings.py \
  --properties-predictions-root $RUN/predictions \
  --real-case-no-attention-predictions-root $RUN/predictions \
  --real-case-attention-predictions-root $RUN/predictions \
  --properties-bootstrap-root $RUN/bootstrap \
  --real-case-no-attention-bootstrap-root $RUN/bootstrap \
  --real-case-attention-bootstrap-root $RUN/bootstrap \
  --ratings-root $RUN/ratings --n-battles 1000
```

Compare `$RUN/ratings/*.csv` with `results/ratings/seed_42_two_leaderboards/`,
or compare run by run with the Level 2 snippet using
`rerun = Path("results_rerun/predictions")`.
`python scripts/results/audit_results_tree.py --predictions-root $RUN/predictions`
lists missing or incomplete runs.
