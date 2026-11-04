# Adding your own dataset or model

This guide has two parts:

- **[Part 1](#part-1-add-a-dataset)** takes you from a spreadsheet of enzyme
  measurements to benchmark results for that dataset.
- **[Part 2](#part-2-train-your-own-model-and-get-its-weights)** shows how to
  get trained model weights, and how to plug in your own model architecture.

To rerun the published benchmark without adding anything, see
[REPRODUCE.md](REPRODUCE.md). Run every command from the top folder of the
repository (the one that contains `README.md`).

## Words used in this guide

- **Task**: your dataset, registered under a short name. This guide uses
  `my_enzyme`. Replace it with your own name everywhere: lowercase letters,
  digits and underscores only.
- **Embedding**: a list of numbers that a protein language model computes for
  each protein sequence. The benchmark trains small models on these numbers.
- **Head**: one of those small models (linear, MLP or XGBoost).
- **Manifest**: a small JSON file that says where the embeddings are stored.

## Before you start

You need the repository set up once:

```bash
git lfs install
git clone <repo-url> && cd enzymado-benchmark
uv sync --extra train
source .venv/bin/activate
```

Run `source .venv/bin/activate` again each time you open a new terminal.

**Check your Python version before going further:**

```bash
python -V        # must be 3.12 or 3.13
```

The code uses syntax introduced in Python 3.12. An older interpreter fails at
Step 3 with a `SyntaxError` in `src/enzymado/dataset_splitting.py`, which looks
like a bug in the repository but is not one. `uv sync` installs a suitable
Python for you; if you built the environment some other way and `python -V`
reports 3.11 or older, delete `.venv` and rerun `uv sync --extra train`.

---

# Part 1: Add a dataset

Part 1 covers the common case: **each row has one number you want to
predict**, such as activity, melting temperature or expression level. For
anything else, see [What Part 1 does not cover](#what-part-1-does-not-cover).

## Step 1. Make the CSV

Create the folder `data/processed/real_case/my_enzyme/` and save your data in
it as `my_enzyme.csv`.

If your source needs reshaping, write the conversion as a script rather than
editing by hand: it keeps the dataset reproducible, and it is the only way to
share one whose licence does not let you redistribute the data.
`scripts/data_prep/convert_biolip_to_csv.py` is a working example, and
`scripts/data_prep/convert_enzyarena_to_csv.py` is another: it produced the
two EnzyArena datasets in `data/processed/real_case/enzyarena_kcat` and
`data/processed/real_case/enzyarena_km`. Those are converted and split but not
registered yet, so no published number depends on them; registering one
follows Steps 4 to 7.

Pick the layout that matches your data.

**A. The number depends only on the protein** (for example, melting
temperature):

```
dataset_id,protein_sequence,melting_point
my_enzyme_0001,MKLLAVFAGS...,62.4
my_enzyme_0002,MKQLAVWTGS...,58.1
```

**B. The number depends on the protein and a substrate** (for example,
activity on different sugars). Add a `smiles` column:

```
dataset_id,protein_sequence,smiles,activity
my_enzyme_0001,MASAPIEGTS...,OCC1OC(O)C(O)C(O)C1O,4.2
my_enzyme_0002,MASAPIEGTS...,OCC(O)C(O)C(O)C(O)C=O,0.3
```

**C. Each row is a mutant, and the number is the change it causes** compared
with its wild type (the unmutated enzyme), for example the change in melting
temperature. Add a `wildtype_sequence` column:

```
dataset_id,protein_sequence,wildtype_sequence,delta_tm
my_enzyme_P21S,MKLLSVFAGS...,MKLLPVFAGS...,-4.3
my_enzyme_F22L,MKLLPLFAGS...,MKLLPVFAGS...,0.8
```

**A or C for mutants?** If all your mutants come from **one** wild type, use
layout A with each mutant's sequence and its measured value; layout C would
give the same result. If they come from **several** wild types, use layout C.
Otherwise the model can score well just by recognizing which protein family a
row belongs to, without learning anything about the mutations.

Rules:

- **`dataset_id`**: a different value on every row. The simplest choice is
  your task name plus a counter (`my_enzyme_0001`) or plus the mutation
  (`my_enzyme_P21S`).
- **`protein_sequence`**: the full amino-acid sequence in capital letters, all
  on one line. No `>` header line, no spaces. In layout C, this is the full
  **mutant** sequence, not a mutation code like `P21S`. If you only have
  UniProt IDs, download the sequences from [uniprot.org](https://www.uniprot.org)
  first.
- **The number column**: name it whatever you like (`melting_point` above).
  Every row needs a number. Remember this column name for Steps 3 and 4.
- **`smiles`** (layout B only): one molecule per row, written as a SMILES
  string. You can copy it from [PubChem](https://pubchem.ncbi.nlm.nih.gov).
- **`wildtype_sequence`** (layout C only): the full wild-type sequence.
- Other columns are allowed. They are ignored unless you list them in Step 4.

## Step 2. Add your sequences to the sequence index

The file `data/processed/protein_pool.csv` gives every protein
sequence an ID, and embeddings are saved under that ID. Run this to add your
new sequences to the end of it. Change `my_enzyme` on the first line:

```bash
python - <<'EOF'
name = "my_enzyme"

import pandas as pd
index_path = "data/processed/protein_pool.csv"
index = pd.read_csv(index_path)
data = pd.read_csv(f"data/processed/real_case/{name}/{name}.csv")

new = data[["dataset_id", "protein_sequence"]].copy()
if "wildtype_sequence" in data:  # layout C: wild types need embeddings too
    wildtype = data[["dataset_id", "wildtype_sequence"]].rename(columns={"wildtype_sequence": "protein_sequence"})
    new = pd.concat([new, wildtype.assign(dataset_id=wildtype.dataset_id + "__wt")])
new["protein_sequence"] = new.protein_sequence.str.replace(r"\s", "", regex=True).str.upper()
new = new.drop_duplicates("protein_sequence")
new = new[~new.protein_sequence.isin(index.protein_sequence)]
assert not new.dataset_id.isin(index.sequence_id).any(), "Some dataset_id values are already used in the index"

new.rename(columns={"dataset_id": "sequence_id"}).to_csv(index_path, mode="a", header=False, index=False)
print(f"Added {len(new)} new sequences.")
EOF
```

Sequences that are already in the index keep their existing ID, so running it
twice is safe. Do **not** run `concat_unique_protein_sequences.py` to rebuild
the index: it can give existing sequences new IDs and break every stored
embedding.

## Step 3. Split the data

This puts 80% of the rows in a training set, 10% in a validation set and 10%
in a test set. It also rescales the number column to a mean of 0 and a
standard deviation of 1, which helps training.

```bash
python scripts/data_prep/generate_dataset_splits.py \
  --input data/processed/real_case/my_enzyme/my_enzyme.csv \
  --mode random \
  --random-split-unit protein \
  --standardize-col melting_point \
  --train-fraction 0.8 --val-fraction 0.1 --test-fraction 0.1 \
  --seed 42
```

Replace `my_enzyme` and `melting_point`. `--random-split-unit protein` keeps
every row carrying the same sequence in one split, so an enzyme measured
against several substrates does not end up half in training and half in test.

**If your rows are variants of a shared parent**, such as a variant library,
a mutational scan or pooled engineering campaigns, use `--mode similarity`
instead. Each mutant is its own sequence, so a random split scatters one
parent's variants across all three sets and the test score measures recall
rather than prediction. Similarity mode clusters sequences with MMseqs2 and
assigns whole clusters, which keeps relatives together:

```bash
  --mode similarity --identity-threshold 0.3 --mmseqs-coverage-threshold 0.8
```

It needs `mmseqs` on your PATH; binaries are at
<https://github.com/soedinglab/MMseqs2/releases>.

Either way it writes `my_enzyme_<mode>_80_10_10_standardized.csv` into the
same folder, with two new columns, `split` and `melting_point_standardized`.

For data where every row is an independent protein and you want the target
balanced across splits, use `--random-split-unit point --point-id-col
dataset_id --stratify-col melting_point` instead.

## Step 4. Register the task

Open `src/enzymado/data_loading/data_registry.py` and find `TASK_REGISTRY`.
Add this entry just before its closing `}`, after the last existing entry:

```python
    "my_enzyme": TaskConfig(
        name="my_enzyme",
        batch_kind=TaskType.PROTEIN_GLOBAL,
        label_kind=LabelType.SCALAR_REGRESSION,
        task_scope="real_case",
        dataset_paths={
            SplitType.RANDOM: _DATA / "real_case/my_enzyme/my_enzyme_random_80_10_10_standardized.csv",
        },
        target_column="melting_point_standardized",
        index_columns=("dataset_id",),
    ),
```

Then adjust it:

- Replace `my_enzyme` (four places) and `melting_point` with your names. Keep
  `_standardized` at the end of `target_column`.
- **Layout B only:** change `TaskType.PROTEIN_GLOBAL` to
  `TaskType.PROTEIN_LIGAND_GLOBAL` and add the line `ligand_column="smiles",`.
- **Layout C only:** change `TaskType.PROTEIN_GLOBAL` to
  `TaskType.MUTATION_DELTA_GLOBAL` and add the line
  `wildtype_sequence_column="wildtype_sequence",`. The model then sees the
  mutant's embedding minus the wild type's.
- **Optional:** to let the model use a numeric condition column such as
  temperature, add `numeric_feature_columns=("temperature",),`.

`task_scope="real_case"` keeps your dataset out of the property tasks. Do not
change it. Registering a task changes no published number: the leaderboards
rate only the tasks listed in `benchmark_card.json` (Step 5).

## Step 5. Describe the dataset

Do this step only for a dataset that will join the published case-study
leaderboard. Once your task is in the card, `compute_ratings.py` expects
results for it from all 75 competitors, so recomputing the leaderboards fails
until they exist. For your own experiment, skip to Step 6; the parity check at
the end of this step then reports your task as missing from the card, which is
expected.

Open `benchmark_card.json`. Inside the `"real_case"` section, copy an existing
entry and rename its key to `my_enzyme`. Copy `cleeveland_galox` for layout B,
or `erickson_petase` otherwise. In your copy, change:

- `description`: one sentence about the data.
- `dataset_id`: a short name for the dataset, for example `my_enzyme`.
- `label_units`: the unit of your number, for example `°C`.
- `path`: `data/processed/real_case/my_enzyme/my_enzyme.csv`.
- the `schema` columns and the `drop_na_in` list, so they name your columns.
  Delete any entry whose column your CSV does not have.
- `provenance`: where the data came from.

Then add your dataset's provenance and a description of its label to
`SOURCES` in `scripts/data_prep/card_metadata.py`, copying the
`erickson_petase` entry, and run:

```bash
python scripts/data_prep/enrich_dataset_cards.py
python scripts/ci/check_registry_card_parity.py
```

The first writes `dataset_card.json` next to your CSV and refreshes the cards
of the other datasets from their published files. The second must end with a
line starting `OK:`. If it prints
`real_case mismatch`, the name in Step 4 and the key in this step differ.

## Step 6. Compute the embeddings

Pick one option.

**Option A: on the Slurm cluster.** Log in, open `tmux` so the job survives
a disconnect, and run:

```bash
TASK=my_enzyme bash scripts/slurm/submit_task_embeddings.sh
```

It prepares everything, then submits one GPU job per embedding model (four by
default). Check progress with `squeue -u $USER`; when your jobs no longer
appear, they are done. Ignore the `sbatch ... train_task_pooled.slurm` command
it suggests at the end, because that script only runs published tasks. Go on
to Step 7.

**Option B: on your own computer.** This uses one model, ESM-2, and works
without a GPU, just more slowly. First install its environment from
`envs_files/venv-esm2.txt`, then run:

```bash
python scripts/data_prep/subset_sequence_index_for_task.py \
  -t my_enzyme -o data/processed/my_enzyme_sequences.csv
python scripts/data_prep/make_fasta_from_unique_sequences.py \
  --input data/processed/my_enzyme_sequences.csv \
  --output data/fastas/my_enzyme_sequences.fasta
python embedding_scripts/esm2_embedding_script.py \
  -i data/fastas/my_enzyme_sequences.fasta \
  -o embeddings/my_enzyme_sequences/esm2_650M_shards \
  --model facebook/esm2_t33_650M_UR50D --per_protein 0 \
  --mean_pooled_output data/embeddings_mean_pooled/my_enzyme_sequences/esm2_650M.safetensors
```

Then create the manifest `embeddings_my_enzyme_pooled.json` in the top folder:

```json
{
  "esm2_650M": {
    "path": "data/embeddings_mean_pooled/my_enzyme_sequences/esm2_650M.safetensors",
    "dim": 1280
  }
}
```

## Step 7. Train and score

Train the three standard heads (linear, MLP and XGBoost) on every embedding in
the manifest:

```bash
python scripts/run_benchmark.py embeddings_my_enzyme_pooled.json \
  --output_dir results_my_enzyme/predictions \
  --task_scope real_case --task_name my_enzyme
```

To train only some of them, add filters such as `--model_prefix mlp_` or
`--emb_name esm2_650M`. [REPRODUCE.md](REPRODUCE.md#2-train) lists them all.

Then compute the scores and their 95% confidence intervals:

```bash
python scripts/results/compute_metrics.py \
  --predictions-root results_my_enzyme/predictions \
  --statistics-root results_my_enzyme/statistics \
  --filter-task my_enzyme
python scripts/results/compute_bootstrap_metrics.py \
  --predictions-root results_my_enzyme/predictions \
  --statistics-root results_my_enzyme/statistics \
  --bootstrap-root results_my_enzyme/bootstrap \
  --filter-task my_enzyme --n-replicates 1000
```

Each result is a `metrics.json` file at:

```
results_my_enzyme/predictions/real_case/no_attention/random/<head>/my_enzyme/<embedding>/metrics.json
```

`<head>` is `linear`, `mlp` or `xgboost`, and `<embedding>` is a name from the
manifest. Under `"test"`, the file reports:

| Score | Meaning | Better when |
|---|---|---|
| `mae` | average size of the prediction error | lower |
| `rmse` | like `mae`, but punishes large errors more | lower |
| `r2` | share of the variation the model explains (1 is perfect) | higher |
| `kendall_tau` | how well the model ranks the proteins in the right order (1 is perfect) | higher |

Errors are in standardized units (Step 3), so an `mae` of 0.5 means the
prediction is off by half a standard deviation on average.

## Common mistakes

- **Leaving out `--task_scope real_case` in Step 7.** The task is skipped
  without any error message, and nothing is trained.
- **Leaving out `--output_dir` or the `--*-root` options in Step 7.** Your
  results are written into the published `results/` folder and mixed with it.
- **Passing `--allowed_ids`.** That list was made before your sequences
  existed, so it removes every row of your dataset.
- **Skipping Step 2.** Step 6 prints `WARNING: ... task sequences are absent
  from the sequence index`, and those proteins get no embeddings.
- **The warning `Device is changed from GPU to CPU`** on a computer without an
  NVIDIA GPU is harmless: XGBoost simply trains on the CPU.

## What Part 1 does not cover

These work in the benchmark, but need more than this guide explains. The
published tasks that use them are good examples to copy from
`data_registry.py` (see the task list in [REPRODUCE.md](REPRODUCE.md#the-tasks)).

- yes/no labels (like `solubility`), EC numbers (`ec_number`), or labels on
  individual residues (`active_sites`)
- the similarity split (`--mode similarity`, used by all property tasks),
  which keeps similar proteins out of the test set
- the attention heads, which need the much larger per-residue embeddings

---

# Part 2: Train your own model and get its weights

Part 2 needs some Python, and some PyTorch for adding an architecture.

## Get weights from the built-in heads

Every finished run saves its trained head in its run folder, next to
`metrics.json`. You get one from Part 1, Step 7, or from
[REPRODUCE.md, Level 2](REPRODUCE.md#level-2-rerun-one-task) for a published
task. The published `results/` folder has no weights; you have to train.

| Head | Files |
|---|---|
| `linear`, `mlp` | `<name>.pt` (PyTorch state dict), `training_checkpoint.bin` |
| attention heads | `<name>.safetensors`, `training_checkpoint.bin` |
| `xgboost` | `model.joblib` |

The weights alone are not enough, because each head also rescales its inputs
the way it learned in training. Load the whole trained head instead:

```python
from pathlib import Path
from enzymado.prediction.restore import restore_fitted_wrapper

run_dir = Path("results_my_enzyme/predictions/real_case/no_attention/random/mlp/my_enzyme/esm2_650M")
model = restore_fitted_wrapper(run_dir)
predictions = model.predict(x)
```

`x` must be built exactly as in training, from the same embedding model: one
mean-pooled embedding per row, plus the ligand descriptors and numeric
columns for layout B, or mutant minus wild type for layout C. `prepare(...)`
in `src/enzymado/pipeline/run_data.py` is the code that builds it. Predictions
come out in standardized units (Part 1, Step 3).

## Change a built-in head's settings

Each head's settings (layer sizes, learning rate, epochs and so on) are in
`src/enzymado/training/model_registry.py`. For example, a two-layer MLP is
`"hidden_dims": (1024, 512)` instead of `(2048,)` in
`_standard_mlp_regressor_entry`. No other code changes are needed. Use a new
`--output_dir`, because results go into the same folders as the original head.

## Add your own architecture

This works for tasks with one number or one yes/no per row (layouts A, B and
C, and published tasks like `ph_opt`, `kcat` or `solubility`):

1. **Write the network** as a PyTorch `nn.Module` in
   `src/enzymado/models/nn_blocks.py`, next to `MLPModel`. It receives a
   tensor of shape `(batch, input_dim)` and returns one of shape
   `(batch, 1)`.
2. **Wrap it.** In `src/enzymado/training/torch_standard_regression.py`, copy
   the class `MLPRegressorImplementation`, rename the copy (for example
   `MyNetRegressor`), and in its `fit` and `fit_loader` methods replace
   `MLPModel(...)` with your network. For yes/no tasks, do the same with
   `MLPClassifierImplementation` in `torch_standard_classification.py`. The
   copied class keeps early stopping, input rescaling and saving.
3. **Register it.** In `src/enzymado/training/model_registry.py`, import your
   class and add it to `MODEL_MAP["regression"]` (or `["classification"]`):

   ```python
   "mynet_regressor": _make_model_entry(
       MyNetRegressor,
       {"lr": 1e-3, "epochs": 100, "early_stopping_patience": 20},
       residue_aware=True,
   ),
   ```

   The name must **not** start with `linear_`, `mlp_`, `xgb_` or
   `attention_`, or it would be treated as a built-in head.
4. **Run only it** by adding `--model_prefix mynet_` to the Part 1, Step 7
   command. Its results and weights go into a `<head>` folder named
   `mynet_regressor`.

## What is and is not possible

| You want to | Possible? |
|---|---|
| a new head for one number or yes/no per row (protein, ligand or mutation tasks) | Yes, as above. |
| a new head for `ec_number` | Yes, with more work: register it in `STRUCTURED_MODEL_MAP["multilabel_ec"]` and predict all four EC levels. Start from `ECMultiHeadClassifierTorch`. |
| a new head for per-residue tasks (`active_sites`, `binding_site`) | Yes, with more work: register it in `STRUCTURED_MODEL_MAP["residue_binary_vector"]` and implement `fit_loader` / `predict_loader`. Start from `ResidueBinaryClassifierTorch`. |
| a new protein language model | Yes, without changing training code: compute its embeddings (Part 1, Step 6) and add it to the manifest. |
| fine-tune the protein language model itself | **No.** The benchmark always trains on frozen, precomputed embeddings. |
| put a new head on the published leaderboards | **No.** The ratings compare the five published heads only. Use the metrics and confidence intervals. |
