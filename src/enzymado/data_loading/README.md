# Adding your own data to the benchmark

This is the practical walkthrough for getting an outside dataset — your own
experimental measurements, or a published dataset you want to test the
foundation models against — registered and running through this pipeline.
Everything here was verified by actually doing it once; see the worked
example at the bottom.

## Step 0: pick a scope

Every task lives in `TASK_REGISTRY` in [`data_registry.py`](data_registry.py),
tagged with a `task_scope`:

- **`"properties"`** — the large aggregated benchmarks (BRENDA-derived `kcat`,
  FireProtDB-derived `ddg`, etc.). Adding here means your data becomes a
  permanent part of the shared corpus everyone's numbers are compared against.
- **`"real_case"`** — one specific study's data, standing on its own row on
  the leaderboard, compared *next to* the properties tasks but never merged
  into them. `erickson_petase` and `cleeveland_galox` are existing examples.

**If you're testing your own data, you almost certainly want `"real_case"`.**
It's non-destructive — nothing about the existing tasks changes.

## Step 1: get your data into one CSV

Required columns, minimum:

| Column | What it is |
|---|---|
| `protein_sequence` | The raw amino-acid string — single-letter codes, **no** FASTA `>header` line, no line breaks (e.g. `MSTAGKVIKCKAAVLW...`). Include whatever was actually expressed (e.g. a His-tag), since the embedding should reflect the real construct. |
| an id column (e.g. `dataset_id`) | Any label you want — a row number, a variant name from your paper. It's bookkeeping only; it does **not** need to be a real UniProt accession. |
| your target column | The value you measured — a number for regression, or a yes/no for a binary call. |
| `smiles` (only if relevant) | Required if what you measured depends on a substrate/ligand (e.g. Km, kcat, specific activity) — the substrate's SMILES string. |

**If you only have UniProt IDs, not sequences:** fetch them —
`https://rest.uniprot.org/uniprotkb/<ID>.fasta` returns the sequence for any
ID, free, no auth. Strip the `>` header line and keep only the sequence
letters.

Why `protein_sequence` matters more than any ID: the embedding lookup
(`EmbeddingStore.lookup_row` in [`features.py`](features.py)) is keyed
primarily by the **sequence content itself** (normalized: whitespace
stripped, uppercased), not by any ID column. An ID with no matching sequence
will never resolve to an embedding.

## Step 2: generate the split + standardized file

Run the repo's actual splitting tool — don't hand-roll this part, it also
computes the z-scored target column your `TaskConfig` will point at:

```bash
python src/dataset_splitting.py \
  --input data/processed/real_case/<your_name>/<your_name>.csv \
  --mode random \
  --train-fraction 0.8 --val-fraction 0.1 --test-fraction 0.1 \
  --standardize-col <your_target_column> \
  --seed 42
```

This writes two files: `..._random_80_10_10_standardized.csv` (what your
`TaskConfig` will point at — has `split` and `<column>_standardized` columns
added) and a `.splits.metadata.json` sidecar recording exactly what it did
(fit statistics, row counts per split, etc.) — worth reading if a result
looks off.

**Small datasets split unevenly.** Rows are grouped by protein sequence by
default (so the same enzyme can't leak across train/val/test) — with only a
handful of distinct proteins, val/test can end up with just one protein
each, which is a noisy generalization signal. More distinct proteins is the
fix, not different flags.

## Step 3: register the task

Add one entry to `TASK_REGISTRY` in [`data_registry.py`](data_registry.py).
Pick the shape that matches what you measured:

| `batch_kind` | Use when... |
|---|---|
| `PROTEIN_GLOBAL` | Just a protein property, no substrate involved (e.g. solubility, pH optimum) |
| `PROTEIN_LIGAND_GLOBAL` | The value depends on a substrate/ligand (kcat, Km, specific activity) — needs `ligand_column` |
| `MUTATION_DELTA_GLOBAL` | You're measuring a mutation's *effect* (needs both `wildtype_sequence_column` and the mutant `protein_sequence`) |
| `PROTEIN_RESIDUE` | Per-residue labels (e.g. which residues are active-site) |

| `label_kind` | Use when... |
|---|---|
| `SCALAR_REGRESSION` | A continuous number |
| `BINARY` | A yes/no call |
| `MULTILABEL_EC` | Hierarchical EC-number labels |
| `RESIDUE_BINARY_VECTOR` | Per-residue binary labels |

```python
"your_name_property": TaskConfig(
    name="your_name_property",
    batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,   # match your data's shape
    label_kind=LabelType.SCALAR_REGRESSION,
    task_scope="real_case",
    dataset_paths={
        SplitType.RANDOM: _DATA / "real_case/your_name/your_name_random_80_10_10_standardized.csv",
    },
    target_column="<your_target_column>_standardized",
    ligand_column="smiles",          # omit if PROTEIN_GLOBAL
    index_columns=("dataset_id",),
),
```

## Step 4: generate embeddings for your sequences

Nothing downstream works until embeddings exist for your specific
sequences. Two things to know:

- **A Mac is genuinely fine for this**, not just a fallback — the scripts in
  [`embedding_scripts/`](../../embedding_scripts) explicitly check for
  `torch.backends.mps.is_available()` and use Apple Silicon's GPU directly.
  The SLURM/cluster setup elsewhere in this repo exists for *scale* (running
  all 19 model variants across hundreds of thousands of sequences), not
  because the computation itself requires it.
- **You don't need to run all 19 models.** That's the whole point of this
  being a comparison benchmark — pick one fast one to confirm things work
  (`esm2_8M`, tiny and quick) and/or a small spread across families if you
  want a real signal about which embedding suits your task
  (`esm2_650M`, `ankh_base`, `prot5_xl_half_uniref50`, `esmc_300m` is a
  reasonable spread).

Your sequences also need to be folded into
`data/processed/protein_sequences_unique.csv` (the deduplicated master list
every embedding run reads from) — regenerate it with
[`scripts/data_prep/concat_unique_protein_sequences.py`](../../scripts/data_prep/concat_unique_protein_sequences.py)
after adding your task.

## What this pipeline does *not* give you

- **No inference-only path.** `scripts/run_benchmark.py` always trains and
  evaluates together — there's no "load an existing checkpoint and just
  score new data" script. Testing your data means training a small
  probe (linear/MLP/XGBoost) on it, not scoring it against something
  pre-trained.
- **Nothing is shared across tasks except the embeddings.** Your task gets
  its own probe, trained only on its own data. It never touches the model
  trained for `kcat`, `solubility`, or anything else.

## A real worked example

`data/processed/real_case/enzyarena_km/` is a real, working example built
exactly this way — 2,250 Km measurements across 110 enzyme families, taken
from EnzyArena's condition-matched mutagenesis campaigns, split via the real
`dataset_splitting.py`, and registered as `enzyarena_km` in `TASK_REGISTRY`.
Read alongside its `TaskConfig` entry as a concrete template — every step
above is what actually produced it.
