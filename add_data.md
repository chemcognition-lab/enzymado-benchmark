# Adding data to the benchmark

Use this guide when adding a new dataset to the benchmark.

## Quick checklist

1. Put your data in one CSV with:
   - `protein_sequence` — required on every row, no FASTA header, no line breaks
   - one column, `dataset_id`, with a unique name to identify each row
   - your target column
   - `smiles` — required if and only if the task is `PROTEIN_LIGAND_GLOBAL`

### `dataset_id`

Becomes the row's `sample_id`, which is what error messages point at. Give it a
unique value per row — this matters most for ligand tasks, where the same
sequence appears once per substrate, so the sequence alone cannot identify a row.

Use a readable composite key when the row has one (`cleeveland_galox` uses
`exegalox_d_galactose`, `nhase_substrate_scope` uses `ptnhase_wt_acrylonitrile`),
or a zero-padded counter otherwise (`align_alpha_amylase_000001`). Any string
works; it does not need to be a real accession.

### `smiles`

Required if and only if you register the task as `PROTEIN_LIGAND_GLOBAL`. Every
row must have one, and it must parse under RDKit. `PROTEIN_GLOBAL` tasks have no
`smiles` column at all.

2. Split + standardize it:

```bash
python src/dataset_splitting.py \
  --input data/processed/real_case/<name>/<name>.csv \
  --mode random \
  --train-fraction 0.8 --val-fraction 0.1 --test-fraction 0.1 \
  --standardize-col <target_column> \
  --seed 42
```

This creates a file like:
`<name>_random_80_10_10_standardized.csv`

3. Register the task in `TASK_REGISTRY` in `src/data_loading/data_registry.py`.

```python
"your_task_name": TaskConfig(
    name="your_task_name",
    batch_kind=TaskType.PROTEIN_LIGAND_GLOBAL,
    label_kind=LabelType.SCALAR_REGRESSION,
    task_scope="real_case",
    dataset_paths={
        SplitType.RANDOM: _DATA / "real_case/your_task_name/your_task_name_random_80_10_10_standardized.csv",
    },
    target_column="<target_column>_standardized",
    ligand_column="smiles",
    index_columns=("dataset_id",),
),
```

4. Add the dataset metadata to the root `benchmark_card.json` under `"real_case"`.

5. Generate a real-case dataset card:

```bash
python scripts/data_prep/create_real_case_dataset_cards.py
```

6. Add the sequences to `data/processed/protein_sequences_unique.csv` and generate embeddings.

7. Run the benchmark:

```bash
python scripts/run_benchmark.py embeddings_models_output.json \
  --task_scope real_case \
  --task_name your_task_name \
  --split_variant random
```

## Rules of thumb

- Use `task_scope="real_case"` for new or personal datasets.
- `protein_sequence` is the real key; IDs alone are not enough.
- If you have only UniProt IDs, fetch the FASTA and strip the header.
- If the measurement depends on a ligand/substrate, include `smiles`.
- You need embeddings for the new sequences before evaluation will work.

## Minimal working pattern

- CSV with sequences + target + optional `smiles`
- split/standardized CSV
- `TaskConfig` in `TASK_REGISTRY`
- metadata in `benchmark_card.json`
- embeddings generated
- benchmark run

That is the full path for adding data in this repo.
