# Agentic ingestion workflow

How a new case-study dataset enters Enzymado, from a published paper to a seat on
the case-study leaderboard. The workflow can be followed by a person or by a coding
agent; 28 of the 36 published case studies were added this way.

This file is the specification: what each step guarantees, which code enforces it,
and how to recover when a step fails. For a step-by-step walkthrough with copyable
commands aimed at an outside user, read [add_data.md](../add_data.md). For rerunning
published results, read [REPRODUCE.md](../REPRODUCE.md).

Terminology: the code says `real_case`, the paper and the cards say *case studies*.
They are the same thing.

## Invariants

These hold across every step. Breaking one invalidates stored results, not just the
dataset being added.

1. **The sequence index is append-only.** `data/processed/protein_pool.csv` maps a
   sequence id to a sequence, and every embedding store is keyed by that id. New
   sequences are appended; existing ids are never renumbered. Rebuilding the pool
   with `scripts/data_prep/concat_unique_protein_sequences.py` re-derives ids across
   all datasets and can rename them, which silently invalidates the 4.6 TB of stored
   embeddings. That script is for a cold rebuild only.
2. **Only new sequences are embedded.** A sequence already in the pool keeps its id
   and its stored vectors, so ingestion costs embedding time proportional to what is
   new, not to the whole pool.
3. **Converters are deterministic.** One script per study, raw input to processed
   CSV, no manual edits of the processed file. Any correction to the source data is
   made in the script and recorded in the dataset card, so the reconciliation from
   published rows to our rows stays auditable.
4. **The leaderboard is all-or-nothing.** Once a task is in `benchmark_card.json`,
   `scripts/results/compute_ratings.py` expects results for it from all 75
   competitors. Register the task in the card only when you intend to run the full
   grid; until then, train it outside the card (see Step 6).
5. **Ratings are relative.** Adding a dataset changes every case-study rating,
   because the Bradley-Terry fit is over the pool of datasets and competitors. Each
   leaderboard file records the datasets it was fitted on.

## Step 1. Parsing

Extract from the source publication and its supplement: wild-type sequences, variant
definitions, substrates, assay conditions, and the measured values.

`scripts/data_prep/real_cases/utils.py` provides the primitives the converters share:

| Need | Function |
| --- | --- |
| Clean a sequence (whitespace, case, gaps, stop symbols) | `normalize_sequence`, `clean_protein_sequence` |
| Apply mutation codes to a parent sequence | `apply_mutations`, `parse_hgvs_missense` |
| Translate a deposited DNA library | `translate_dna` |
| Canonicalize substrate SMILES with RDKit | `canonicalize_smiles`, `canonicalize_multi_smiles` |
| Parse numbers, including the `ND`/`NS`/`NA`/`-` missing tokens | `parse_float` |

Mutation codes are applied to the parent and, by default (`verify_wt=True`), checked
against the parent residue at that position, so a renumbered or off-by-one mutation
table fails loudly instead of producing a wrong sequence.

## Step 2. Feasibility gate

A candidate dataset is admitted when all of the following hold.

**Readout.** The measured quantity must be attributable to the enzyme under study.
Direct catalytic and biophysical assays qualify. So do selections and display
screens in which the enzyme's own reaction is the selected step: the AmiE amidase
growth scans (growth on an amide as sole nitrogen source), the TrpB growth scan
(growth coupled to tryptophan synthesis), the D-amino acid oxidase EP-Seq screens,
and the PLpro abundance and activity scans. Whole-cell screens whose readout depends
on cellular processes unrelated to the enzyme do not qualify. Admitted datasets whose
readout still depends on expression or abundance are marked
`is_expression_decoupled = false` in their dataset card, so analyses can separate
them (Appendix H3 of the paper).

**Size and shape.** Enforced in code by `utils.validate_dataset_rows`, called from
`utils.write_processed_dataset`:

| Rule | Value |
| --- | --- |
| Rows | 300 to 20,000 (`min_rows`, `max_rows`) |
| Distinct protein sequences | at least 10 (`min_unique_seqs`) |
| Sequence validity | at least 30 residues, 20 canonical amino acids only (`is_valid_sequence`) |
| `dataset_id` | unique within the dataset |

Three converters predate this validator and write through `utils.write_csv`
directly: `align_tournament.py`, `cleveland_galox.py` and `erickson_petase.py`. One
published dataset sits below the 300-row bound as a result,
`align_glucosidase_melting` with 106 rows, kept because it is one endpoint of the
Protein Engineering Tournament suite. New converters should call
`write_processed_dataset` and inherit the bounds.

**Overlap.** The candidate must not duplicate a property dataset. A variant library
of an enzyme that also appears in a property task is acceptable; re-importing rows
that are already in a property task is not.

Datasets that fail the gate are not converted. Record the candidate and the reason
in the pull request or ingestion notes, so the same paper is not re-evaluated from
scratch later. Categories deferred so far: whole-cell growth screens with no
enzyme-attributable readout, coarse ordinal family screens, and libraries above
20,000 rows (such as the 160,000-variant four-site TrpB library).

## Step 3. Conversion

Write one deterministic script per study under `scripts/data_prep/real_cases/`, with
a CLI entry point that accepts `--output-root` through `utils.parse_output_root`:

```bash
python scripts/data_prep/real_cases/<study>.py
```

The script maps the parsed data to a layout (Appendix M.2 of the paper):

| Layout | Columns it must produce | Builder |
| --- | --- | --- |
| A, protein only | `dataset_id`, `protein_sequence`, target (+ condition columns) | `utils.build_layout_a_rows` |
| B, protein and substrate | the same plus `smiles` | `utils.build_layout_b_rows` |
| C, mutation | the same plus `wildtype_sequence` | written by the converter |

Replicates are merged by `utils.aggregate_replicates`, which averages the values
sharing a key (sequence, substrate and conditions). Targets spanning two or more
orders of magnitude are converted with `utils.compute_log10`, whose `offset`
argument keeps zeros finite; the offset used by each dataset is listed in Appendix H
of the paper and repeated in the dataset card. The script writes
`data/processed/real_case/<task>/<task>.csv`.

## Step 4. Sequence index

Append the new sequences to `data/processed/protein_pool.csv` with the snippet in
[add_data.md](../add_data.md#step-2-add-your-sequences-to-the-sequence-index). It
normalizes and deduplicates against the pool, keeps existing ids, and asserts that
no new `dataset_id` collides with an id already in the index. Layout C datasets must
append the wild-type sequences too, under a `__wt` suffix, because the mutation head
subtracts the wild-type embedding.

Running the snippet twice is safe: sequences already in the pool are filtered out.

## Step 5. Splitting

Case studies use one random split by row, stratified on ten bins of the target, at
80/10/10 with seed 42. For a dataset that will join the published leaderboard, add
it to the registry (Step 6) and regenerate with:

```bash
python scripts/data_prep/generate_real_case_splits.py
```

which reads every `real_case` task from `TASK_REGISTRY` and calls
`dataset_splitting.split_dataset_csv` with `random_split_unit="point"`,
`point_id_col="dataset_id"`, `stratify_bins=10` and `seed=42`. It writes
`<task>_random_80_10_10_standardized.csv` plus a
`*.splits.metadata.json` sidecar that records the strategy, the fit split, and the
standardization statistics. Regression targets are standardized on the training
split only.

Row-stratified splitting puts the same sequence in training and test when an enzyme
is measured against several substrates or conditions. That is a known property of
this split, not a bug: it is why the paper separates the 22 single-enzyme variant
libraries from the 14 multi-row panels (Section 3.5).

## Step 6. Registration

1. Add the task to `_REAL_CASE_TASKS` in
   `src/enzymado/data_loading/data_registry.py`. `TASK_REGISTRY` is the runtime
   source of truth: task scope, layout, columns, split files and target column.
2. Add the matching entry to the `real_case` block of `benchmark_card.json`, and
   the dataset's provenance to `SOURCES` in `scripts/data_prep/card_metadata.py`.
3. Generate both cards and check parity:

```bash
python scripts/data_prep/enrich_dataset_cards.py
python scripts/ci/check_registry_card_parity.py
```

The parity script compares the task-name sets of the registry and the card and must
end with a line starting `OK:`. Nothing runs it automatically today, so run it here
and before opening a pull request.

The computed half of the dataset card is written from the processed and split files;
the curated half, in `docs/data_cards/<task>.md` and
`docs/data_cards/<task>_dataset_card_enriched.json`, records what cannot be computed:
the assay and its STRENDA conditions, the replicate design, the label noise the
source reports, and a row-by-row reconciliation from the published table to our CSV.
Every publication-derived field carries a verbatim quote of at most 250 characters
with its page number.

## Step 7. Embedding

Only the appended sequences are embedded:

```bash
TASK=<task> bash scripts/slurm/submit_task_embeddings.sh
```

It prepares the manifest and submits one GPU job per embedding model. For a dataset
that will join the leaderboard, all 14 PLMs plus the `no-plm` reference must be
present before the grid can be considered complete. `add_data.md` Step 6 documents a
single-model CPU path for a local experiment.

## Step 8. Training and evaluation

Train the five architectures (linear, MLP and XGBoost on mean-pooled vectors; linear
and MLP with attention pooling) and score them:

```bash
python scripts/run_benchmark.py <manifest>.json \
  --output_dir results/predictions --task_scope real_case --task_name <task>

python scripts/results/compute_metrics.py --filter-task <task>
python scripts/results/compute_bootstrap_metrics.py --filter-task <task> --n-replicates 1000
```

Attention-pooling runs go through `scripts/slurm/run_training_attention_cluster.slurm`
on the cluster, since they need a GPU for hours rather than seconds. Each run writes
its predictions, a `metrics.json`, and a `run_manifest.json` recording the dataset
hash, the embedding-store inventory hash, the source-tree and lockfile hashes, the
package versions and the seed. Editing any `.py` under `src/` or `scripts/` changes
the source hash and prevents a checkpoint from resuming, by design.

## Step 9. Rating

Refit the case-study leaderboard:

```bash
python scripts/results/compute_ratings.py
```

`--n-battles` must equal the number of bootstrap replicates produced in Step 8.
`compute_ratings.py` checks the shape of every bootstrap array and fails if a file
holds a different number of values, so a mismatch is caught rather than averaged
away. The output records the datasets and the rating configuration it used, and both
boards (properties and case studies) are fitted separately.

## Failure recoveries

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Some dataset_id values are already used in the index` | The converter's ids collide with ids already in the pool | Prefix the ids with the task name in the converter and re-run Steps 3 and 4 |
| A training run reports sequences missing from the embedding store | Step 4 was skipped, or Step 7 ran before the sequences were appended | Append, re-embed only the missing ids, re-run the task |
| `check_registry_card_parity.py` prints `real_case mismatch` | The registry key and the card key differ | Rename one so both match, then regenerate the cards |
| `compute_ratings.py` fails on a missing competitor | The task is in the card but the grid is incomplete | Finish the 75 runs, or remove the task from the card while iterating |
| `Task ... has N rows, outside bounds` | The gate of Step 2 rejected the dataset | Do not relax the bound in the converter; re-evaluate the candidate |
| A checkpoint refuses to resume | The source tree changed between launch and resume | Finish the run with the original tree, or restart it |

## What this workflow does not cover

- Property datasets. They are built by `scripts/data_prep/propertyprep.py` and the
  source-specific converters, under a different split protocol (random and
  similarity, 80/10/10, grouped by protein).
- New predictor heads. See `add_data.md` Part 2.
- New embedding models, which require embedding the whole shared sequence index
  rather than one dataset's sequences.
