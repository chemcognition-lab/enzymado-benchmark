# Enzymado

**A reproducible benchmark for protein language model embeddings on enzyme property prediction.**

Enzymado asks a narrow, practical question: *if you take a frozen protein language model
and train a simple head on top of its embeddings, how well does that work for real enzyme
engineering tasks?* It answers it the same way for every model, on every dataset, with
every split, so the numbers are comparable.

<p align="center">
  <img src="assets/enzymado_evaluation_diagram.png"
       alt="Enzymado evaluation design: 10 property datasets and 8 real-case datasets, crossed with 14 protein language models, 2 split types, and 5 baselines, for 1,960 model-dataset evaluations."
       width="820">
</p>

## Where to start

| You want to | Read |
|---|---|
| Rerun our benchmark and get the published numbers, from recomputing the leaderboards on a laptop to retraining everything on a cluster | [REPRODUCE.md](REPRODUCE.md) |
| Run only some tasks, embedding models or heads, without the full benchmark | [REPRODUCE.md, Level 2](REPRODUCE.md#level-2-rerun-one-task) |
| Benchmark your own dataset | [add_data.md, Part 1](add_data.md#part-1-add-a-dataset) |
| Get trained model weights, or train your own architecture | [add_data.md, Part 2](add_data.md#part-2-train-your-own-model-and-get-its-weights) |

Requires Python ≥3.12,<3.14, [uv](https://docs.astral.sh/uv/) and **Git LFS**. Without
Git LFS the datasets arrive as pointer files and every run fails.

```bash
git lfs install
git clone <repo-url> && cd enzymado-benchmark
uv sync --extra train
uv run pytest        # optional: check that the installation works
```

---

## What it evaluates

| axis | values |
|---|---|
| **Property datasets** | 10 - `binding`, `binding_site`, `active_sites`, `ec_number`, `solubility`, `kcat`, `km`, `ph_opt`, `dtm`, `ddg` |
| **Real-case datasets** | 8 - amylase (activity / expression / stability), glucosidase (activity / expression / melting), GalOx, PETase. All regression. |
| **Protein language models** | 14 - see the table below |
| **Split types** | 2 - `random` and `similarity` (MMseqs2 at 30% minimum sequence identity, 80/10/10, so test proteins are not near-duplicates of training ones). Real-case datasets are `random` only. |
| **Baselines** | 5 - linear, MLP and XGBoost without attention, plus linear and MLP over learned attention pooling. How the embedding reaches the head depends on the task: protein-level tasks mean-pool it, residue-level tasks (`active_sites`, `binding_site`) feed one embedding per residue with no pooling at all. |

The 14 embedding models, with the manifest key each one is addressed by:

| family | manifest key | dim |
|---|---|---|
| ESM-2 | `esm2_650M`, `esm2_3B` | 1280, 2560 |
| ESM-3 | `esm3_open` | 1536 |
| ESM-C | `esmc_300m`, `esmc_600m` | 960, 1152 |
| Ankh | `ankh_base`, `ankh_large` | 768, 1536 |
| Ankh3 | `ankh3_large`, `ankh3_xl` | 1536, 2560 |
| ProtT5 | `prot5_xl_uniref50`, `prot5_xl_half_uniref50` | 1024, 1024 |
| ProstT5 | `prost5` | 1024 |
| ProFluent-E1 | `profluent_e1_300m`, `profluent_e1_600m` | 1024, 1280 |

`dtm` and `ddg` are **mutation tasks**: each row is a mutant of a known wild-type
protein, and the head sees the mutant's embedding minus the wild type's, so it has to
predict the effect of the mutation rather than recognize the protein. The six
amylase and glucosidase real-case datasets are variant libraries of a single enzyme.

That grid is **1,960 model–dataset evaluations**: 1,400 on properties (10 × 14 × 2 × 5)
and 560 on real cases (8 × 14 × 1 × 5).

Embeddings are **frozen**. Nothing fine-tunes the language model — the benchmark measures
what the representation already carries, not what a full fine-tune could extract.

Every model competes on the **same proteins**. The 14 embedding stores do not cover the
sequence index identically, so `scripts/embeddings/build_embedding_intersection.py`
computes their intersection once and training restricts every run to it via
`--allowed_ids`. The published results use **269,409 shared sequence IDs**, with 10,235
excluded; `results/embedding_intersection/report.tsv` records the per-model coverage that
produced that set.

Every published run uses **training seed 42**.

---

## How a run works

The pipeline is a straight line, and each stage has one job:

```
data_loading  →  training  →  prediction  →  evaluation
```

1. **`data_loading`** turns a task name into a ready-to-train bundle: rows from the dataset
   CSV, embeddings pulled from safetensors shards, labels built per task type, and the
   split assignment. Residue-level tasks stream embeddings lazily so that a ~390 GB store
   never has to fit in RAM.
2. **`training`** takes that bundle plus an architecture from `models`, runs the training
   loop, keeps the epoch with the best validation score (Kendall tau for regression,
   AUPRC for classification, validation loss for EC and residue tasks), and returns a
   fitted estimator.
3. **`prediction`** takes what training produced and writes predictions for each split as
   compressed safetensors, alongside the model weights and the run's hyperparameters.
4. **`evaluation`** reads those prediction artifacts and produces metrics, bootstrap
   confidence intervals, and finally the global leaderboards.

`pipeline` is the orchestration layer that resolves a CLI selection into concrete runs and
drives those four stages. Two packages sit underneath them: **`models`** holds only
architectures, and **`artifacts`** owns the result-tree layout, serialization, and
provenance. Dependencies run one way, `evaluation` never imports `training`.

Three more packages build the inputs rather than consume them, and run only when datasets
are regenerated: **`propertyprep`** (source-specific ingestion for BRENDA, BioLiP,
SwissProt, FireProtDB, CatPred, Proteinea and UniProt), **`splitting`** (random and
MMseqs2 similarity splits, plus their metadata), and **`dataset_card`** (per-dataset
statistics, clustering, and UMAP figures).

### Metrics

| task type | metrics |
|---|---|
| classification | `accuracy`, `precision`, `recall`, `f1`, `auroc`, `auprc` |
| regression | `mae`, `rmse`, `r2`, `kendall_tau` |

Classification metrics are computed from per-row sufficient statistics, confusion counts
for the threshold metrics, score histograms for AUROC and AUPRC. That keeps each of the
1,000 bootstrap replicates an O(1) aggregation instead of a full recomputation, which is
what makes interval estimation tractable on tasks with 10⁸ predictions — `ec_number` alone
scores 1.36 × 10⁸ test predictions.

### Ratings

Per-task metrics become two global leaderboards (`properties` and `real_case`) via
**Bradley–Terry maximum likelihood** over pairwise outcomes, anchored at 1500. Defaults:
1,000 battle rounds, Kendall tau weighted double among regression metrics, and property
splits weighted 55% similarity / 45% random.

Every registered metric is computed and bootstrapped for every run, but not every metric
cell is allowed to cast battles:

- **`r2` carries rating weight 0.** Within a bootstrap replicate SS_tot is fixed, so `r2`
  is a strictly monotone function of `rmse`; weighting both would count squared error
  twice. It is still reported per model.
- **`benchmark_card.json:rating_exclusions` withholds degenerate cells.** `accuracy` is
  withheld below 5% positive-class prevalence, and `kendall_tau` / `auroc` are withheld
  when the best competitor on that task and split fails to clear 0.1 / 0.55. In the
  published results this withheld 6 of 90 property metric cells — `accuracy` on
  `active_sites`, `binding_site` and `ec_number`, in both splits, where prevalence runs
  from 0.07% to 2.5% — leaving 84. No real-case cell was withheld. Each leaderboard's
  JSON lists exactly which cells it dropped and why.

---

## Extending it

To add a dataset or a model, follow [add_data.md](add_data.md). To add a metric, declare
it in `src/enzymado/evaluation/metrics_registry.py` with its task type and direction.
Classification metrics must be derivable from confusion counts or score histograms so
the bootstrap stays fast.

---

## Results layout

```
results/
  predictions/            per-run artifacts: predictions, weights, hparams.json, metrics.json
  bootstrap/              resampled metric distributions
  ratings/                the two global leaderboards
  embedding_intersection/ the shared sequence-ID set every run is restricted to
  analysis/               training-runtime breakdown
  figures/                cluster diversity and UMAP projections
```

Runs are addressed by
`<scope>/<pool_type>/<split>/<model_family>/<task>/<embedding_model>/`, and each run
directory carries a `hparams.json` fingerprinting its dataset paths, task config,
hyperparameters and seed, so a published number can be traced back to the exact inputs
that produced it.

Only the numbers are published. Prediction tensors, model weights and per-epoch logs stay
local — they are gigabytes of LFS, and every figure and leaderboard is recomputable from
the tracked `metrics.json`, `hparams.json` and bootstrap distributions. Per-dataset cards
live next to the data, at `data/processed/properties/<task>/dataset_card.json`.

---

## License

MIT — see [LICENSE](LICENSE).
