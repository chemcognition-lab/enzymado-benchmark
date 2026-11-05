---
card_type: dataset
task_id: catpred_kcat
dataset_id: catpred_kcat
property_name: kcat
track: properties
task_type: protein_ligand_global
label_kind: scalar_regression
layout: Layout B
doi: 10.1038/s41467-025-57215-9
license: CC BY-NC-ND 4.0 (article); underlying BRENDA/SABIO-RK records licensed separately
source_pdfs:
  - docs/pdfs/catpred_boorla2025_natcommun.pdf (main, pp. 1-17)
  - docs/pdfs/catpred_boorla2025_suppinfo.pdf (si, pp. 1-20; combined pp. 18-37)
noise_regime: inter_lab_aggregated
label_noise_floor_mae_log10: ~0.38-0.71
label_noise_floor_fold_error: ~2.4-5.2x
scope_kind: global_multi_family
computed_statistics_ref: data/processed/properties/kcat/dataset_card.json#/statistics
---

# `catpred_kcat` — Enzyme turnover number ($k_{\text{cat}}$)

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, MMseqs2 diversity metrics, and per-split label moments are maintained programmatically in [`data/processed/properties/kcat/dataset_card.json`](../../data/processed/properties/kcat/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Three framings govern what models can and cannot learn here:

1. **It is a literature aggregate, not a single measurement.** Each label was produced by an independent lab, under its own conditions and assay protocol, curated into BRENDA or SABIO-RK, and then collapsed to one scalar. The quantity being modelled is not "the $k_{\text{cat}}$ of this enzyme on this substrate"—there is no such condition-independent constant—but *"the maximum value anyone has published for this pair, under conditions nobody recorded in the feature vector."* Section 1 quantifies what that costs.
2. **The target is an upper envelope, not a central estimate.** Duplicate measurements of the same enzyme–substrate pair collapse by **maximum** for $k_{\text{cat}}$ (whereas geometric mean is used for $K_{\text{m}}$ and $K_{\text{i}}$—do not carry that intuition over). An enzyme characterised across twelve papers receives the best of twelve draws; one characterised once gets a single draw. Upward bias scales with how heavily studied an enzyme is, which correlates with how likely its family is to be well-represented in training.
3. **Conditions are absent from the model inputs, not constant.** Temperature and pH survive as provenance columns in the CSV (with 841 and 710 NaNs respectively) but are excluded from model inputs (`numeric_feature_columns = ()`). Any model trained here marginalises over an unobserved condition distribution. This is the single largest irreducible error source, and the authors state so directly.

**Practical consequences for use:**
- Report generalisation metrics on the **`similarity_mmseqs2_80_10_10`** split (`--min-seq-id 0.3`). The `random_80_10_10` split groups by UniProt accession rather than homology, so a test enzyme may have a $>99\%$-identical ortholog in training; numbers from `random` are not cross-family generalisation numbers.
- Always pair $R^2$ with MAE or Kendall's $\tau$ on the similarity split: its test-set target standard deviation is **0.8066** (vs. **1.0000** in train; see `dataset_card.json#/statistics/label_distribution_by_method`), so variance-normalised metrics like $R^2$ read lower partly because of the split's narrower denominator.
- Beating $\sim 0.6\text{ }R^2$ or $\sim 0.7\text{ }\log_{10}\text{ MAE}$ on held-out data is approaching the data noise ceiling rather than an architectural limitation.
- Do not use models trained on this task to rank point mutants: engineered and mutant enzymes were explicitly filtered out upstream.

---

## 1. Label noise and the performance ceiling

**Noise is the defining property of this dataset.** The source paper states the ceiling explicitly: *"prediction accuracy may be limited by the inherent noise in the training datasets."* `[Source: Main, Page 8]` and *"The performance of machine learning models is predominantly limited by the experimental uncertainty in the datasets (i.e. aleatoric uncertainty)"* `[Source: Main, Page 11]`.

### 1.1 Where the noise comes from

The authors' own enumeration: *"measurements made at different assay conditions and/or using different protocols, systematic/random experimental errors and curation errors in databases, can all contribute to aleatoric uncertainty"* `[Source: Main, Page 9]`.

Decomposed in rough order of expected magnitude:

| Noise source | Physical or curation mechanism | Mitigable? |
|---|---|---|
| **Unrecorded $T$ and pH** | $k_{\text{cat}}$ roughly doubles every $10\,^\circ\text{C}$ up to $T_{\text{opt}}$ and follows a bell-shaped pH profile; the same pair assayed at $25\,^\circ\text{C}$ vs. $60\,^\circ\text{C}$ collapses to one unconditioned row | No — annotations are incomplete upstream (`numeric_feature_columns = ()`) |
| **Protocol & $[E]_0$ heterogeneity** | Different assay chemistries, enzyme preparations, and total protein (Bradford/$A_{280}$) vs. active-site titration denominators across thousands of papers | No |
| **Max-aggregation bias** | Target is the maximum of $N$ reported values; $N$ varies strongly with enzyme popularity | Partially — raw duplicates exist in upstream BRENDA/SABIO-RK |
| **Database curation errors** | Unit conversions ($\text{U/mg}$ vs. $\text{s}^{-1}$ using monomer vs. oligomer MW) and transcription errors in parent databases | No, at this layer |
| **Substrate-name $\to$ SMILES mapping** | Identical chemicals carry different names across PubChem/KEGG/ChEBI; interconversion can yield divergent SMILES (flagged by authors, Main p. 12) | Partially |
| **Isozyme conflation** | Isozymes of one organism cannot always be separated with current annotations (authors, Main p. 12) | No |
| **2D stereochemistry loss** | Enzymado's `RDKit2DNormalized` is stereo-invariant; enantiomers with different true $k_{\text{cat}}$ share an identical 200-D feature vector | Yes — change ligand featurizer |

### 1.2 Quantitative floor

No row carries an experimental standard deviation, and no replicate structure exists in the table. However, because the paper demonstrates that *"aleatoric uncertainty is greater than the corresponding epistemic uncertainty (see Figure S3)."* `[Source: Main, Page 11]` across roughly 74–98% of predictions, the error of well-trained ensembles provides a direct empirical estimate of the label noise floor.

All values below are in **$\log_{10}(\text{s}^{-1})$ units**; the fold-error column is $10^{\text{MAE}}$, the typical multiplicative error on the raw $\text{s}^{-1}$ scale:

| Empirical anchor | MAE ($\log_{10}$) | $\approx$ Multiplicative error | Provenance |
|---|---|---|---|
| CatPred ensemble, held-out test | 0.7138 | **5.2×** | Table 2, Main p. 9 |
| CatPred ensemble, out-of-distribution test | 0.9870 | 9.7× | Table 2, Main p. 9 |
| Nearest-neighbour baseline (geometric mean of 3 closest training sequences) | 0.999 | 10.0× | Suppl. Table 8b, SI p. 17 (Combined p. 34) |
| CatPred on its **most-confident quartile** of held-out predictions | 0.38054 | **2.4×** | Suppl. Table 5a, SI p. 14 (Combined p. 31) |

**How to read this table:** On the cleanest quarter of the data—the subset where an ensemble of models agrees and predicted aleatoric variance is lowest—the best model still misses by a factor of **$\sim 2.4\times$**. Across the whole held-out test set, it misses by **$\sim 5.2\times$**. Furthermore, *"approximately 76%, 84% and 68% of held-out test predictions fall within an order of magnitude error"* `[Source: Main, Page 7]` for $k_{\text{cat}}$, $K_{\text{m}}$, and $K_{\text{i}}$ respectively—meaning **roughly one $k_{\text{cat}}$ prediction in four is off by more than $10\times$, and that represents the state of the art.**

### 1.3 Implications for benchmark evaluation

- **Set expectations in orders of magnitude, not percentages.** A gap between $R^2 = 0.60$ and $R^2 = 0.64$ lies well inside the experimental noise envelope of any individual database entry.
- **Residual analysis will surface structure that is label noise.** Large positive residuals cluster on heavily studied model enzymes (max-aggregation inflation) and thermophiles (assayed at elevated temperatures invisible to the model). Do not try to fix these by sample reweighting.

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`global_multi_family`):** Wild-type enzymes only; all engineered and site-directed mutant entries were removed upstream. The dataset spans **2,657 unique EC numbers** and **1,685 organisms** across EC 1 through EC 6 (Table 1 and Fig. 1b, Main p. 4).
- **Evolutionary diversity vs. mutational series:** In `dataset_card.json#/statistics`, MMseqs2 clustering at 50% identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) yields `ratio_unique_variants = 0.6562` (4,601 clusters across 7,012 proteins, of which **3,503 are singletons** and the largest cluster has size 29). This confirms a broad, sparse sampling of unrelated enzyme families rather than dense mutational scans around a few parents. *(Note: this 50%-identity internal redundancy metric is distinct from the `--min-seq-id 0.3` threshold used to construct the `similarity_mmseqs2_80_10_10` split.)*
- **Construct vs. canonical sequence gap:** Sequences are canonical full-length UniProt sequences fetched by accession, **not** the exact recombinant constructs (e.g., N/C-terminal His-tags or truncated catalytic domains) used in each primary publication. In addition, 523 rows (124 distinct proteins) exceed 1,024 residues (up to 2,515 aa) and 65 rows contain non-standard amino acid characters; all rows survive the shared PLM intersection (`used_in_runs`).
- **Chemical scope (`has_ligand_smiles: true`):** Crucially, the `smiles` column is **not** a single primary substrate—it is a **period-concatenated canonical SMILES string of all reactants in the reaction**: *"For kcat entries, all listed reactants are used to obtain a concatenated canonical SMILES string"* `[Source: Main, Page 3]`. Co-substrates and organic cofactors therefore enter through the ligand string: *"not only the primary substrate but also the co-substrates (such as secondary substrates, cofactors etc."* `[Source: Main, Page 11]`. Because reactant ordering and co-substrate completeness depend on whether the entry came from BRENDA's reaction table or SABIO-RK, chemically equivalent assays can carry slightly different reactant sets. Enzymado featurizes `smiles` into 200 stereo-invariant 2D descriptors via `descriptastorus` `RDKit2DNormalized` (non-finite $\to 0.0$).

---

## 3. Experimental measurement and STRENDA protocol

Because `catpred_kcat` aggregates thousands of primary papers, **no single assay protocol exists**. Almost all STRENDA state variables are `not_reported` at the dataset level—and that unconditioned heterogeneity is the primary physical cause of the noise floor described in Section 1.

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Heterogeneous *in vitro* steady-state Michaelis–Menten assays across thousands of papers | No | `reported` | *"contain hundreds of thousands of in vitro kinetic measurements manually curated from primary research literature"* `[Source: Main, Page 2]` |
| **Analytical readout & calibration** | Unrecorded (UV-Vis, fluorescence, LC-MS, coupled assays vary by paper) | No | `not_reported` | Not parsed from BRENDA or SABIO-RK |
| **Expression & purification** | Unrecorded (host, tags, and purity vary by paper) | No | `not_reported` | Properties of each source publication; not carried in parsed fields |
| **Expression decoupled from label** | **true** (intensive rate constant $V_{\max}/[E]_0$ on isolated enzyme) | — | `interpreted` | *"one each for kcat, Km and Ki in vitro measurements of wild-type enzymes"* `[Source: Main, Page 3]`. Decoupled from cellular abundance, though $[E]_0$ quantification accuracy varies across labs |
| **Temperature ($T$)** | Per-paper; column retained in CSV with **841 NaNs** (3.7%) | **No** | `not_reported` | *"Environmental conditions such as temperature and pH can have substantial effects on enzyme kinetic parameters, and these are not explicitly accounted for by CatPred due to lack of proper annotations in parent databases"* `[Source: Main, Page 12]` |
| **pH** | Per-paper; column retained in CSV with **710 NaNs** (3.1%) | **No** | `not_reported` | Same as temperature (`numeric_feature_columns = ()`) |
| **Buffer & ionic strength** | — | No | `not_reported` | Not parsed from either parent database |
| **Cofactors & metals** | Concentrations unrecorded; organic co-substrate *identities* included in concatenated SMILES | Implicit in SMILES | `not_reported` | Identity is not concentration; divalent metal loading is unrecorded |
| **Substrate loading ($[S]$)** | Saturating regime ($[S] \gg K_{\text{m}}$) implied by $k_{\text{cat}}$ fit; concentration grid unrecorded | No | `not_reported` | — |
| **Reaction time window** | Initial-velocity regime implied; duration unrecorded | No | `not_reported` | — |
| **Replicates & inter-assay error** | No experimental replicates; duplicates collapsed by **maximum** | — | `not_reported` | Paper's "10 replicates" are deep-ensemble training seeds, not experimental repeats |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Upstream curation in the source paper

1. **Essential-annotation filter:** *"All entries of the downloaded databases were parsed while discarding entries that do not have the essential annotations of (1)"* `[Source: Main, Page 13]` (requires enzyme identity, substrate names, and numeric parameter value).
2. **Organism validation:** *"To maintain the accuracy of organisms’ names, we retain entries only if they are listed in the NCBI Taxonomy database"* `[Source: Main, Page 3]`.
3. **Wild-type restriction:** *"Entries belonging to engineered or mutated enzymes were discarded by parsing the comments attached to each entry."* `[Source: Main, Page 13]`.
4. **Full-reactant SMILES canonicalisation:** *"For kcat entries, all listed reactants are used to obtain a concatenated canonical SMILES string"* `[Source: Main, Page 3]` and *"Each SMILES string was canonicalized using the RDKit Python library."* `[Source: Main, Page 13]`, using PubChem Identifier Exchange when database structures were missing.
5. **Duplicate collapse by maximum:** *"then the maximum (for kcat) and the geometric mean (for Km and Ki) value, respectively is retained"* `[Source: Main, Page 3]`, justified as *"The selection of the maximum value for kcat is carried out because it likely maps to the optimal growth conditions"* `[Source: Main, Page 3]`.
6. **Base-10 log transform:** *"the values were log10-transformed to obtain approximately normal distributions for each"* `[Source: Main, Page 13]`.

### 4.2 Enzymado ingestion, reconciliation, and target standardization

- **Pipeline rules:** `drop_na_in = ["protein_sequence", "kcat"]` $\to$ key-level deduplication on `deduplicate_by = ["uniprot_id", "smiles"]` $\to$ removal of truncated UniProt fragments via `remove_uniprot_fragments.py` against `data/processed/uniprot_fragments.csv` $\to$ `numeric_feature_columns = ()`.
- **Paper-to-benchmark delta:** CatPred-DB-$k_{\text{cat}}$ reports **23,197 entries across 7,183 unique sequences** (Table 1, Main p. 4), whereas Enzymado's cleaned dataset has **22,693 rows across 7,012 distinct proteins** ($-504$ rows, $-171$ proteins). This delta arises from removing truncated UniProt fragments and deduplicating on `(uniprot_id, smiles)` rather than `(sequence, smiles)`.
- **Target transformation & invertibility:**
  $$z = \frac{\log_{10}(k_{\text{cat}}\text{ [s}^{-1}\text{]}) - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  The $\log_{10}$ transformation was applied upstream in CatPred-DB and is **not** applied a second time. Standardization parameters $(\mu_{\text{train}}, \sigma_{\text{train}})$ are fitted strictly on the `train` split and saved in `*.splits.metadata.json`, making raw $\text{s}^{-1}$ units exactly recoverable. Because all curated $k_{\text{cat}}$ values are strictly positive, no pseudocount or zero-floor offset is needed.

---

## 5. Split semantics and leakage caveats

Two 80/10/10 split protocols are published (`catpred_kcat_random_80_10_10_standardized` and `catpred_kcat_similarity_mmseqs2_80_10_10_standardized`; see [`dataset_card.json`](../../data/processed/properties/kcat/dataset_card.json) for exact split counts and moments):

- **What `random_80_10_10` tests:** Proteins (`uniprot_id`) are partitioned 80/10/10 at random so all substrates of a given UniProt accession stay in the same split. However, because it does not cluster by sequence similarity, orthologous or isozymic proteins with $>90\%$ sequence identity frequently straddle `train` and `test`.
- **What `similarity_mmseqs2_80_10_10` tests:** Whole MMseqs2 sequence clusters built at **30% sequence identity** (`--min-seq-id 0.3`) are assigned intact to `train`, `val`, or `test`, guaranteeing that no test enzyme shares $\ge 30\%$ sequence identity with any training enzyme.

### Summary of leakage risks and structural caveats

1. **Close-homolog leakage in `random_80_10_10`:** Always report sequence-generalisation claims from `similarity_mmseqs2_80_10_10`.
2. **Target variance compression in the similarity test split:** The similarity test split has `std = 0.8066` (vs. `1.0000` in train and `1.0274` in the random test split), which mechanically lowers $R^2$ relative to MAE.
3. **Shared substrates across protein splits:** Because grouping is performed on the protein axis, ubiquitous co-substrates ($\text{ATP}$, $\text{NAD(P)H}$, $\text{O}_2$) and common substrates appear in both `train` and `test`.
4. **Irreducible condition & max-collapse bias:** Both `train` and `test` inherit the upward max-deduplication bias on heavily studied enzymes and the unobserved $T$/pH variance.
5. **2D stereoisomer collisions:** `RDKit2DNormalized` assigns identical ligand vectors to stereoisomers.
6. **Long-sequence truncation tail:** 523 rows (124 distinct proteins) exceed 1,024 residues and are truncated by 1,024-token PLMs.
