---
task_id: catpred_km
dataset_id: catpred_km
property_name: km
task_type: protein_ligand_global
label_type: scalar_regression
layout: Layout B
computed_statistics_ref: data/processed/properties/km/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1038/s41467-025-57215-9
domain_review_doi: https://doi.org/10.1371/journal.pbio.3001402
---

# Dataset Card: `catpred_km` ($K_m$ — Michaelis Constant)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/km/dataset_card.json`](../../data/processed/properties/km/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `catpred_km`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Dataset Paper**: Boorla, V. S. & Maranas, C. D. *CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters.* **Nature Communications** 16, 2072 (2025). DOI: [`10.1038/s41467-025-57215-9`](https://doi.org/10.1038/s41467-025-57215-9) (main PDF, SI PDF).
- **Domain Review / Companion Benchmark Paper**: Kroll, A., Engqvist, M. K. M., Heckmann, D. & Lercher, M. J. *Deep learning allows genome-scale prediction of Michaelis constants from structural features.* **PLOS Biology** 19(10), e3001402 (2021). DOI: [`10.1371/journal.pbio.3001402`](https://doi.org/10.1371/journal.pbio.3001402) (review PDF).
- **Upstream Repositories**:
  - **BRENDA (Release 2022_2)** & **SABIO-RK (Release 2023_05)**: Literature-curated enzyme kinetic parameter databases providing raw $K_m$ values, substrate names, organism names, and database cross-references.
  - **UniProt & GenBank**: Queried dynamically to resolve full-length amino-acid sequences from protein accessions or organism + EC number + molecular-weight filters.
  - **PubChem, ChEBI, KEGG & RDKit (v2023_03)**: Used to map substrate names to canonical SMILES strings.
- **License**: Article text published under CC BY-NC-ND 4.0 (main PDF p. 1); CatPred-DB dataset and code archived on Zenodo ([`10.5281/zenodo.14782924`](https://doi.org/10.5281/zenodo.14782924)).

### Reconciliation: Paper-Reported vs. Enzymado Cleaned Counts

| Metric | CatPred Paper (Table 1, p. 4) | Enzymado `overall` (`catpred_km.csv`) | Enzymado `used_in_runs` | Cause of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Total entries (rows)** | 41,174 | **40,088** | **40,088** (100%) | Removal of **1,086** UniProt fragment rows (`832` train, `120` val, `134` test) by `scripts/data_prep/remove_uniprot_fragments.py` |
| **Distinct protein sequences** | 12,355 | **12,004** | **12,004** (100%) | `-351` unique sequences lost when fragment accessions were removed |
| **Unique `uniprot_id` strings** | — | **12,114** | **12,114** | 110 identical sequences shared across distinct isoform/secondary accessions |
| **Unique organisms** | 2,419 | — | — | Not carried as an explicit column in `catpred_km.csv` |
| **Unique EC classes** | 3,550 | — | — | Not carried as an explicit column in `catpred_km.csv` |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`catpred_km` is a **protein–ligand global scalar regression** task (`Layout B`). Each row pairs a full-length enzyme amino-acid sequence (`protein_sequence`) with a canonical SMILES string (`smiles`) and asks the model to predict the standardized base-10 logarithm of the Michaelis constant (`km_standardized`).
- **Crucial difference from `kcat` on the `smiles` column**: Whereas `catpred_kcat` concatenates the SMILES of *all* participating reactants separated by dots (`.`), in `catpred_km` the `smiles` string represents **only the single titrated substrate** whose $K_m$ was measured (*"For Km or Ki prediction, the SMILES string corresponding to the relevant substrate is used."* — Boorla & Maranas 2025, p. 3).
- **Assay conditions are provenance-only**: Although `temperature` ($^\circ\text{C}$) and `ph` are included in `catpred_km.csv`, they are missing in **1,747** (4.36%) and **1,391** (3.47%) rows respectively, and `src/enzymado/data_loading/data_registry.py` sets `numeric_feature_columns = ()`. Models therefore predict $K_m$ from sequence and substrate chemical structure alone.

### Physical & Mechanistic Meaning of the Target
The Michaelis constant $K_m$ is the substrate concentration $[S]$ at which an enzyme-catalysed reaction proceeds at half of its maximal velocity ($v = V_{\max}/2$). Under Briggs–Haldane steady-state kinetics for a simple one-substrate mechanism $E + S \underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} ES \overset{k_{\text{cat}}}{\longrightarrow} E + P$:
$$K_m = \frac{k_{-1} + k_{\text{cat}}}{k_1} = K_d + \frac{k_{\text{cat}}}{k_1}$$
When catalytic turnover is slow relative to substrate dissociation ($k_{\text{cat}} \ll k_{-1}$), $K_m \approx K_d$ directly measures the thermodynamic dissociation constant of the enzyme–substrate complex. Across natural enzymes, $K_m$ spans roughly eight orders of magnitude ($10^{-7}\text{ mM}$ to $10^{1}\text{ mM}$), peaking near physiological intracellular metabolite concentrations ($\sim 0.1\text{–}1\text{ mM}$).

### Why Duplicate Aggregation Uses Geometric Mean Instead of Maximum
A key design distinction between `catpred_km` and `catpred_kcat` is how replicate measurements for identical `(protein_sequence, smiles)` pairs are aggregated:
- In `catpred_kcat`, replicates are collapsed via the **maximum** value to discard suboptimal assay conditions.
- In `catpred_km`, replicates (after discarding any group whose $\log_{10}$ standard deviation exceeds `1.0`) are collapsed via the **geometric mean** (i.e., arithmetic mean on the $\log_{10}$ scale, stored upstream as `log10km_mean`). As Boorla & Maranas (2025, p. 3) note, $K_m$ reflects enzyme–substrate binding affinity more directly than environmental turnover optima, so log-space averaging suppresses random experimental scatter without biasing toward artificially high (weak-binding) $K_m$ values.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Within-group replicate variance up to $1.0\text{ }\log_{10}$ unit**: Upstream deduplication retains enzyme–substrate groups with $\sigma(\log_{10} K_m) \le 1.0$ (up to a 10-fold spread across studies), embedding an immediate inter-laboratory measurement floor of $\sim 0.25\text{–}0.35\text{ }\log_{10}$ units.
2. **Unobserved cosubstrate concentrations in multisubstrate enzymes**: For bisubstrate reactions (e.g., oxidoreductases using $\text{NAD(P)H}$ or transferases using $\text{ATP}$), the apparent $K_m$ of the primary substrate depends on whether the cosubstrate was held at saturating or subsaturating concentration and whether the mechanism is ordered sequential, random sequential, or ping-pong. Because only the single titrated substrate is represented in `smiles`, variation in cosubstrate assay design appears as label noise.
3. **Unmodeled pH, temperature, and ionic strength**: Shifts in assay pH or ionic strength alter the protonation states of catalytic residues and charged substrates (e.g., phosphates, carboxylates), perturbing electrostatics in the binding pocket without altering the input SMILES or sequence.
4. **Heuristic sequence mapping in BRENDA**: When a BRENDA record lacks an explicit UniProt or GenBank ID, CatPred queries UniProt/GenBank by `organism + EC number` (filtered by molecular weight when available), which can map kinetic measurements to a paralogous isozyme with different substrate affinity.

### Quantitative Performance Anchors from the Literature

| Model / Baseline | Evaluation Regime | $R^2$ | $\text{MAE}$ ($\log_{10}$) | $\text{RMSE}$ ($\log_{10}$) | Fold Error ($10^{\text{MAE}}$) | $p_{1\text{mag}}$ ($<10\times$ error) | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CatPred (Seq-Attn + ESM-2 + EGNN + D-MPNN)** | CatPred held-out test ($10\%$) | **0.6422** | **0.5582** | — | **3.62×** | **83.91%** | Boorla & Maranas (2025), Table 2 (p. 9) & SI Table 8b (p. 17) |
| **CatPred (full ensemble)** | CatPred OOD test ($<99\%$ seq id) | **0.5332** | **0.6528** | — | **4.50×** | **79.04%** | Boorla & Maranas (2025), Table 2 (p. 9) & SI Table 8b (p. 17) |
| **CatPred (most-confident quartile)** | CatPred held-out test (bottom 25% unc.) | — | **0.3643** | — | **2.31×** | — | Boorla & Maranas (2025), SI Table 6a (p. 14) |
| **Nearest-Neighbour Sequence Baseline** | CatPred held-out test ($10\%$) | — | **0.7550** | — | **5.69×** | — | Boorla & Maranas (2025), SI Table 8b (p. 17) |
| **Kroll et al. (2021) (UniRep + GNN)** | BRENDA natural pairs test ($N=2,335$) | **0.53** | — | **0.64** ($\text{MSE}=0.41$) | — | — | Kroll et al. (2021), Abstract & Results (pp. 1, 6) |

---

## 3. Specificity & Breadth of the Dataset

- **Protein Sequence Diversity**: The 40,088 rows cover **12,004 distinct protein sequences** (average **3.34 substrates per enzyme**). MMseqs2 clustering at 50% sequence identity and 80% alignment coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) yields **7,039 clusters** (`ratio_unique_variants = 0.5864`), including **5,061 singletons** (71.9% of clusters) and a maximum cluster size of **73** (mean cluster size `1.705`).
- **Sequence Length Distribution**: Lengths span **10 to 2,590 aa** (row-level mean `443.06`, std `216.17`, median `392.0`, IQR `308–523`; distinct-protein mean `443.92`, median `399.0`), with **980 rows** (`256` distinct proteins) exceeding `1,024` residues and **97 rows** containing non-standard amino-acid characters (`B`, `J`, `O`, `U`, `X`, `Z`). Every single row (`40,088 / 40,088`) is retained in `used_in_runs`.
- **Taxonomic & Functional Breadth**: Upstream CatPred-DB-$K_m$ spans **2,419 organisms** across all domains of life and **3,550 unique four-digit EC numbers** covering all seven top-level EC classes (hydrolases, oxidoreductases, transferases, lyases, ligases, isomerases, and translocases).

---

## 4. How the Dataset Was Built

1. **Upstream Extraction (`CatPred-DB-Km`)**:
   - Parsed BRENDA Release 2022_2 and SABIO-RK Release 2023_05 for positive numerical $K_m$ records, excluding engineered/mutant enzymes.
   - Mapped single-substrate names to canonical SMILES via PubChem, ChEBI, KEGG, and RDKit (`v2023_03`).
   - Queried UniProt and GenBank for amino-acid sequences via direct accession or `(organism, EC number, molecular weight)` matching.
   - Grouped identical `(sequence, substrate_smiles)` records, discarded groups with $\sigma(\log_{10} K_m) > 1.0$, and collapsed remaining groups to their geometric mean (`log10km_mean`), yielding **41,174 entries** across **12,355 unique sequences**.
2. **Enzymado Processing (`src/enzymado/propertyprep/catpred.py` & `scripts/data_prep/remove_uniprot_fragments.py`)**:
   - Validated sequences (`common.is_valid_sequence`), re-canonicalised SMILES (`common.canonicalize_smiles`), renamed columns to `['uniprot_id', 'protein_sequence', 'smiles', 'temperature', 'ph', 'km']`, and dropped exact row duplicates.
   - Removed **1,086 rows** corresponding to UniProt sequence fragments (`remove_uniprot_fragments.py`), leaving **40,088 rows** and **12,004 unique sequences**.
   - Discarded upstream CatPred split assignments, re-partitioned into `random_80_10_10` and `similarity_mmseqs2_80_10_10` grouped by `protein_sequence`, and z-score standardized `km` using training-split statistics (`km_standardized`).

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Steady-state initial rates $v_0$ are measured across serial substrate dilutions ($[S]$ typically spanning $0.2\,K_m$ to $5\,K_m$) via UV-Vis spectrophotometry (e.g., $\text{NAD(P)H}$ absorbance at $340\text{ nm}$), fluorescence, HPLC/MS, or radiometric assays, and fitted to the Michaelis–Menten hyperbola.
- **STRENDA Reporting Completeness**: While STRENDA mandates full reporting of temperature, pH, buffer composition, ionic strength, and cosubstrate concentrations, **4.36%** (`1,747` rows) of `catpred_km` entries lack `temperature`, **3.47%** (`1,391` rows) lack `ph`, and cosubstrate/buffer concentrations are unrecorded.
- **Unit Scale Note ($\text{mM}$ vs. $\text{M}$)**: Raw BRENDA $K_m$ values and `CatPred_km.csv` (`log10km_mean`) are numerically on the **$\log_{10}(\text{mM})$ scale** (training mean $\approx -0.733$ in `random_80_10_10` and $-0.709$ in `similarity_mmseqs2_80_10_10`, corresponding to $K_m \approx 0.19\text{ mM} = 190\,\mu\text{M}$, as noted in `scripts/data_prep/card_metadata.py` and Kroll et al. 2021 p. 3), even though Boorla & Maranas (2025) and `benchmark_card.json` label the axis as $\log_{10}(\text{mM})$ or $\log_{10}(\text{M})$. Because shifting from $\text{mM}$ to $\text{M}$ adds a constant $-3.0$ to every $\log_{10}$ target, z-score standardization (`km_standardized`), $R^2$, Spearman $\rho$, and back-transformed $\log_{10}$ MAE/RMSE are completely unaffected.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Rows (Proteins) | Val Rows (Proteins) | Test Rows (Proteins) | Train Raw Mean ± SD ($\log_{10}\text{ mM}$) | Test Standardized Mean ± SD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 32,126 (9,603) | 3,925 (1,200) | 4,037 (1,201) | $-0.7330 \pm 1.2775$ | $+0.0177 \pm 1.0079$ |
| **`similarity_mmseqs2_80_10_10`** | 31,953 (10,185) | 4,063 (1,016) | 4,072 (803) | $-0.7089 \pm 1.2654$ | **$-0.2490 \pm 1.0334$** |

1. **Sequence-Grouped Splitting vs. Shared Substrates**: Both splits group strictly by `protein_sequence` (`0` shared sequences across splits), and `similarity_mmseqs2_80_10_10` enforces `<30%` pairwise sequence identity (`--min-seq-id 0.3`) between clusters in different splits (preventing homologous enzyme leakage, which is far stricter than CatPred's `<99%` OOD filter). However, ubiquitous cofactors and metabolites (`ATP`, `NAD+`, `NADH`, `pyrophosphate`) are shared across train, val, and test splits; the similarity split therefore tests **unseen enzyme scaffolds acting on a mix of seen and unseen substrates**.
2. **Downward Target Shift in the Similarity Test Split**: Because `similarity_mmseqs2_80_10_10` assigns whole MMseqs2 clusters to splits to hit an 80/10/10 row target, the test split receives fewer distinct proteins (`803`) with higher substrate multiplicity (`5.07` substrates/protein vs. `3.14` in train) and a **$-0.249\text{ }\sigma$ lower standardized mean** ($\approx -0.315\text{ }\log_{10}$ units, i.e., $\sim 2\times$ tighter average binding affinity). Models that fail to adapt to this family-level prior shift will exhibit a systematic positive bias on the similarity test set.
3. **Minor `uniprot_id` vs. `protein_sequence` Multiplicity**: While `12,004` unique `protein_sequence` strings are partitioned cleanly without leakage, `12,114` unique `uniprot_id` strings appear in the table (`110` cases where identical sequences carry distinct isoform/secondary accessions, and `11` accessions associated with $>1$ sequence string). Always key external sequence lookups by `protein_sequence` rather than `uniprot_id`.

