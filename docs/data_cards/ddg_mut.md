---
task_id: fireprotdb_ddg
dataset_id: fireprotdb_ddg
property_name: ddg
task_type: mutation_delta_global
label_type: scalar_regression
layout: Layout C
computed_statistics_ref: data/processed/properties/ddg_mut/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkaa981
domain_review_doi: https://doi.org/10.1093/bib/bbab555
---

# Dataset Card: `fireprotdb_ddg` ($\Delta\Delta G$ — Mutational Change in Folding Free Energy)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/ddg_mut/dataset_card.json`](../../data/processed/properties/ddg_mut/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `fireprotdb_ddg`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Paper**: Stourac, J. et al. *FireProtDB: database of manually curated protein stability data.* **Nucleic Acids Research** 49, D319–D324 (2021). DOI: [`10.1093/nar/gkaa981`](https://doi.org/10.1093/nar/gkaa981) ([main PDF](../pdfs/fireprotdb_musil2021_nar.pdf)).
- **Domain Review & Benchmark Papers**:
  - Pancotti, C. et al. *Predicting protein stability changes upon single-point mutation: a thorough comparison of the available tools on a new dataset.* **Briefings in Bioinformatics** 23(2), bbab555 (2022). DOI: [`10.1093/bib/bbab555`](https://doi.org/10.1093/bib/bbab555) ([review PDF](../pdfs/rev_ddg_pancotti2022_bib.pdf)).
  - Sanavia, T. et al. *Limitations and open challenges in machine learning applications for protein stability prediction.* **Computational and Structural Biotechnology Journal** 18, 1968–1979 (2020). DOI: [`10.1016/j.csbj.2020.07.011`](https://doi.org/10.1016/j.csbj.2020.07.011) ([review PDF](../pdfs/rev_stability_sanavia2020_csbj.pdf)).
- **Upstream Repositories**:
  - **FireProtDB (`data/raw/fireprotdb/fireprotdb_ec.tsv`)**: Manually curated single-point thermostability records integrating corrected ProTherm entries, ProtaBank, Varibench, and primary literature, filtered by `scripts/data_prep/download_fireprotdb_ec.py` to retain only entries with an explicit EC number.
  - **UniProtKB**: Queried via `UNIPROTKB` accession to fetch canonical wild-type sequences and verify 0 sequence fragments.
- **License**: FireProtDB article published under CC BY 4.0 (main PDF p. 1).

### Reconciliation: Upstream FireProtDB vs. Enzymado Cleaned Counts

| Metric | FireProtDB 1.0 Paper (Stourac et al. 2021, p. 4) | Raw EC-Filtered Export (`fireprotdb_ec.tsv`) | Enzymado `overall` (`fireprotdb_ddg.csv`) | Enzymado `used_in_runs` | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Total stability rows** | 15,442 ($\Delta T_m$ + $\Delta\Delta G$, all proteins) | 3,992 (non-null `DDG` with EC number) | **3,774** | **3,760** | `14` rows of giant human titin (`Q8WZ42`, length `34,350` aa) excluded in `used_in_runs` by PLM length cap |
| **Unique single-point mutants (`protein_sequence`)** | 6,579 (all proteins) | — | **2,532** | **2,518** | **1,242** extra rows come from replicate measurements of the same mutant under different experimental conditions (`dataset_id`) |
| **Unique wild-type parent proteins (`uniprot_id`)** | 237 (all proteins) | — | **69** | **68** | 69 wild-type enzymes (forming **62** MMseqs2 50%-identity clusters; `0` UniProt fragments removed) |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`fireprotdb_ddg` is a **mutation-delta global scalar regression** task (`Layout C`). Each row provides:
- `wildtype_sequence`: canonical amino-acid sequence of the unmutated parent enzyme.
- `mutation`: 1-indexed single-point substitution code (`^([A-Z])(\d+)([A-Z])$`, e.g., `I96A`).
- `protein_sequence`: full-length mutant sequence obtained by applying `mutation` to `wildtype_sequence`.
- `ddg` / `ddg_standardized`: experimental change in folding free energy ($\Delta\Delta G$ in $\text{kcal/mol}$) and its training-split z-score standardized counterpart.

### ⚠️ Critical Sign Convention Alert (Positive $\Delta\Delta G$ = Destabilizing in FireProtDB!)
In the protein stability literature, two opposite sign conventions coexist depending on whether $\Delta G$ is written for folding ($\text{U} \to \text{F}$, where $\Delta G_{\text{folding}} < 0$) or unfolding ($\text{F} \to \text{U}$, where $\Delta G_{\text{unfolding}} > 0$).
- **FireProtDB's actual definition (Stourac et al. 2021, p. 5)**:
  $$\Delta\Delta G > +1\text{ kcal/mol} \implies \text{\textbf{Destabilizing}}, \qquad \Delta\Delta G < -1\text{ kcal/mol} \implies \text{\textbf{Stabilizing}}$$
- **Empirical verification in `fireprotdb_ddg`**: Because $\sim 75\text{–}86\%$ of single-point mutations are destabilizing or neutral, the raw training-set mean of `ddg` in `fireprotdb_ddg_80_10_10.splits.metadata.json` is **`+1.3267 kcal/mol`** (`std = 2.1723 kcal/mol` in `random_80_10_10`; `+1.3510 ± 2.2571 kcal/mol` in `similarity_mmseqs2_80_10_10`).
- **Correction to `card_metadata.py` prose**: Note that `scripts/data_prep/card_metadata.py` inadvertently stated `"where positive is stabilising"` in its summary string for `ddg_mut`. In fact, **positive `ddg` in `fireprotdb_ddg` is destabilizing** (and **negative `ddg` is stabilizing**), which is opposite in sign to `fireprotdb_dtm` (where positive $\Delta T_m$ is stabilizing). Always select **negative** predicted `ddg` values when ranking candidate mutations for improved thermostability!

### Effective Sample Size on the Protein Axis (69 Parent Enzymes & 1,242 Replicate Rows)
1. **Only 69 wild-type parent scaffolds (`68` in `used_in_runs`)**: All 3,774 rows and 2,532 unique mutant sequences originate from just **69 unique wild-type parent enzymes** (`uniprot_id`).
2. **Why `3,774 rows > 2,532 unique mutant sequences`**: Because `fireprotdb.py` deduplicates on `['protein_sequence', 'ddg']` and FireProtDB stores experiment-level records (`dataset_id = EXPERIMENT_ID`), **1,242 rows** represent replicate measurements of an already-present mutant sequence under different pH values, temperatures, denaturants (urea vs. GdnHCl), or publications.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Unmodeled experimental conditions across `1,242` replicate rows**: $\Delta\Delta G$ depends on assay temperature, pH, ionic strength, and whether unfolding was induced chemically (urea/GdnHCl with linear extrapolation) or thermally (van 't Hoff / calorimetric analysis). Because models only see `(wildtype_sequence, protein_sequence)`, multi-condition replicates with different `ddg` targets create irreducible conditional noise.
2. **Experimental error floor ($\sim 1.0\text{ kcal/mol}$)**: Sanavia et al. (2020, p. 7) establish that experimental uncertainty across independent determinations imposes an upper bound on Pearson correlation of **$r \approx 0.80$** and a lower-bound RMSE floor of **$\sim 1.0\text{ kcal/mol}$** ($\approx 0.45$ standardized units).
3. **Destabilizing class bias & antisymmetry**: Without reverse-mutation training ($\Delta\Delta G(\text{mut} \to \text{wt}) = -\Delta\Delta G(\text{wt} \to \text{mut})$), regressors overfit the $+1.33\text{ kcal/mol}$ destabilizing mean and struggle to identify stabilizing ($\Delta\Delta G < -1\text{ kcal/mol}$) variants.

### Quantitative Performance Anchors from the Literature

| Benchmark / Regime | Typical Pearson $r$ | Typical RMSE ($\text{kcal/mol}$) | Source |
| :--- | :---: | :---: | :--- |
| **Theoretical experimental reproducibility ceiling** | **$\sim 0.80$** | **$\sim 1.00$** | Sanavia et al. (2020), Section 3.3 (p. 7) |
| **Within-scaffold random mutation split (e.g., mCSM)** | **0.70 – 0.80** | **1.00 – 1.25** | Sanavia et al. (2020), Section 3.4 (p. 8) |
| **Per-site / Per-protein cross-scaffold split** | **0.51 – 0.54** | — | Sanavia et al. (2020), Section 3.4 (p. 8) |
| **S669 blind OOD benchmark (<25% seq id, 21 tools, direct variants)** | **0.20 – 0.50** | **$\ge 1.44$** | Pancotti et al. (2022), Table 2 & p. 4 |

---

## 3. Specificity & Breadth of the Dataset

- **Parent Scaffold Clustering (`69` Wild-Type Enzymes in `overall`, `68` in `used_in_runs`)**: MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) groups the 69 parent sequences into **62 clusters** (`ratio_unique_variants = 0.8986`; `61` clusters across `68` parents in `used_in_runs`), with **57 singletons** (`56` in `used_in_runs`) and a maximum cluster size of **2** (mean `1.11`).
- **The Human Titin (`Q8WZ42`, `34,350` aa) Outlier in `overall` vs. `used_in_runs`**:
  - In `overall` (`3,774` rows, `2,532` mutants), sequence lengths span **64 to 34,350 aa** with a heavily skewed mean of `322.25` aa and std of `1,899.49` aa (median `149.0`, IQR `110–221`; `38` rows $>1,024$ aa). This extreme tail is caused by **14 mutations** mapped to canonical full-length human titin (`Q8WZ42`, EC 2.7.11.1, `34,350` aa), even though the underlying biophysical experiments measured an isolated ~90–100 aa immunoglobulin/kinase domain in vitro.
  - In `used_in_runs` (`3,760` rows, `2,518` mutants), those `14` titin rows are automatically dropped by the PLM embedding intersection, reducing the maximum sequence length to **1,863 aa** (`24` rows $>1,024$ aa) and the row-level mean length to **195.55 ± 149.18 aa** (median `149.0`, IQR `110–215`).

---

## 4. How the Dataset Was Built

1. **EC-Filtered Download (`scripts/data_prep/download_fireprotdb_ec.py`)**:
   - Exported FireProtDB entries with a non-empty `EC_NUMBER` to `data/raw/fireprotdb/fireprotdb_ec.tsv` (`3,992` rows with finite `DDG`).
2. **Substitution Validation & Sequence Construction (`src/enzymado/propertyprep/fireprotdb.py`)**:
   - Filtered `SUBSTITUTION` strings to canonical single-point mutations `^([A-Z])(\d+)([A-Z])$`.
   - Retrieved canonical `wildtype_sequence` from UniProt via `UNIPROTKB`, verified residue identity at the 1-indexed position, constructed `protein_sequence`, and dropped duplicates on `['protein_sequence', 'ddg']`, yielding **3,774 rows** across **2,532 unique mutant sequences** and **69 parent enzymes** (`0` UniProt fragments removed).

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Isothermal chemical denaturation (urea or GdnHCl titrations monitored by CD or intrinsic fluorescence and fitted via linear extrapolation) or reversible thermal denaturation (DSC / CD van 't Hoff analysis).
- **STRENDA Reporting Gaps**: Assay pH, temperature, buffer composition, and denaturant identity are stored under `EXPERIMENT_ID` in FireProtDB but omitted from model inputs in `fireprotdb_ddg.csv`. Additionally, mutations assayed on isolated domains of multidomain enzymes are mapped onto the full-length UniProt sequence.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Rows (Mutants / Parents) | Val Rows (Mutants / Parents) | Test Rows (Mutants / Parents) | Parents Straddling Splits | Train Raw Mean ± SD ($\text{kcal/mol}$) | Test Standardized Mean ± SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 3,015 (2,025 / 67) | 380 (253 / 36) | 379 (254 / 37) | **45 / 69 (65.2%)** | $+1.3267 \pm 2.1723$ | $-0.0566 \pm 1.0772$ |
| **`similarity_mmseqs2_80_10_10`** | 2,741 (1,738 / 55) | 656 (521 / 8) | 377 (273 / 6) | **0 / 69 (0.0%)** | $+1.3510 \pm 2.2571$ | **$-0.0680 \pm 0.7798$** |

1. **Fundamental Difference in Evaluation Regime (`random` = Within-Scaffold Interpolation vs. `similarity` = Zero-Shot Cross-Scaffold Generalisation)**:
   - Because `propertyprep.py` groups splits by `protein_sequence` (the *mutant* sequence), in **`random_80_10_10`** different point mutants of the same wild-type parent enzyme are split across `train`, `val`, and `test` (**45 of the 69 parent enzymes straddle multiple splits**). `random_80_10_10` therefore evaluates within-scaffold mutational interpolation.
   - In **`similarity_mmseqs2_80_10_10`**, MMseqs2 clustering at 30% sequence identity (`--min-seq-id 0.3`) forces all mutants of a parent enzyme ($>99\%$ sequence identity) and any homologous parents ($\ge 30\%$ identity) into the same cluster, so **0 / 69 parent enzymes straddle splits**. `similarity_mmseqs2_80_10_10` therefore evaluates true zero-shot generalisation to unseen enzyme scaffolds (analogous to the S669 benchmark of Pancotti et al. 2022).
2. **Validation Split Size & Test Variance Compression in `similarity_mmseqs2_80_10_10`**: Assigning whole parent clusters intact across 69 parents places **656 rows** (17.4% of the dataset, `8` parents) into `val` and **377 rows** (10.0%, `6` parents) into `test`, with a compressed test standardized standard deviation (`0.7798` vs. `1.0000` in train).

