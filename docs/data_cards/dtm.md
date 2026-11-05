---
task_id: fireprotdb_dtm
dataset_id: fireprotdb_dtm
property_name: dtm
task_type: mutation_delta_global
label_type: scalar_regression
layout: Layout C
computed_statistics_ref: data/processed/properties/dtm/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkaa981
domain_review_doi: https://doi.org/10.1016/j.csbj.2020.07.011
---

# Dataset Card: `fireprotdb_dtm` ($\Delta T_m$ — Mutational Change in Melting Temperature)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/dtm/dataset_card.json`](../../data/processed/properties/dtm/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `fireprotdb_dtm`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Paper**: Stourac, J. et al. *FireProtDB: database of manually curated protein stability data.* **Nucleic Acids Research** 49, D319–D324 (2021). DOI: [`10.1093/nar/gkaa981`](https://doi.org/10.1093/nar/gkaa981) (main PDF).
- **Domain Review Paper**: Sanavia, T. et al. *Limitations and open challenges in machine learning applications for protein stability prediction.* **Computational and Structural Biotechnology Journal** 18, 1968–1979 (2020). DOI: [`10.1016/j.csbj.2020.07.011`](https://doi.org/10.1016/j.csbj.2020.07.011) (review PDF).
- **Upstream Repositories**:
  - **FireProtDB (`data/raw/fireprotdb/fireprotdb_ec.tsv`)**: Manually curated single-point thermostability records integrating corrected ProTherm entries, ProtaBank, Varibench, and primary literature, filtered by `scripts/data_prep/download_fireprotdb_ec.py` to retain only entries with an explicit EC number.
  - **UniProtKB**: Queried via `UNIPROTKB` accession to fetch canonical wild-type sequences and verify 0 sequence fragments.
- **License**: FireProtDB article published under CC BY 4.0 (main PDF p. 1); database freely available at `https://loschmidt.chemi.muni.cz/fireprotdb/`.

### Reconciliation: Upstream FireProtDB vs. Enzymado Cleaned Counts

| Metric | FireProtDB 1.0 Paper (Stourac et al. 2021, p. 4) | Raw EC-Filtered Export (`fireprotdb_ec.tsv`) | Enzymado `overall` & `used_in_runs` (`fireprotdb_dtm.csv`) | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Total stability rows** | 15,442 ($\Delta T_m$ + $\Delta\Delta G$, all proteins) | 3,515 (non-null `DTM` with EC number) | **3,338** (100% in `used_in_runs`) | Restricted to EC-annotated enzymes with valid 1-indexed single-point substitutions and deduplicated on `[protein_sequence, dtm]` |
| **Unique single-point mutants (`protein_sequence`)** | 6,579 (all proteins) | — | **2,312** | **1,026** extra rows come from replicate measurements of the same mutant across different experimental conditions/studies (`dataset_id`) |
| **Unique wild-type parent proteins (`uniprot_id`)** | 237 (all proteins) | — | **91** | 91 wild-type enzymes (forming **82** MMseqs2 50%-identity clusters; `0` UniProt fragments removed) |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`fireprotdb_dtm` is a **mutation-delta global scalar regression** task (`Layout C`). Each row provides:
- `wildtype_sequence`: canonical amino-acid sequence of the unmutated parent enzyme.
- `mutation`: 1-indexed single-point substitution code (`^([A-Z])(\d+)([A-Z])$`, e.g., `A123V`).
- `protein_sequence`: full-length mutant amino-acid sequence obtained by applying `mutation` to `wildtype_sequence`.
- `dtm` / `dtm_standardized`: experimental change in melting temperature ($\Delta T_m$ in $^\circ\text{C}$) and its training-split z-score standardized counterpart.

### Physical & Mechanistic Meaning & Sign Convention
$\Delta T_m = T_m(\text{mutant}) - T_m(\text{wild-type})$ measures the shift in the thermal denaturation midpoint ($^\circ\text{C}$) upon a single amino-acid substitution:
- **Sign convention**: **$\Delta T_m > 0\,^\circ\text{C}$ is thermostabilizing** (the mutant unfolds at a higher temperature than wild-type) and **$\Delta T_m < 0\,^\circ\text{C}$ is destabilizing**. *(Contrast this with `fireprotdb_ddg`, where FireProtDB defines $\Delta\Delta G > 0\text{ kcal/mol}$ as destabilizing and $\Delta\Delta G < 0\text{ kcal/mol}$ as stabilizing!)*
- **Destabilizing skew**: Because random single-residue substitutions in evolved enzymes predominantly disrupt hydrophobic core packing or hydrogen bonding, the target distribution is centered below zero (`random_80_10_10` train mean $= -2.8017 \pm 7.3029\,^\circ\text{C}$; `similarity_mmseqs2_80_10_10` train mean $= -2.6177 \pm 7.1670\,^\circ\text{C}$). In FireProtDB overall, 43% of mutations are destabilizing ($\Delta T_m < -1\,^\circ\text{C}$), 43% are neutral ($[-1, +1]\,^\circ\text{C}$), and only 14% are stabilizing ($\Delta T_m > +1\,^\circ\text{C}$; Stourac et al. 2021, p. 5).

### Effective Sample Size on the Protein Axis (91 Parent Enzymes & 1,026 Multi-Condition Rows)
Two structural properties of `fireprotdb_dtm` are essential for interpreting model behavior:
1. **Only 91 wild-type parent scaffolds**: All 3,338 rows and 2,312 unique mutant sequences derive from just **91 unique wild-type enzymes** (`uniprot_id`).
2. **Why `3,338 rows > 2,312 unique mutant sequences`**: `src/enzymado/propertyprep/fireprotdb.py` deduplicates on `['protein_sequence', 'dtm']`. Because FireProtDB links each mutant to multiple independent experiments (`dataset_id = EXPERIMENT_ID`) conducted at different pH values, buffer compositions, or laboratories, **1,026 rows** share an identical `(wildtype_sequence, mutation, protein_sequence)` input tuple with at least one other row in the dataset while carrying a different numerical `dtm` value.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Unmodeled experimental conditions across replicate rows (`1,026` multi-measurement rows)**: Thermal stability depends strongly on assay pH, ionic strength, cosolvents, and protein concentration. Because only sequence inputs are provided to the model, replicate rows for the same mutant with different $\Delta T_m$ values impose a hard irreducible variance floor.
2. **Irreversible thermal aggregation vs. two-state equilibrium unfolding**: Whereas small globular proteins (e.g., T4 lysozyme, barnase) often unfold reversibly, thermal melting of larger enzymes frequently involves kinetically controlled irreversible aggregation, making apparent $T_m$ sensitive to calorimetric scan rate.
3. **Antisymmetry and destabilizing class imbalance**: Models trained solely on direct wild-type $\to$ mutant pairs without reverse-mutation augmentation ($\Delta T_m(\text{mut} \to \text{wt}) = -\Delta T_m(\text{wt} \to \text{mut})$) systematically regress toward the negative training mean ($\approx -2.8\,^\circ\text{C}$), underpredicting rare thermostabilizing mutations (Sanavia et al. 2020, pp. 6–8).

### Quantitative Performance Anchors from the Literature

| Benchmark / Regime | Typical Pearson $r$ | Lower-Bound Error Floor | Source |
| :--- | :---: | :---: | :--- |
| **Theoretical experimental reproducibility ceiling** | **$\sim 0.80$** | **$\sim 1\text{–}2\,^\circ\text{C}$** ($\sim 1\text{ kcal/mol}$) | Sanavia et al. (2020), Section 3.3 (p. 7) |
| **Within-scaffold random mutation split (e.g., mCSM)** | **0.70 – 0.80** | — | Sanavia et al. (2020), Section 3.4 (p. 8) |
| **Per-site / Per-protein cross-scaffold split (<25–30% seq id)** | **0.51 – 0.54** (often $0.20\text{–}0.50$ on blind sets) | — | Sanavia et al. (2020), Section 3.4 (p. 8) |

---

## 3. Specificity & Breadth of the Dataset

- **Parent Scaffold Clustering (`91` Wild-Type Enzymes)**: In `data/processed/properties/dtm/dataset_card.json`, MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) is computed over the **91 unique wild-type parent sequences**, yielding **82 clusters** (`ratio_unique_variants = 0.9011`), of which **74 (90.2%) are singletons** and the maximum cluster size is **2** (mean `1.11`).
- **Mutant Sequence Length Distribution**: Across all **3,338 rows** (`2,312` distinct mutant sequences; 100% retained in `used_in_runs`), lengths span **64 to 1,029 aa** (row-level mean `207.28`, std `148.55`, median `148.0`, IQR `110–265`; distinct-mutant mean `239.13`, std `162.80`, median `164.0`, IQR `117–358`), with only `2` rows exceeding `1,024` aa and `0` rows containing non-standard amino acids.
- **Historical Model-Protein Skew**: The gap between the row-level median length (`148` aa) and the distinct-mutant median (`164` aa) reflects intense mutational saturation of compact model enzymes (such as T4 lysozyme, ribonuclease, and barnase) in the primary biophysical literature.

---

## 4. How the Dataset Was Built

1. **EC-Filtered Download (`scripts/data_prep/download_fireprotdb_ec.py`)**:
   - Exported FireProtDB records carrying a non-empty `EC_NUMBER` field to `data/raw/fireprotdb/fireprotdb_ec.tsv` (`3,515` rows with finite `DTM`).
2. **Substitution Validation & Sequence Construction (`src/enzymado/propertyprep/fireprotdb.py`)**:
   - Filtered `SUBSTITUTION` strings to canonical single-point mutations `^([A-Z])(\d+)([A-Z])$`.
   - Retrieved canonical `wildtype_sequence` from UniProt via `UNIPROTKB`, verified residue identity at the 1-indexed target position, and generated `protein_sequence` via `parsing.apply_substitutions`.
   - Validated amino-acid alphabets and dropped exact duplicates on `['protein_sequence', 'dtm']`, yielding **3,338 rows** across **2,312 unique mutant sequences** and **91 parent enzymes** (`0` UniProt fragments found by `remove_uniprot_fragments.py`).

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Determined via DSC, CD spectroscopy, DSF, or intrinsic fluorescence thermal denaturation curves comparing purified mutant and wild-type enzymes under matched buffer conditions.
- **STRENDA Reporting Gaps**: While FireProtDB records pH and buffer metadata under `EXPERIMENT_ID` (`dataset_id`), those experimental condition columns are not included as model inputs in `fireprotdb_dtm.csv`.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Rows (Mutants / Parents) | Val Rows (Mutants / Parents) | Test Rows (Mutants / Parents) | Parents Straddling Splits | Train Raw Mean ± SD ($^\circ\text{C}$) | Test Standardized Mean ± SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 2,668 (1,850 / 88) | 351 (231 / 46) | 319 (231 / 46) | **60 / 91 (65.9%)** | $-2.8017 \pm 7.3029$ | $-0.0439 \pm 0.9963$ |
| **`similarity_mmseqs2_80_10_10`** | 2,326 (1,599 / 71) | 679 (456 / 10) | 333 (257 / 10) | **0 / 91 (0.0%)** | $-2.6177 \pm 7.1670$ | **$+0.1194 \pm 0.7286$** |

1. **Fundamental Difference in Evaluation Regime (`random` = Within-Scaffold Interpolation vs. `similarity` = Zero-Shot Cross-Scaffold Generalisation)**:
   - Because `propertyprep.py` groups splits by `protein_sequence` (the *mutant* sequence), in **`random_80_10_10`** different single-point mutants of the same parent enzyme are distributed across `train`, `val`, and `test` (**60 of the 91 parent enzymes have mutants in multiple splits**). `random_80_10_10` therefore measures how well a model interpolates additional mutations on an enzyme scaffold already seen during training.
   - In **`similarity_mmseqs2_80_10_10`**, MMseqs2 clustering at 30% sequence identity (`--min-seq-id 0.3`) places all mutants of a parent enzyme ($>99\%$ sequence identity) and any homologous parents ($\ge 30\%$ identity) into a single cluster, ensuring **0 / 91 parent enzymes straddle splits**. `similarity_mmseqs2_80_10_10` therefore tests true zero-shot generalisation to unseen enzyme families.
2. **Validation Split Size & Test Variance Compression in `similarity_mmseqs2_80_10_10`**: Because whole parent-protein mutation libraries are assigned intact across only 91 parent scaffolds, the similarity `val` split contains **679 rows** (20.3% of rows) and the `test` split (`333` rows across `10` parent enzymes) exhibits a narrower standardized standard deviation (`0.7286` vs. `1.0000` in train) and a $+0.1194\,\sigma$ mean shift.

