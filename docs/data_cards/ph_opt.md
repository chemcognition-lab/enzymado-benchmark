---
task_id: brenda_phopt
dataset_id: brenda_phopt
property_name: ph_opt
task_type: protein_global
label_type: scalar_regression
layout: Layout A
computed_statistics_ref: data/processed/properties/ph_opt/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkaa1025
domain_review_doi: https://doi.org/10.1101/2023.06.22.544776
---

# Dataset Card: `brenda_phopt` ($\text{pH}_{\text{opt}}$ — Enzyme Optimum pH)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/ph_opt/dataset_card.json`](../../data/processed/properties/ph_opt/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `brenda_phopt`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Paper**: Chang, A. et al. *BRENDA, the ELIXIR core data resource in 2021: new developments and updates.* **Nucleic Acids Research** 49, D498–D508 (2021). DOI: [`10.1093/nar/gkaa1025`](https://doi.org/10.1093/nar/gkaa1025) ([main PDF](../pdfs/brenda_chang2021_nar.pdf)).
- **Domain Review / Companion Benchmark Paper**: Gado, J. E. et al. *Deep learning prediction of enzyme optimum pH.* **bioRxiv** (2023) / **Nature Machine Intelligence** (EpHod). DOI: [`10.1101/2023.06.22.544776`](https://doi.org/10.1101/2023.06.22.544776) ([review PDF](../pdfs/rev_phopt_ephod_gado2023_biorxiv.pdf), [SI PDF](../pdfs/rev_phopt_ephod_gado2023_suppinfo.pdf)).
- **Upstream Repositories**:
  - **BRENDA (Release 2025_1 JSON dump, `data/raw/brenda/brenda_2025_1.json`)**: Source of `ph_optimum` functional annotations, EC numbers, organisms, and protein accession links extracted from $\sim 157,000$ primary literature references.
  - **UniProtKB**: Used by `src/enzymado/propertyprep/brenda.py` (`sequence_getter`) to retrieve canonical amino-acid sequences and by `scripts/data_prep/remove_uniprot_fragments.py` to prune incomplete sequence fragments.
- **License**: BRENDA article published under CC BY 4.0 (main PDF p. 1).

### Reconciliation: Literature Benchmark vs. Enzymado Cleaned Counts

| Metric | EpHod Benchmark (Gado et al. 2023) | Enzymado `overall` (`brenda_phopt.csv`) | Enzymado `used_in_runs` | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Total rows** | 9,855 | **11,139** | **11,113** | Extracted from newer BRENDA 2025_1; retains median when replicate range $\ge 1.0$; removes **325** UniProt fragments |
| **Distinct protein sequences** | 9,855 | **10,794** | **10,770** | `24` ultra-long sequences (up to `7,096` aa) excluded in `used_in_runs` by PLM length cap (max `2,836` aa) |
| **Unique `uniprot_id` strings** | 9,855 | **10,851** | **10,827** | `57` identical sequences shared across secondary/isoform UniProt accessions |
| **Single-measurement rows (`n_measurements == 1`)** | — | **9,249** (83.03%) | **9,226** | `phopt_std` is `NaN` whenever `n_measurements == 1` |
| **Multi-measurement rows (`n_measurements >= 2`)** | — | **1,890** (16.97%) | **1,887** | Carries finite sample standard deviation `phopt_std` |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`brenda_phopt` is a **sequence-only protein-global scalar regression** task (`Layout A`). Given a single enzyme amino-acid sequence (`protein_sequence`), the model predicts the z-score standardized median optimum catalytic pH (`phopt_median_standardized`).
- **Auxiliary columns are diagnostic only**: While `brenda_phopt.csv` contains `n_measurements` (number of underlying BRENDA records) and `phopt_std` (sample standard deviation across replicates; `NaN` in **9,249 / 11,139 = 83.03%** of rows), `src/enzymado/data_loading/data_registry.py` sets `numeric_feature_columns = ()`. No metadata features are passed to models at training or inference time.

### Physical & Mechanistic Meaning of the Target
The optimum pH ($\text{pH}_{\text{opt}}$) is the operational pH at which an enzyme exhibits maximal catalytic activity in vitro. Physically, $\text{pH}_{\text{opt}}$ is controlled by:
1. **Active-site $\text{p}K_a$ values**: General acid/base and nucleophilic catalytic residues (e.g., His, Asp, Glu, Cys, Lys, Tyr) require specific protonation states to stabilize transition states, producing a classic bell-shaped activity profile bounded by $\text{p}K_{a1}$ and $\text{p}K_{a2}$.
2. **Global electrostatic fold stability & substrate ionisation**: Extreme pH values denature the tertiary scaffold or alter substrate protonation. However, as Gado et al. (2023, p. 6) demonstrate, global sequence isoelectric point ($\text{pI}$) and `PROPKA3` folding stability optima explain almost none of the variance in catalytic $\text{pH}_{\text{opt}}$ ($R^2 = 0.029$ and $0.002$), confirming that $\text{pH}_{\text{opt}}$ is an active-site-dominated property.
- **No additional log transformation**: Because $\text{pH} = -\log_{10}[\text{H}^+]$ is already logarithmic, `phopt_median` is standardized directly on the pH scale (`random_80_10_10` train mean $= 7.1939$, std $= 1.1841$; `similarity_mmseqs2_80_10_10` train mean $= 7.2466$, std $= 1.1295$).

### Severe Concentration Around Neutral pH
Nearly **75%** of BRENDA $\text{pH}_{\text{opt}}$ values fall between pH 6.0 and 8.0, and **94%** lie between pH 5.0 and 9.0 (Gado et al. 2023, p. 3). A naive baseline predicting the training mean ($\approx 7.2$) achieves an unweighted RMSE of $\sim 1.18$ pH units while completely missing acidophilic ($\text{pH} < 4$) and alkaliphilic ($\text{pH} > 9$) enzymes.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Single-pH `"assay at"` annotations in BRENDA**: Many BRENDA `ph_optimum` entries carry the commentary `"assay at"`, indicating that the literature study only measured activity at one standard buffer pH (typically pH 7.0–7.5) rather than titrating across pH values. In Gado et al. (2023, SI p. 14), filtering out `"assay at"` records reduced the test set from `1,971` to `993` sequences ($\sim 50\%$ of entries).
2. **Range-to-midpoint collapsing**: When BRENDA reports a broad pH range (e.g., `"6.0-9.0"`), `parse_ph` collapses it to its midpoint (`7.5`), conflating broad pH-tolerant enzymes with narrow-optimum enzymes.
3. **Retained high-variance multi-measurement entries**: Unlike EpHod—which discarded any enzyme whose BRENDA replicate measurements spanned $\ge 1.0$ pH unit—Enzymado retains all entries and takes the **median** (`phopt_median`).
4. **Same sequence with multiple rows (`345` extra rows)**: Because `dataprep.py` groups by `['ec', 'protein_id', 'organism', 'uniprot_from_brenda']` prior to sequence lookup and deduplicates on `['protein_sequence', 'phopt_median', 'n_measurements', 'phopt_std']`, multifunctional enzymes or enzymes curated under multiple EC numbers/accessions appear multiple times with identical `protein_sequence` and different `phopt_median` targets (`11,139` rows vs. `10,794` unique sequences).

### Quantitative Performance Anchors from the Literature

| Model / Baseline | Evaluation Regime | Sample-Weighted $R^2$ | Unweighted / Weighted RMSE (pH units) | Weighted $F_1$ (3-class) | Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **EpHod (ESM-1v + RLA + SVR ensemble)** | BRENDA held-out test ($N=1,971$) | **0.662** | **1.25** (weighted) / **~0.79** (unweighted) | **0.659** | Gado et al. (2023), pp. 4–6, 12 |
| **Frozen ESM-1v + SVR** | BRENDA held-out test ($N=1,971$) | **0.644** | — | — | Gado et al. (2023), Fig. 2 & p. 5 |
| **iFeature + Random Forest** | BRENDA held-out test ($N=1,971$) | **0.415** | — | — | Gado et al. (2023), Fig. 2 & p. 5 |
| **Isoelectric Point ($\text{pI}$) / PROPKA3 Stability** | BRENDA held-out test ($N=1,971$) | **0.029 / 0.002** | — | — | Gado et al. (2023), p. 6 |

---

## 3. Specificity & Breadth of the Dataset

- **Sequence & Evolutionary Diversity**: Across **11,139 rows** (`10,794` unique sequences; `11,113` rows and `10,770` unique sequences in `used_in_runs`), MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) yields **6,534 clusters** (`ratio_unique_variants = 0.6053`; `6,515` clusters in `used_in_runs`), with **4,768 singletons** (73.0% of clusters) and a maximum cluster size of **57** (mean `1.652`).
- **Sequence Lengths**: In `overall`, lengths span **12 to 7,096 aa** (mean `448.03`, std `257.15`, median `406.0`, IQR `306–523`), with `269` rows exceeding `1,024` aa and `19` rows containing non-standard amino-acid characters. In `used_in_runs`, `24` ultra-long proteins ($>2,836$ aa) are excluded by PLM embedding intersection, bounding sequence lengths to **12–2,836 aa** (mean `440.13`, std `215.27`, median `405.0`).

---

## 4. How the Dataset Was Built

1. **JSON Extraction & Range Midpoint Conversion (`scripts/data_prep/dataprep.py`)**:
   - Parsed `ec_data["ph_optimum"]` across all EC numbers in `data/raw/brenda/brenda_2025_1.json`.
   - Converted range strings `"low-high"` to `(low + high) / 2` via `parse_ph`.
   - Grouped by `['ec', 'protein_id', 'organism', 'uniprot_from_brenda']` to compute `phopt_median`, `n_measurements`, and `phopt_std`.
2. **UniProt Sequence Mapping & Fragment Removal (`src/enzymado/propertyprep/brenda.py` & `scripts/data_prep/remove_uniprot_fragments.py`)**:
   - Fetched canonical protein sequences from UniProt, validated standard sequence formatting, and dropped exact duplicates on `[protein_sequence, phopt_median, n_measurements, phopt_std]`.
   - Removed **325** UniProt sequence fragment rows (`259` train, `28` val, `38` test), yielding **11,139** cleaned rows.
   - Partitioned into `random_80_10_10` and `similarity_mmseqs2_80_10_10` grouped by `protein_sequence`, and computed `phopt_median_standardized` from training-split statistics.

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Activity is assayed across buffered pH series via spectrophotometric, fluorometric, or chromatographic product quantification.
- **STRENDA Reporting Gaps**: Assay temperature, buffer chemical species, ionic strength, cosubstrate concentrations, and pre-incubation time are omitted from the tabular dataset, and single-point `"assay at"` records are included alongside full titration curves.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Rows (Proteins) | Val Rows (Proteins) | Test Rows (Proteins) | Train Raw Mean ± SD (pH) | Val Standardized Mean ± SD | Test Standardized Mean ± SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 8,912 (8,635) | 1,114 (1,079) | 1,113 (1,080) | $7.1939 \pm 1.1841$ | $-0.0235 \pm 1.0079$ | $-0.0258 \pm 0.9963$ |
| **`similarity_mmseqs2_80_10_10`** | 8,912 (8,633) | 1,113 (1,081) | 1,114 (1,080) | $7.2466 \pm 1.1295$ | **$-0.2830 \pm 1.1266$** | **$-0.1557 \pm 1.2921$** |

1. **Zero Sequence or Accession Leakage Across Splits**: Both splits group strictly by `protein_sequence`, and `0 / 10,851` `uniprot_id` accessions straddle splits in either `random_80_10_10` or `similarity_mmseqs2_80_10_10`.
2. **Substantial Variance Expansion & Acidic Shift in Similarity Val/Test Splits**: When clustering by 30% sequence identity (`--min-seq-id 0.3`), large mesophilic cytosolic enzyme clusters are concentrated in the training split (`std = 1.1295` pH units), whereas smaller acidophilic and alkaliphilic families are assigned to `val` and `test`. As a result, `similarity_mmseqs2_80_10_10` exhibits a downward shift in standardized mean (`-0.283` in `val`, `-0.156` in `test`) and a **29.2% wider standardized standard deviation in test (`1.2921` vs. `1.0000` in train; i.e., $1.46$ pH units vs. $1.13$ pH units)**. Always evaluate both RMSE/MAE and $R^2$ when comparing `random_80_10_10` against `similarity_mmseqs2_80_10_10`.

