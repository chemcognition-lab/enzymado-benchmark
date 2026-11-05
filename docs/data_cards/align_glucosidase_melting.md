# Dataset Card: `align_glucosidase_melting` — $\beta$-Glucosidase Single-Mutant Melting Temperature $T_m$ (ALIGN Tournament / D2D)

> ** JSON Companion**: [`align_glucosidase_melting_dataset_card_enriched.json`](../../docs/data_cards/align_glucosidase_melting_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/align_glucosidase_melting/dataset_card.json#/statistics`](../../data/processed/real_case/align_glucosidase_melting/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `align_glucosidase_melting` / `align_glucosidase` |
| **Property (`property_name`)** | `melting_point` ($T_m$ in $^\circ\text{C}$, linear scale; `z`-score standardized in split) |
| **Task / Label / Layout** | `protein_global` / `scalar_regression` / **Layout A (protein-only)** |
| **Enzyme & EC Number** | Family 1 Glycoside Hydrolase (GH1) $\beta$-glucosidase — **EC `3.2.1.21`** |
| **Source Organism & Wild-Type ID** | *Paenibacillus polymyxa* $\beta$-glucosidase B (`BglB`, UniProt **`P22505`**, 445 aa, PDB `2O9P`/`2O9T` in D2D & ALIGN release; annotated as *Streptomyces sp.* `Bgl3`, UniProt **`Q59976`** in `card_metadata.py`) |
| **Primary Benchmark Paper** | Armer, C. et al. (2025). *Proteins: Structure, Function, and Bioinformatics*, 93(12), 2005–2014. DOI: `10.1002/prot.70008` (`align_tournament2025_proteins.pdf`, `align_tournament2025_suppinfo.pdf`) |
| **Upstream Donor & Papers** | **Design to Data (D2D)** undergraduate research program at UC Davis (Vater & Siegel, `https://d2dcure.com/data/?protein=BglB`); Huang et al. (2020) *ACS Omega* 5, 6487–6493; Carlin et al. (2017) *PLoS One* 12, e0176255; Vater et al. (2020) *J. Chem. Educ.* 98, 400–409 |
| **Processed Dataset Size** | **106** unique 445-aa protein variants (from 120 deduplicated measurements; **86** train / **10** val / **10** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`align_glucosidase_melting.csv`](../../data/processed/real_case/align_glucosidase_melting/b_glucosidase__melting_point.csv) represents one unique 445-residue GH1 $\beta$-glucosidase sequence (`protein_sequence`; `1` wild-type + `105` single-amino-acid substitutions), the number of deduplicated cohort measurements averaged for that sequence (`n_measurements`, equal to `13` for `WT`, `2` for 2 repeated single mutants, and `1` for 103 single mutants), and its linear-scale melting temperature (`melting_point`, in $^\circ\text{C}$).
- **Why $N = 106$ instead of $354$**: Although Table 1 of Armer et al. (2025) describes the donated $\beta$-glucosidase B dataset as `"456 sequences measured for activity and melting point"`, only `122` raw rows in `train.csv` (`120` after exact deduplication) and `0` rows in `test (with values).csv` contain a non-null `melting point` value. Collapsing the `13` distinct `WT` rows and `2` duplicated single mutants yields **106 unique variants**. Biochemically, Differential Scanning Fluorimetry (DSF) requires sufficient soluble purified protein concentration to yield a clean SYPRO Orange unfolding curve, so poorly expressing or unfolded mutants lack $T_m$ measurements.
- **Biochemical meaning of the label**: `melting_point` ($T_m$, in $^\circ\text{C}$) is the apparent thermal unfolding midpoint determined from the peak of the first derivative $d(\text{Fluorescence})/dT$ in a SYPRO Orange DSF assay at $\text{pH } 7.0$. Across the 106 variants, $T_m$ spans **$36.68\,^\circ\text{C}$ to $54.35\,^\circ\text{C}$** ($\text{mean} = 45.06\,^\circ\text{C}$, $\text{SD} = 2.66\,^\circ\text{C}$, $\text{median} = 45.25\,^\circ\text{C}$, $q_{25} = 43.91\,^\circ\text{C}$, $q_{75} = 46.63\,^\circ\text{C}$, $\text{skewness} = -0.2367$, with `WT` mean $= 45.85\,^\circ\text{C}$).

---

## 3. Label Noise & Performance Ceiling

- **Direct empirical noise anchor from 13 wild-type (`WT`) DSF replicates**:
  - Unlike specific activity or soluble expression yield, $T_m$ is an **intensive midpoint temperature** that does not require dividing by a noisy protein concentration measurement.
  - Across the **13 independent `WT` measurements** in `b-glucosidase/train.csv`, $T_m$ has **$\text{mean} = 45.85\,^\circ\text{C}$**, **$\text{SD} = 1.00\,^\circ\text{C}$**, **$\text{median} = 45.52\,^\circ\text{C}$**, and spans **$44.37\,^\circ\text{C}$ to $47.61\,^\circ\text{C}$**.
  - However, across the **2** single-mutant variants measured twice in `train.csv`, the mean absolute difference is $2.74\,^\circ\text{C}$ (median within-variant $\text{SD} = 2.19\,^\circ\text{C}$), and $50\%$ of all 106 variants in the dataset fall within a narrow **$2.72\,^\circ\text{C}$ interquartile band** ($43.91\text{–}46.63\,^\circ\text{C}$) that is only $\sim 2.7\times$ wider than the $1.00\,^\circ\text{C}$ `WT` noise floor.
- **Estimated performance ceiling**:
  - Using the WT inter-cohort variance ($\sigma_{\text{WT}} = 1.00\,^\circ\text{C}$) against the dataset variance ($\sigma_{\text{total}} = 2.66\,^\circ\text{C}$) gives an upper bound of $R^2_{\max} \approx 1 - (1.00 / 2.66)^2 \approx 0.86$, which tightens to **$R^2 \approx 0.40\text{–}0.65$**, **Spearman $\rho \approx 0.50\text{–}0.70$**, and **$\text{RMSE} \approx 1.0\text{–}1.5\,^\circ\text{C}$** ($0.38\text{–}0.56$ standardized units) when accounting for higher DSF curve-fitting noise on low-concentration mutant eluates and the 10-point test split size.

---

## 4. Specificity & Breadth

- **Protein diversity**: `106` unique 445-residue sequences (`1` WT + `105` single-point mutants; `0` sequences $>1024$ aa) of the D2D GH1 $\beta$-glucosidase (*Paenibacillus polymyxa* `BglB`, UniProt `P22505`, PDB `2O9P`/`2O9T`; cross-referenced as *Streptomyces sp.* `Bgl3`, UniProt `Q59976` in `card_metadata.py`).
- **Ligand & substrate diversity**: Layout A (no ligand column; substrate-free thermal unfolding assay of purified enzyme in the presence of SYPRO Orange reporter dye).
- **Experimental condition diversity**: Single standardized pH (`7.0`) buffer and DSF thermal ramp protocol across all rows.

---

## 5. How the Dataset Was Built & Cleaned

1. **Upstream D2D curation & tournament release**: Donated by the UC Davis D2D program in `data/raw/align/b-glucosidase/train.csv` (`426` rows, of which `122` contain non-null `melting point` values; all `30` rows in `test (with values).csv` have empty `melting point` cells).
2. **Exact deduplication**: In [`scripts/data_prep/real_cases/align_tournament.py`](../../scripts/data_prep/real_cases/align_tournament.py), [`load_align_rows`](../../scripts/data_prep/real_cases/align_tournament.py#L130-L195) removes `11` exact duplicate rows in `train.csv` (including `2` exact duplicate rows among those with `melting point`), leaving `120` deduplicated rows with finite `melting_point` in `data/interim/align/align_glucosidase.csv`.
3. **Repeat collapse & linear scaling**: [`build_align_rows`](../../scripts/data_prep/real_cases/align_tournament.py#L254-L303) averages repeat measurements by unique 445-aa sequence (`13` distinct `WT` rows $\to 1$ `WT` row; `2` repeated single mutants averaged across 2 rows each), yielding **`106` unique protein sequences**, and retains `melting_point` on the linear Celsius scale before train-split `z`-score standardization.

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (UC Davis D2D Protocol / Metadata) |
| :--- | :--- |
| **Expression & Purification** | *Escherichia coli* BL21(DE3) (`pET29b(+)`, C-terminal $6\times\text{His}$ tag), purified via HisPur Ni-NTA spin columns |
| **Assay Method** | **Differential Scanning Fluorimetry (DSF / Thermofluor / Protein Thermal Shift)** with **SYPRO Orange** hydrophobic dye |
| **Buffer & pH** | **pH 7.0** ($50\text{ mM}$ HEPES / phosphate buffer, $100\text{ mM}$ NaCl) |
| **Readout & Observable** | Continuous fluorescence scan across a linear thermal ramp ($\sim 25\text{ to }95\,^\circ\text{C}$); apparent melting temperature **$T_m$ ($^\circ\text{C}$)** determined from the peak of $d(\text{Fluorescence})/dT$ |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`align_glucosidase_melting_random_80_10_10_standardized.csv`](../../data/processed/real_case/align_glucosidase_melting/b_glucosidase__melting_point_random_80_10_10_standardized.csv) partitions the `106` unique variants into **86 train / 10 val / 10 test** via a deterministic (seed 42) 10-decile stratified 80/10/10 random split keyed on unique sequence, with `z`-score standardization fit on the training split ($\mu_{\text{train}} = 45.0592\,^\circ\text{C}$, $\sigma_{\text{train}} = 2.6841\,^\circ\text{C}$).
- **Caveats & leakage analysis**:
  - **Very small test set ($n = 10$)**: Treat single-split test point estimates with caution; a single outlier among the 10 test variants strongly shifts $R^2$.
  - **Survivorship bias**: Only solubly expressing variants capable of producing a clean DSF fluorescence transition are present ($106$ of $354$ BglB variants).
