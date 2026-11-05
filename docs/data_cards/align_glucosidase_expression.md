# Dataset Card: `align_glucosidase_expression` — $\beta$-Glucosidase Single-Mutant Soluble Expression (ALIGN Tournament / D2D)

> ** JSON Companion**: [`align_glucosidase_expression_dataset_card_enriched.json`](../../docs/data_cards/align_glucosidase_expression_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/align_glucosidase_expression/dataset_card.json#/statistics`](../../data/processed/real_case/align_glucosidase_expression/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `align_glucosidase_expression` / `align_glucosidase` |
| **Property (`property_name`)** | `log10_expression` ($\log_{10}(\overline{\text{expression}})$, offset $= 0.0$; soluble IMAC-purified protein concentration) |
| **Task / Label / Layout** | `protein_global` / `scalar_regression` / **Layout A (protein-only)** |
| **Enzyme & EC Number** | Family 1 Glycoside Hydrolase (GH1) $\beta$-glucosidase — **EC `3.2.1.21`** |
| **Source Organism & Wild-Type ID** | *Paenibacillus polymyxa* $\beta$-glucosidase B (`BglB`, UniProt **`P22505`**, 445 aa, PDB `2O9P`/`2O9T` in D2D & ALIGN release; annotated as *Streptomyces sp.* `Bgl3`, UniProt **`Q59976`** in `card_metadata.py`) |
| **Primary Benchmark Paper** | Armer, C. et al. (2025). *Proteins: Structure, Function, and Bioinformatics*, 93(12), 2005–2014. DOI: `10.1002/prot.70008` (`align_tournament2025_proteins.pdf`, `align_tournament2025_suppinfo.pdf`) |
| **Upstream Donor & Papers** | **Design to Data (D2D)** undergraduate research program at UC Davis (Vater & Siegel, `https://d2dcure.com/data/?protein=BglB`); Vater et al. (2020) *J. Chem. Educ.* 98, 400–409; Huang et al. (2020) *ACS Omega* 5, 6487–6493; Carlin et al. (2017) *PLoS One* 12, e0176255 |
| **Processed Dataset Size** | **324** unique 445-aa protein variants (from 415 deduplicated measurements; **264** train / **30** val / **30** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`align_glucosidase_expression.csv`](../../data/processed/real_case/align_glucosidase_expression/b_glucosidase__expression.csv) represents one unique 445-residue GH1 $\beta$-glucosidase sequence (`protein_sequence`; `1` wild-type + `323` single-amino-acid substitutions), the number of deduplicated cohort measurements averaged for that sequence (`n_measurements`, equal to `54` for `WT`, `2–3` for 35 repeated single mutants, and `1` for 288 single mutants), and its log-transformed soluble expression concentration (`log10_expression`).
- **Why $N = 324$ instead of $354$**: In the 2023 Protein Engineering Tournament (`main` Table 1, p. 4), teams were scored only on `Activity` and `Melting point` for the $\beta$-glucosidase B challenge; `expression` was provided as an auxiliary training column in `train.csv` (`426` raw rows $\to 415$ deduplicated rows $\to 324$ unique sequences) and left blank in all `30` rows of `test (with values).csv`.
- **Biochemical meaning of the label & the `1.0` spike**: Raw `expression` measures the soluble protein concentration ($\text{mg/mL}$) of $6\times\text{His}$-tagged $\beta$-glucosidase eluted from Ni-NTA spin columns after heterologous expression in *E. coli* BL21(DE3), quantified via Bradford / $A_{280}$ assay. Because all 415 interim values are strictly positive (`[0.0287, 35.9]`, `0` zeros), Enzymado applies $\log_{10}(\overline{\text{expression}})$ with `offset = 0.0` (range `[-1.3566, +1.5551]`, mean $= -0.1072$, SD $= 0.4534$, median $= 0.0000$, skewness $= -0.2581$). Crucially, **137 of the 415 interim rows (~33%)** have a raw expression value of **exact `1.0`** (`log10_expression = 0.0000`, standardized `+0.2361`), reflecting a subset of D2D entries recorded as WT-normalized (`1.0`) or default placeholder values alongside continuous $\text{mg/mL}$ entries.

---

## 3. Label Noise & Performance Ceiling

- **Direct empirical noise anchor from 58 wild-type (`WT`) replicates**:
  - Across the **58 raw `WT` rows** (`54` distinct rows after exact deduplication) measured across independent student cohorts in `b-glucosidase/train.csv`, raw soluble expression has **$\text{mean} = 1.765\text{ mg/mL}$**, **$\text{SD} = 1.237\text{ mg/mL}$ ($\text{CV} = 70.1\%$)**, **$\text{median} = 1.501\text{ mg/mL}$**, and spans **$0.091$ to $4.552\text{ mg/mL}$** (a **$50.0\text{-fold}$** range across student cohorts!).
  - On the $\log_{10}$ scale, the standard deviation across `WT` replicates alone is **$0.416\;\log_{10}\text{ units}$**—which is **$91.8\%$ as large as the entire dataset's standard deviation across all 324 variants ($0.4534\;\log_{10}\text{ units}$)**!
  - Across all **36** sequences (`WT` + 35 single mutants) with $\ge 2$ distinct expression measurements in `train.csv`, pair-wise replicate correlation is only **Spearman $\rho = 0.229$** ($\text{Pearson } r = 0.012$, mean $|\Delta y| = 1.876\text{ mg/mL}$, median within-variant $\text{SD} = 0.740\text{ mg/mL}$).
- **Estimated performance ceiling**:
  - Because unnormalized student spin-column purification variance ($\sigma_{\text{WT}} = 0.416\;\log_{10}\text{ units}$) and the 137-row spike at `1.0` account for most of the variance in `log10_expression`, the theoretical performance ceiling is low: **Spearman $\rho \approx 0.25\text{–}0.45$** and **$R^2 \approx 0.10\text{–}0.25$**. Low $R^2$ on this task is a direct consequence of the empirical noise floor rather than model failure.

---

## 4. Specificity & Breadth

- **Protein diversity**: `324` unique 445-residue sequences (`1` WT + `323` single-point mutants; `0` sequences $>1024$ aa) of the D2D GH1 $\beta$-glucosidase (*Paenibacillus polymyxa* `BglB`, UniProt `P22505`, PDB `2O9P`/`2O9T`; cross-referenced as *Streptomyces sp.* `Bgl3`, UniProt `Q59976` in `card_metadata.py`).
- **Ligand & substrate diversity**: Layout A (no ligand column; substrate-free measurement of purified protein eluate concentration).
- **Experimental condition diversity**: Single standardized *E. coli* BL21(DE3) expression and Ni-NTA spin-column purification protocol across all rows.

---

## 5. How the Dataset Was Built & Cleaned

1. **Upstream D2D curation & tournament release**: Donated by the UC Davis D2D program in `data/raw/align/b-glucosidase/train.csv` (`426` rows); the `30` rows in `test (with values).csv` have empty `expression` cells.
2. **Exact deduplication**: In [`scripts/data_prep/real_cases/align_tournament.py`](../../scripts/data_prep/real_cases/align_tournament.py), [`load_align_rows`](../../scripts/data_prep/real_cases/align_tournament.py#L130-L195) removes `11` exact duplicate rows in `train.csv`, leaving `415` deduplicated training rows with finite `expression` in `data/interim/align/align_glucosidase.csv`.
3. **Repeat collapse & $\log_{10}$ transformation**: [`build_align_rows`](../../scripts/data_prep/real_cases/align_tournament.py#L254-L303) averages repeat measurements by unique 445-aa sequence (`54` distinct `WT` rows $\to 1$ `WT` row; `35` repeated single mutants averaged across 2–3 rows each), yielding **`324` unique protein sequences**, and applies $\log_{10}(\overline{\text{expression}})$ (`offset = 0.0`).

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (UC Davis D2D Protocol / Metadata) |
| :--- | :--- |
| **Expression System** | *Escherichia coli* BL21(DE3) harboring `pET29b(+)` with C-terminal $6\times\text{His}$ tag |
| **Purification Method** | Chemical/sonication cell lysis followed by **HisPur Ni-NTA spin-column** immobilized metal affinity chromatography (IMAC) |
| **Analytical Readout** | **Bradford protein assay ($595\text{ nm}$) / UV absorbance at $280\text{ nm}$ ($A_{280}$)** on IMAC-purified eluate at **$25\,^\circ\text{C}$** |
| **Reported Units** | Soluble protein concentration in **$\text{mg/mL}$** (with 137 of 415 interim rows recorded at `1.0`), transformed as $\log_{10}(\text{expression})$ |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`align_glucosidase_expression_random_80_10_10_standardized.csv`](../../data/processed/real_case/align_glucosidase_expression/b_glucosidase__expression_random_80_10_10_standardized.csv) partitions the `324` unique variants into **264 train / 30 val / 30 test** via a deterministic (seed 42) 10-decile stratified 80/10/10 random split keyed on unique sequence, with `z`-score standardization fit on the training split ($\mu_{\text{train}} = -0.1072$, $\sigma_{\text{train}} = 0.4534$).
- **Caveats & leakage analysis**:
  - **High experimental noise floor**: $\sigma_{\text{WT}} = 0.416\;\log_{10}\text{ units}$ vs. $\sigma_{\text{dataset}} = 0.4534\;\log_{10}\text{ units}$, and repeat Spearman $\rho = 0.229$.
  - **Spike at `log10_expression = 0.0` (`+0.2361` standardized)**: ~33% of observations sit at raw `expression == 1.0`.
