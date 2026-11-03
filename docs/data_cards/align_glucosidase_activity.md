# Dataset Card: `align_glucosidase_activity` — $\beta$-Glucosidase Single-Mutant $k_{\text{cat}}$ Activity (ALIGN Tournament / D2D)

> ** JSON Companion**: [`align_glucosidase_activity_dataset_card_enriched.json`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/align_glucosidase_activity_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/align_glucosidase_activity/dataset_card.json#/statistics`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_glucosidase_activity/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `align_glucosidase_activity` / `align_glucosidase` |
| **Property (`property_name`)** | `log10_activity` ($\log_{10}(\overline{\text{activity}} + 0.006)$, where raw activity is $k_{\text{cat}}$ in $\text{min}^{-1}$ on pNPG) |
| **Task / Label / Layout** | `protein_global` / `scalar_regression` / **Layout A (protein-only)** |
| **Enzyme & EC Number** | Family 1 Glycoside Hydrolase (GH1) $\beta$-glucosidase — **EC `3.2.1.21`** |
| **Source Organism & Wild-Type ID** | *Paenibacillus polymyxa* $\beta$-glucosidase B (`BglB`, UniProt **`P22505`**, 445 aa, PDB `2O9P`/`2O9T` in D2D & ALIGN release; annotated as *Streptomyces sp.* `Bgl3`, UniProt **`Q59976`** in `card_metadata.py`) |
| **Primary Benchmark Paper** | Armer, C. et al. (2025). *Proteins: Structure, Function, and Bioinformatics*, 93(12), 2005–2014. DOI: `10.1002/prot.70008` ([`align_tournament2025_proteins.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/align_tournament2025_proteins.pdf), [`align_tournament2025_suppinfo.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/align_tournament2025_suppinfo.pdf)) |
| **Upstream Donor & Papers** | **Design to Data (D2D)** undergraduate research program at UC Davis (Vater & Siegel, `https://d2dcure.com/data/?protein=BglB`); Vater et al. (2020) *J. Chem. Educ.* 98, 400–409; Huang et al. (2020) *ACS Omega* 5, 6487–6493; Carlin et al. (2017) *PLoS One* 12, e0176255 |
| **Processed Dataset Size** | **354** unique 445-aa protein variants (from 445 deduplicated measurements; **284** train / **35** val / **35** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`align_glucosidase_activity.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_glucosidase_activity/align_glucosidase_activity.csv) represents one unique 445-residue GH1 $\beta$-glucosidase sequence (`protein_sequence`; `1` wild-type + `353` single-amino-acid substitutions), the number of deduplicated cohort measurements averaged for that sequence (`n_measurements`, equal to `54` for `WT`, `2–3` for 35 repeated single mutants, and `1` for 318 single mutants), and its log-transformed catalytic turnover constant (`log10_activity`).
- **Biochemical meaning of the label**: Raw `activity` is the Michaelis–Menten catalytic rate constant $k_{\text{cat}}$ ($\text{min}^{-1}$) for hydrolysis of the chromogenic substrate *p*-nitrophenyl-$\beta$-D-glucopyranoside (`pNPG`) at $\text{pH } 7.0$ and $25\,^\circ\text{C}$, normalized by the IMAC-purified soluble protein concentration. In [`align_glucosidase_activity.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_glucosidase_activity/align_glucosidase_activity.csv), Enzymado transforms raw values via $\log_{10}(\overline{\text{activity}} + 0.006)$. The target distribution is bimodal (`skewness = -1.036`): **69 unique variants (19.5%)** have raw `activity == 0.0` (catalytically inactive or below the detection limit) and sit at the exact floor $\log_{10}(0.006) = -2.22185$ (`-1.9255` standardized), whereas active variants cluster around the wild-type mean of $117.7\text{ min}^{-1}$ ($\log_{10} \approx 2.07$; overall dataset median $= 1.3664$, max $= 3.8484$ [$7,053\text{ min}^{-1}$]).
- **What predictive success actually demonstrates**: Success on `align_glucosidase_activity` demonstrates a model's ability to separate loss-of-function single-point mutations from tolerated or hyper-activating single substitutions on a $(\beta/\alpha)_8$ TIM-barrel glycoside hydrolase. Because all 353 non-WT variants are single substitutions (`n_mut = 1`), this task evaluates single-variant fitness prediction rather than higher-order epistasis.

---

## 3. Label Noise & Performance Ceiling

- **Direct empirical noise anchor from 58 wild-type (`WT`) replicates**:
  - Because data in the UC Davis D2D Course-based Undergraduate Research Experience (CURE) network were generated across dozens of independent student laboratory cohorts, `train.csv` contains **58 raw `WT` control rows** (`54` distinct measurement rows after exact deduplication).
  - Across those 58 `WT` measurements of the exact same amino-acid sequence, raw $k_{\text{cat}}$ has **$\text{mean} = 117.72\text{ min}^{-1}$**, **$\text{SD} = 90.66\text{ min}^{-1}$ ($\text{CV} = 77.0\%$)**, **$\text{median} = 91.80\text{ min}^{-1}$**, and spans **$31.97$ to $600.61\text{ min}^{-1}$** (an $18.8\text{-fold}$ range across cohorts, corresponding to a standard deviation of **$0.275\;\log_{10}\text{ units}$** on `WT` alone!).
  - Across all **36** sequences (`WT` + 35 single mutants) with $\ge 2$ distinct measurements in `train.csv`, pair-wise replicate rank correlation is **Spearman $\rho = 0.665$** (median within-variant $\text{SD} = 23.23\text{ min}^{-1}$).
- **Additional noise sources & estimated ceiling**:
  - **Denominator error propagation**: Because $k_{\text{cat}} = V_{\max} / [E]$, underestimation of purified protein concentration $[E]$ in poorly expressing mutants can artificially inflate apparent $k_{\text{cat}}$ (e.g., the outlier at $7,052.98\text{ min}^{-1}$).
  - **Estimated ceiling**: Anchored by the empirical inter-cohort replicate correlation ($\rho = 0.665$) and WT batch dispersion ($\sigma_{\text{WT}} = 0.275\;\log_{10}\text{ units}$), the achievable performance ceiling is **Spearman $\rho \approx 0.65\text{–}0.75$** and **$R^2 \approx 0.45\text{–}0.60$**.

---

## 4. Specificity & Breadth

- **Protein diversity**: `354` unique 445-residue sequences (`1` WT + `353` single-point mutants; `0` sequences $>1024$ aa). Note on nomenclature: in the D2D Database (`https://d2dcure.com/data/?protein=BglB`), Armer et al. (2025) Table 1, and Carlin et al. (2017), the 445-aa sequence (`NTFIFPATFMWGTSTSSYQIEGGTDEGGRTPSIWDTFCQIPGKVIGGDCGDVACDHFHHF...`) is *Paenibacillus polymyxa* $\beta$-glucosidase B (`BglB`, UniProt `P22505`, PDB `2O9P`/`2O9T`), whereas [`scripts/data_prep/card_metadata.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/card_metadata.py) labels the family metadata under the homologous GH1 $\beta$-glucosidase *Streptomyces sp.* `Bgl3` (UniProt `Q59976`).
- **Ligand & substrate diversity**: Layout A (no ligand column). All variants are assayed against the same chromogenic substrate, *p*-nitrophenyl-$\beta$-D-glucopyranoside (`pNPG`).
- **Experimental condition diversity**: Single standardized buffer, pH (`7.0`), and temperature ($25\,^\circ\text{C}$) across all rows.

---

## 5. How the Dataset Was Built & Cleaned

1. **Upstream D2D curation & tournament release**: The UC Davis D2D program donated `426` training rows (`58` `WT` replicate rows + `368` single-mutant rows) and `30` test rows (`30` single-mutant rows, which report `activity` only and leave `expression` and `melting point` blank) to the 2023 Protein Engineering Tournament (`456` raw rows total, cited as `"456 sequences"` in Table 1 of Armer et al.).
2. **Deduplication & sequence-level averaging**: In [`scripts/data_prep/real_cases/align_tournament.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/align_tournament.py), [`load_align_rows`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/align_tournament.py#L130-L195) verifies `0` sequence overlaps between `test (with values).csv` and `train.csv` and removes `11` exact duplicate rows (`4` exact duplicate `WT` rows and `7` exact duplicate mutant rows), leaving `445` deduplicated interim rows in `data/interim/align/align_glucosidase.csv` (`415` train + `30` test).
3. **Repeat collapse & $\log_{10}$ transformation**: [`build_align_rows`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/align_tournament.py#L254-L303) averages all repeat measurements of identical 445-aa sequences (`54` distinct `WT` rows $\to 1$ `WT` row; `35` repeated single-mutant sequences averaged across 2–3 rows each), yielding **`354` unique protein sequences**, and applies $\log_{10}(\overline{\text{activity}} + 0.006)$ (`offset = 0.006` for the `69` zero-activity variants).

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (UC Davis D2D Protocol / Metadata) |
| :--- | :--- |
| **Expression & Purification** | *Escherichia coli* BL21(DE3) (`pET29b(+)`, C-terminal $6\times\text{His}$ tag), purified via Ni-NTA immobilized metal affinity chromatography (IMAC) spin columns |
| **Assay Reaction & Substrate** | Michaelis–Menten steady-state hydrolysis of ***p*-nitrophenyl-$\beta$-D-glucopyranoside (pNPG)** to $\beta$-D-glucose and *p*-nitrophenolate |
| **Buffer, pH & Temperature** | **pH 7.0** ($50\text{ mM}$ HEPES / phosphate buffer, $100\text{ mM}$ NaCl) at **$25\,^\circ\text{C}$** |
| **Detection & Observable** | Spectrophotometric absorbance at **$400\text{–}420\text{ nm}$** in 96-well plates; initial velocities fit across a pNPG concentration series and normalized by purified $[E]$ to yield **$k_{\text{cat}}$ ($\text{min}^{-1}$)** |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`align_glucosidase_activity_random_80_10_10_standardized.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_glucosidase_activity/align_glucosidase_activity_random_80_10_10_standardized.csv) partitions the `354` unique variants into **284 train / 35 val / 35 test** via a deterministic (seed 42) 10-decile stratified 80/10/10 random split keyed on unique sequence, with `z`-score standardization fit on the training split ($\mu_{\text{train}} = 0.8931$, $\sigma_{\text{train}} = 1.6214$).
- **Caveats & leakage analysis**:
  - **Zero exact sequence overlap**, with $99.78\%$ sequence identity to WT across all single mutants.
  - **Position-level sharing**: Different amino-acid substitutions at the same sequence position can appear in both train and test.
  - **Small test set ($n = 35$) & 19.5% inactive floor**: Report both Spearman $\rho$ and RMSE, and account for sampling variance on 35 test variants.
