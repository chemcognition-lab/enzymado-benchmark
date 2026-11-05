# Dataset Card: `align_amylase_expression` — *Bacillus subtilis* $\alpha$-Amylase Secreted Expression (ALIGN Tournament)

> ** JSON Companion**: [`align_amylase_expression_dataset_card_enriched.json`](../../docs/data_cards/align_amylase_expression_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/align_amylase_expression/dataset_card.json#/statistics`](../../data/processed/real_case/align_amylase_expression/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `align_amylase_expression` / `align_amylase` |
| **Property (`property_name`)** | `expression` (linear scale, parent-normalized secreted soluble protein concentration; `z`-score standardized in split) |
| **Task / Label / Layout** | `protein_global` / `scalar_regression` / **Layout A (protein-only)** |
| **Enzyme & EC Number** | Endo-1,4-$\alpha$-D-glucan glucanohydrolase ($\alpha$-amylase AmyE, GH13) — **EC `3.2.1.1`** |
| **Source Organism & Wild-Type ID** | *Bacillus subtilis* — UniProt **`P00692`** (425 aa mature catalytic sequence; PDB `1BAG`, `1UA7`) |
| **Primary Benchmark Paper** | Armer, C. et al. (2025). *Proteins: Structure, Function, and Bioinformatics*, 93(12), 2005–2014. DOI: `10.1002/prot.70008` (`align_tournament2025_proteins.pdf`, `align_tournament2025_suppinfo.pdf`) |
| **Upstream Donor & Preprint** | Donated by **International Flavors and Fragrances (IFF)**; van der Flier et al. (2023) *bioRxiv* `10.1101/2023.09.25.559319v2`, OSF `10.17605/OSF.IO/QRN3C`, and patents EP 2,698,434 A1 / US 8,323,945 B2 / US 8,975,056 B2 / US 9,090,887 B2 |
| **Processed Dataset Size** | **9,421** unique 425-aa protein variants (from 9,473 deduplicated measurements; **7,541** train / **940** val / **940** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`align_amylase_expression.csv`](../../data/processed/real_case/align_amylase_expression/alpha_amylase__expression.csv) represents one unique 425-residue *Bacillus subtilis* $\alpha$-amylase (`AmyE`) variant sequence (`protein_sequence`), the count of deduplicated cross-campaign measurements averaged for that sequence (`n_measurements`, equal to `1` for 9,369 variants and `2` for 52 variants), and its linear-scale secreted expression ratio (`expression`).
- **Biochemical meaning of the label**: The label quantifies the concentration of soluble $\alpha$-amylase secreted into the extracellular culture supernatant after 3 days of *B. subtilis* growth in microtiter plates at $37\,^\circ\text{C}$, measured in parts per million (`ppm`) by size-exclusion chromatography (UHPLC/HPLC SEC at $220\text{ nm}$ against a purified $\alpha$-amylase calibrator, or Bradford protein assay) and normalized to the parent control on each plate. Because it measures cumulative extracellular accumulation in a Gram-positive host, it captures the combined effects of translation rate, Sec-translocon export efficiency, co-translational folding, and resistance to extracellular *Bacillus* proteases. Unlike `align_amylase_activity` and `align_amylase_stability`, `expression` is kept on a **linear scale** (`[0.05, 3.194]`, mean $= 0.6929$, SD $= 0.4534$, median $= 0.616$, skewness $= +0.5625$) because the distribution is already compact and strictly positive.
- **What predictive success actually demonstrates**: Success on `align_amylase_expression` shows that a model can predict how single-point mutations (`7,546` variants from `dataset=1`) and rational combinations of up to 11 mutations (`1,875` variants from `dataset=2`) perturb extracellular soluble protein yield in *B. subtilis*. Under the default 80/10/10 random split, test evaluation reflects a mix of dense single-mutant interpolation and combinatorial multi-mutant regression across shared mutational building blocks.

---

## 3. Label Noise & Performance Ceiling

- **Primary noise sources**:
  1. **Microtiter-plate fermentation variance**: Secreted protein accumulation over a 3-day $37\,^\circ\text{C}$ fermentation in 96/384-well plates is inherently more variable than normalized biochemical assays on already-expressed protein, due to well-to-well differences in colony inoculum, dissolved oxygen, evaporation, and cell growth kinetics.
  2. **Cross-campaign (`dataset=1` vs. `dataset=2`) normalization drift**: Across the **52** single-mutant sequences assayed independently in both `dataset=1` and `dataset=2`, inter-campaign correlation is very low ($\text{Pearson } r = 0.073$, $\text{Spearman } \rho = 0.104$, mean $|\Delta y| = 0.602$, median within-variant $\text{SD} = 0.457$ in raw units). Notably, the tournament paper (`main` p. 3) also documents that control protein expression on microtiter plates can experience technical scaling bias (up to $\sim 3\times$ offset relative to variant wells), making plate-to-plate parent normalization a substantial source of batch variance.
  3. **Lower reporting floor at `0.05`**: The minimum reported value across all 9,473 measurements is exactly `0.050` (`-1.4188` standardized), while 553 of the 9,974 interim variants have missing (`NaN`) expression values.
- **Published reproducibility & estimated ceiling**:
  - In the Supplementary Discussion on Data Quality (`si` p. 9, Supp. Fig. 1), IFF and Align to Innovate report an inter-replicate **Spearman $\rho = 0.79$** between biological replicates (Replicate A vs. Replicate B across two clones in duplicate) for expression—the lowest of the three $\alpha$-amylase phenotypes ($\rho = 0.88$ for thermostability, $\rho = 0.92$ for specific activity).
  - Combining the within-campaign biological replicate ceiling ($\rho = 0.79$) with the cross-campaign (`dataset=1` vs. `dataset=2`) batch shift yields an estimated performance ceiling of **Spearman $\rho \approx 0.75\text{–}0.80$** and **$R^2 \approx 0.60\text{–}0.68$**.

---

## 4. Specificity & Breadth

- **Protein diversity**: Single-scaffold mutational landscape (`9,421` unique variants, all exactly `425` aa in length; `0` sequences $>1024$ aa) on *Bacillus subtilis* AmyE (`P00692`). Includes `7,546` unique single-point mutants (`dataset=1`, ~80.1%) and `1,875` unique combinatorial multi-mutants with up to 11 substitutions (`dataset=2`, ~19.9%). Note that 9,421 matches Table 1 of Armer et al. (`9422 sequences measured for expression...`) up to the 1 implicit wild-type reference sequence.
- **Ligand & substrate diversity**: Layout A (no ligand column). Expression is measured directly on filtered culture supernatant prior to any catalytic substrate addition.
- **Experimental condition diversity**: Single standardized fermentation and SEC analytical protocol across all rows; no environmental condition columns are passed to the model.

---

## 5. How the Dataset Was Built & Cleaned

1. **Upstream screening & tournament release**: IFF donated `11,724` training rows (`8,500` in `dataset=1`, `3,224` in `dataset=2`) and `570` test rows (`dataset=2`) from its *B. subtilis* AmyE engineering campaigns (van der Flier et al. 2023; EP 2,698,434 A1; US 8,323,945 B2; US 8,975,056 B2; US 9,090,887 B2).
2. **Tournament test-set overlap removal**: In [`scripts/data_prep/real_cases/align_tournament.py`](../../scripts/data_prep/real_cases/align_tournament.py), [`load_align_rows`](../../scripts/data_prep/real_cases/align_tournament.py#L130-L195) drops all `570` rows from `test (with values).csv` because 100% of their sequences already appear in `train.csv` (`570 tournament test rows already in train`).
3. **Exact measurement deduplication**: `1,750` exact duplicate rows in `train.csv` are removed, leaving `9,974` unique interim rows in `data/interim/align/align_amylase.csv`.
4. **Task filtering, repeat averaging, and linear scaling**: Filtering for finite `expression` retains `9,473` measurements across `9,421` unique sequences (`553` interim rows lack an expression measurement). The `52` sequences with 2 distinct measurements (`dataset=1` and `dataset=2`) are averaged by arithmetic mean (`n_measurements = 2`), and the target is retained on the linear scale (`expression`) before train-split `z`-score standardization.

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (IFF Automated Protocol / Metadata) |
| :--- | :--- |
| **Expression Host & Culture** | *Bacillus subtilis* grown in microtiter plates (MTPs) for **3 days (72 h) at $37\,^\circ\text{C}$** with shaking at **250 rpm** and humidified aeration |
| **Sample Clarification** | Culture broths filtered to remove cells prior to chromatography and biochemical assays |
| **Analytical Technique** | UHPLC/HPLC size-exclusion chromatography (SEC) on a **Waters Acquity BEH300 200A SEC column** (benchtop metadata also notes Bradford protein assay at $25\,^\circ\text{C}$) |
| **Mobile Phase & Flow Rate** | **$25\text{ mM}$ acetate buffer, pH 5.8**, with **$100\text{ mM}$ NaCl** at **$0.3\text{ mL/min}$** |
| **Detection & Calibration** | UV absorbance at **$220\text{ nm}$**; integrated peak area calibrated against a purified $\alpha$-amylase standard of known concentration (reported in `ppm` and normalized to plate parent control) |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`align_amylase_expression_random_80_10_10_standardized.csv`](../../data/processed/real_case/align_amylase_expression/alpha_amylase__expression_random_80_10_10_standardized.csv) partitions the `9,421` unique variants into **7,541 train / 940 val / 940 test** via a deterministic (seed 42) 10-decile stratified 80/10/10 random split keyed on unique sequence, with `z`-score standardization fit on the training split ($\mu_{\text{train}} = 0.6929$, $\sigma_{\text{train}} = 0.4534$).
- **Caveats & leakage analysis**:
  - **Zero exact sequence overlap**, with $>97.4\%$ pairwise sequence identity across all 9,421 variants of *B. subtilis* AmyE.
  - **Pooled single + multi-mutant regime**: Both `dataset=1` single mutants and `dataset=2` combinatorial multi-mutants are present in train, val, and test; use `source_dataset` in `data/interim/align/align_amylase.csv` if evaluating strict single-to-multi-mutant generalization.
