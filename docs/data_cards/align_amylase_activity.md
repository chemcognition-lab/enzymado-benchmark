# Dataset Card: `align_amylase_activity` — *Bacillus subtilis* $\alpha$-Amylase Specific Activity (ALIGN Tournament)

> ** JSON Companion**: [`align_amylase_activity_dataset_card_enriched.json`](../../docs/data_cards/align_amylase_activity_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/align_amylase_activity/dataset_card.json#/statistics`](../../data/processed/real_case/align_amylase_activity/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `align_amylase_activity` / `align_amylase` |
| **Property (`property_name`)** | `log10_specific_activity` ($\log_{10}(\overline{\text{specific\_activity}} + 0.001)$) |
| **Task / Label / Layout** | `protein_global` / `scalar_regression` / **Layout A (protein-only)** |
| **Enzyme & EC Number** | Endo-1,4-$\alpha$-D-glucan glucanohydrolase ($\alpha$-amylase AmyE, GH13) — **EC `3.2.1.1`** |
| **Source Organism & Wild-Type ID** | *Bacillus subtilis* — UniProt **`P00692`** (425 aa mature catalytic sequence; PDB `1BAG`, `1UA7`) |
| **Primary Benchmark Paper** | Armer, C. et al. (2025). *Proteins: Structure, Function, and Bioinformatics*, 93(12), 2005–2014. DOI: `10.1002/prot.70008` (`align_tournament2025_proteins.pdf`, `align_tournament2025_suppinfo.pdf`) |
| **Upstream Donor & Preprint** | Donated by **International Flavors and Fragrances (IFF)**; van der Flier et al. (2023) *bioRxiv* `10.1101/2023.09.25.559319v2`, OSF `10.17605/OSF.IO/QRN3C`, and patents EP 2,698,434 A1 / US 8,323,945 B2 / US 8,975,056 B2 / US 9,090,887 B2 |
| **Processed Dataset Size** | **9,031** unique 425-aa protein variants (from 9,069 deduplicated measurements; **7,230** train / **901** val / **900** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`align_amylase_activity.csv`](../../data/processed/real_case/align_amylase_activity/alpha_amylase__specific_activity.csv) represents one unique 425-residue *Bacillus subtilis* $\alpha$-amylase (`AmyE`) variant sequence (`protein_sequence`), the number of deduplicated cross-campaign measurements averaged for that sequence (`n_measurements`, equal to `1` for 8,993 variants and `2` for 38 variants), and its base-10 log-transformed specific activity (`log10_specific_activity`).
- **Biochemical meaning of the label**: Specific activity quantifies the rate of endo-amylolytic corn-starch hydrolysis (reducing ends detected at $562\text{ nm}$ via the bicinchoninic acid [BCA] assay) divided by the secreted soluble enzyme concentration in culture supernatant (measured via UHPLC/HPLC size-exclusion chromatography at $220\text{ nm}$ or Bradford protein assay), normalized relative to the parent/wild-type control. Dividing activity by secreted protein concentration decouples intrinsic catalytic turnover ($k_{\text{cat}}/K_M$-like starch cleavage efficiency per enzyme molecule) from expression yield. A value near `0.0` (`raw ~ 1.0`; dataset median raw $= 0.91$, $\log_{10} = -0.0405$) indicates wild-type-like catalytic efficiency, values up to `+1.285` (`raw = 19.26`) indicate up to $\sim 19\text{-fold}$ hyper-activity, and the lower bound at `-3.000` represents catalytically dead variants (`raw = 0.0`, 274 variants) mapped by the $+0.001$ log offset.
- **What predictive success actually demonstrates**: Achieving high accuracy on `align_amylase_activity` shows that a model can simultaneously distinguish loss-of-function single mutations across the 425-residue GH13 TIM-barrel fold (`7,556` unique variants from `dataset=1`) and predict epistatic catalytic enhancements across rationally combined multi-mutants carrying 2 to 11 simultaneous substitutions (`1,475` unique variants from `dataset=2`). Because the default split is a uniform 80/10/10 random split over all 9,031 variants, test performance reflects a combination of dense single-mutant landscape interpolation and combinatorial multi-mutant regression across shared mutational building blocks.

---

## 3. Label Noise & Performance Ceiling

- **Primary noise sources**:
  1. **Quotient error propagation**: Because specific activity is computed as $\text{Activity}_{\text{BCA}} / \text{Expression}_{\text{SEC}}$, measurement noise in either assay compounds in the ratio, especially for poorly secreted variants where a small denominator inflates ratio variance.
  2. **Cross-campaign (`dataset=1` vs. `dataset=2`) batch shift**: In the raw data, 38 single-mutant sequences were assayed independently in both `dataset=1` (the single-mutant SSM campaign) and `dataset=2` (the rational multi-mutant campaign). Across these 38 cross-campaign repeat pairs, agreement is modest ($\text{Pearson } r = 0.289$, $\text{Spearman } \rho = 0.300$, mean $|\Delta y| = 0.601$, median within-variant $\text{SD} = 0.467$ in raw units), reflecting plate-normalization and fermentation batch drift between separate screening campaigns.
  3. **Left-censored zero-activity pileup at `-3.000`**: Exactly **274** variants have raw `specific activity == 0.0` (3.03% of the dataset). The $\log_{10}(y + 0.001)$ transform places all 274 variants at a discrete point mass at `-3.000`, creating strong negative skewness (`-3.035`) and penalizing $L_2$-based metrics (`RMSE`, $R^2$) whenever a model misclassifies an inactive variant as weakly active.
- **Published reproducibility & estimated ceiling**:
  - Within a single automated campaign, IFF tested two clones per variant in duplicate (four biological replicates) and reported an inter-replicate **Spearman $\rho = 0.92$** for specific activity (`si` p. 9, Supp. Fig. 1)—the highest reproducibility among the three $\alpha$-amylase phenotypes ($\rho = 0.88$ for thermostability, $\rho = 0.79$ for expression).
  - Accounting for the cross-campaign batch offset and the `-3.000` zero-activity spike, the realistic performance ceiling on the standardized 80/10/10 split is **$R^2 \approx 0.80\text{–}0.86$**, **Spearman $\rho \approx 0.88\text{–}0.92$**, and **$\text{RMSE} \approx 0.15\text{–}0.22\;\log_{10}\text{ units}$** on active variants.

---

## 4. Specificity & Breadth

- **Protein diversity**: Single-enzyme mutational scanning benchmark (`9,031` unique variants, all exactly `425` aa in length; `0` sequences $>1024$ aa) on *Bacillus subtilis* AmyE (`P00692`). Combines `7,556` unique single-point mutants (`dataset=1`, ~83.7% of variants) with `1,475` unique combinatorial variants carrying up to 11 mutations (`dataset=2`, ~16.3% of variants).
- **Ligand & substrate diversity**: Layout A (no ligand column). All variants are assayed on the same polymeric substrate: $1\%\text{ (w/v)}$ boiled soluble corn starch.
- **Experimental condition diversity**: Single standardized assay condition across all rows; no environmental condition columns are encoded as model features.

---

## 5. How the Dataset Was Built & Cleaned

1. **Upstream screening & tournament release**: IFF compiled high-throughput *B. subtilis* AmyE screening data from its industrial patent campaigns (EP 2,698,434 A1; US 8,323,945 B2; US 8,975,056 B2; US 9,090,887 B2) and van der Flier et al. (2023) into `train.csv` (`11,724` rows: `8,500` from `dataset=1` and `3,224` from `dataset=2`) and `test (with values).csv` (`570` rows, all from `dataset=2`) for the 2023 Protein Engineering Tournament.
2. **Tournament test-set overlap discovery & removal**: In [`scripts/data_prep/real_cases/align_tournament.py`](../../scripts/data_prep/real_cases/align_tournament.py), [`load_align_rows`](../../scripts/data_prep/real_cases/align_tournament.py#L130-L195) reads `train.csv` followed by `test (with values).csv`. Inspection reveals that **all 570 sequences in `test (with values).csv` were already present in `train.csv`**. Enzymado drops all 570 redundant tournament test rows (`570 tournament test rows already in train`) to prevent duplicate leakage.
3. **Exact measurement deduplication**: Within `train.csv`, `1,750` rows are exact duplicates of existing `(mutated_sequence, expression, thermostability, specific activity)` tuples and are removed, leaving `9,974` unique interim rows in `data/interim/align/align_amylase.csv` (`8,077` in `dataset=1`, `1,897` in `dataset=2`).
4. **Task filtering, repeat averaging, and $\log_{10}$ transformation**: Filtering for non-null `specific activity` retains `9,069` measurements across `9,031` unique sequences (`905` interim rows lack a specific activity value). The `38` sequences with 2 distinct measurements (`dataset=1` and `dataset=2`) are averaged by arithmetic mean (`n_measurements = 2`), and the target is transformed via $\log_{10}(\overline{\text{specific\_activity}} + 0.001)$ (`offset = 0.001` to accommodate the `274` exact zeros).

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (IFF Automated Protocol / Metadata) |
| :--- | :--- |
| **Expression & Sample Prep** | *B. subtilis* cultures grown in microtiter plates (MTPs) for 3 days at $37\,^\circ\text{C}$, 250 rpm with humidified aeration; broth filtered prior to assay |
| **Concentration Assay** | UHPLC/HPLC size-exclusion chromatography (Waters Acquity BEH300 200A SEC column, $25\text{ mM}$ acetate pH 5.8 + $100\text{ mM}$ NaCl at $0.3\text{ mL/min}$, UV $A_{220}$ calibrated against purified $\alpha$-amylase standard; Bradford assay at $25\,^\circ\text{C}$ noted in benchtop metadata) |
| **Starch Hydrolysis Buffer & pH** | $50\text{ mM}$ MOPS buffer, **pH 7.15**, containing $5\text{ ppm Ca}^{2+}$, $20\text{ ppm Na}^{+}$, and $0.002\%\text{ TWEEN-80}$ (benchtop metadata notes pH 5.6) |
| **Substrate & Reaction Volume** | **$1\%\text{ (w/v)}$ boiled corn starch** (Ingredion Inc.) + $\sim 0.05\,\mu\text{g/L}$ enzyme in **$50\,\mu\text{L}$** total volume (Greiner 384-well polypropylene v-bottom MTP) |
| **Incubation Temperature & Time** | **$60\,^\circ\text{C}$ for 30 min** at 1150 rpm in an iEMS heated shaking incubator (benchtop metadata notes $37\,^\circ\text{C}$) |
| **Reducing-End Readout** | **$5\,\mu\text{L}$** reaction mixture + **$30\,\mu\text{L}$ Pierce BCA reagent** in 384-well PCR plate, heated at **$95\,^\circ\text{C}$ for 3 min**, cooled, $25\,\mu\text{L}$ transferred to flat-bottom read plate, optical density measured at **$562\text{ nm}$** |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`align_amylase_activity_random_80_10_10_standardized.csv`](../../data/processed/real_case/align_amylase_activity/alpha_amylase__specific_activity_random_80_10_10_standardized.csv) partitions the `9,031` unique variants into **7,230 train / 901 val / 900 test** via a deterministic (seed 42) 80/10/10 uniform random split keyed on `protein_sequence`, with `z`-score standardization fit on the training split.
- **Caveats & leakage analysis**:
  - **Zero exact sequence overlap**, but all rows share $>97.4\%$ sequence identity ($1\text{–}11$ mutations on a 425-aa scaffold).
  - **Pooled single + multi-mutant regime**: Because the raw Tournament `train.csv` already contained all 570 rows of `test (with values).csv` plus 2,654 additional `dataset=2` multi-mutant rows, Enzymado's random split distributes both `dataset=1` single mutants and `dataset=2` multi-mutants across train, val, and test. Models intending to benchmark strict single-to-multi-mutant generalization should split using `source_dataset` (`1` $\to$ train, `2` $\to$ test) from `data/interim/align/align_amylase.csv`.
