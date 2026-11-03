# Dataset Card: `align_amylase_stability` — *Bacillus subtilis* $\alpha$-Amylase Thermostability (ALIGN Tournament)

> ** JSON Companion**: [`align_amylase_stability_dataset_card_enriched.json`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/align_amylase_stability_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/align_amylase_stability/dataset_card.json#/statistics`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_amylase_stability/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `align_amylase_stability` / `align_amylase` |
| **Property (`property_name`)** | `log10_thermostability` ($\log_{10}(\overline{\text{thermostability}})$, offset $= 0.0$) |
| **Task / Label / Layout** | `protein_global` / `scalar_regression` / **Layout A (protein-only)** |
| **Enzyme & EC Number** | Endo-1,4-$\alpha$-D-glucan glucanohydrolase ($\alpha$-amylase AmyE, GH13) — **EC `3.2.1.1`** |
| **Source Organism & Wild-Type ID** | *Bacillus subtilis* — UniProt **`P00692`** (425 aa mature catalytic sequence; PDB `1BAG`, `1UA7`) |
| **Primary Benchmark Paper** | Armer, C. et al. (2025). *Proteins: Structure, Function, and Bioinformatics*, 93(12), 2005–2014. DOI: `10.1002/prot.70008` ([`align_tournament2025_proteins.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/align_tournament2025_proteins.pdf), [`align_tournament2025_suppinfo.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/align_tournament2025_suppinfo.pdf)) |
| **Upstream Donor & Preprint** | Donated by **International Flavors and Fragrances (IFF)**; van der Flier et al. (2023) *bioRxiv* `10.1101/2023.09.25.559319v2`, OSF `10.17605/OSF.IO/QRN3C`, and patents EP 2,698,434 A1 / US 8,323,945 B2 / US 8,975,056 B2 / US 9,090,887 B2 |
| **Processed Dataset Size** | **8,736** unique 425-aa protein variants (from 8,766 deduplicated measurements; **6,989** train / **874** val / **873** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`align_amylase_stability.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_amylase_stability/align_amylase_stability.csv) represents one unique 425-residue *Bacillus subtilis* $\alpha$-amylase (`AmyE`) variant sequence (`protein_sequence`), the number of deduplicated cross-campaign measurements averaged for that sequence (`n_measurements`, equal to `1` for 8,706 variants and `2` for 30 variants), and its base-10 log-transformed thermostability ratio (`log10_thermostability`).
- **Biochemical meaning of the label**: Thermostability is measured as the **unitless residual activity ratio**—the starch-hydrolysis activity remaining after a high-temperature thermal stress challenge ($85\,^\circ\text{C}$ for 20 min in the generative round protocol; $58.5\,^\circ\text{C}$ for 30 min at pH 5.6 in the benchtop metadata) divided by the initial starch-hydrolysis activity of the same enzyme sample before thermal stress, normalized relative to the parent control on each plate. Because it divides post-stress activity by pre-stress activity of the same supernatant aliquot, it is self-normalized for secreted enzyme concentration and measures **kinetic resistance to irreversible thermal inactivation** (rather than equilibrium $T_m$). Raw values are strictly positive across `[0.008, 49.18]` (median $= 0.86$, `0` zeros), so $\log_{10}(\overline{\text{thermostability}})$ is computed with `offset = 0.0`, producing a nearly symmetric, bell-shaped target distribution ($\text{mean} = -0.1753$, $\text{SD} = 0.5495$, $\text{median} = -0.0655$, $\text{skewness} = -0.0729$, range `[-2.097, +1.692]`).
- **What predictive success actually demonstrates**: Strong performance on `align_amylase_stability` demonstrates that a model can identify both thermally destabilizing single substitutions across the 425-residue GH13 scaffold (`7,560` unique variants from `dataset=1`) and strongly stabilizing multi-mutant combinations (`1,176` unique variants from `dataset=2` reaching up to $49.18\times$ parent residual activity). Under the default 80/10/10 random split, evaluation reflects pooled single-mutant interpolation and shared-background combinatorial regression.

---

## 3. Label Noise & Performance Ceiling

- **Primary noise sources**:
  1. **Post-stress / pre-stress quotient amplification**: Variants with low initial (pre-stress) activity have a small denominator in $\text{Activity}_{\text{post-stress}} / \text{Activity}_{\text{pre-stress}}$, amplifying BCA absorbance noise.
  2. **Cross-campaign (`dataset=1` vs. `dataset=2`) shift**: Across the **30** single-mutant sequences measured in both `dataset=1` and `dataset=2`, inter-campaign agreement is low (mean $|\Delta y| = 1.16$, median within-variant $\text{SD} = 0.738$ in raw units), reflecting differences in thermal challenge stringency and parent residual-activity normalization across screening campaigns.
  3. **High-temperature 384-well thermal ramp & condensation**: Heating enzyme supernatants to $85\,^\circ\text{C}$ for 20 min in 384-well plates followed by $60\,^\circ\text{C}$ starch hydrolysis and $95\,^\circ\text{C}$ BCA detection introduces minor well-position thermal variance.
- **Published reproducibility & estimated ceiling**:
  - Within a single automated campaign, IFF and Align to Innovate report an inter-replicate **Spearman $\rho = 0.88$** between biological replicates (Replicate A vs. Replicate B across two clones in duplicate; `si` p. 9, Supp. Fig. 1).
  - Because `log10_thermostability` has no zero-censoring spike (`0` zeros, skewness $= -0.0729$), both rank and linear metrics are well-behaved, with an estimated performance ceiling of **Spearman $\rho \approx 0.84\text{–}0.88$** and **$R^2 \approx 0.75\text{–}0.83$**.

---

## 4. Specificity & Breadth

- **Protein diversity**: Single-scaffold mutational landscape (`8,736` unique variants, all `425` aa in length; `0` sequences $>1024$ aa) on *Bacillus subtilis* AmyE (`P00692`). Comprises `7,560` unique single mutants (`dataset=1`, ~86.5%) and `1,176` unique combinatorial multi-mutants with up to 11 substitutions (`dataset=2`, ~13.5%). Note that `1,238` of the `9,974` interim variants (including `721` `dataset=2` multi-mutants) lack a finite thermostability value because variants without measurable pre-stress activity cannot yield a residual activity ratio.
- **Ligand & substrate diversity**: Layout A (no ligand column). Pre- and post-stress activities are assayed on $1\%\text{ (w/v)}$ boiled soluble corn starch.
- **Experimental condition diversity**: Single standardized thermal challenge and starch-hydrolysis assay protocol; $\text{Ca}^{2+}$ concentration (which strongly governs GH13 $\alpha$-amylase kinetic stability) is held fixed at $5\text{ ppm}$ in the dilution buffer.

---

## 5. How the Dataset Was Built & Cleaned

1. **Upstream screening & tournament release**: IFF donated `11,724` training rows (`8,500` in `dataset=1`, `3,224` in `dataset=2`) and `570` test rows (`dataset=2`) from its *B. subtilis* AmyE engineering campaigns (van der Flier et al. 2023; EP 2,698,434 A1; US 8,323,945 B2; US 8,975,056 B2; US 9,090,887 B2).
2. **Tournament test-set overlap removal**: In [`scripts/data_prep/real_cases/align_tournament.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/align_tournament.py), [`load_align_rows`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/align_tournament.py#L130-L195) drops all `570` rows from `test (with values).csv` because 100% of their sequences already appear in `train.csv` (`570 tournament test rows already in train`).
3. **Exact measurement deduplication**: `1,750` exact duplicate rows in `train.csv` are removed, leaving `9,974` unique interim rows in `data/interim/align/align_amylase.csv`.
4. **Task filtering, repeat averaging, and $\log_{10}$ transformation**: Filtering for finite `thermostability` retains `8,766` measurements across `8,736` unique sequences (`1,208` interim rows lack a thermostability measurement). The `30` sequences with 2 distinct measurements (`dataset=1` and `dataset=2`) are averaged by arithmetic mean (`n_measurements = 2`), and the target is transformed via $\log_{10}(\overline{\text{thermostability}})$ (`offset = 0.0`).

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (IFF Automated Protocol / Metadata) |
| :--- | :--- |
| **Thermal Stress Challenge** | Filtered enzyme supernatant heated at **$85\,^\circ\text{C}$ for 20 min** prior to substrate addition (benchtop AmyE metadata notes **$58.5\,^\circ\text{C}$ for 30 min** at pH 5.6) |
| **Starch Hydrolysis Buffer & pH** | $50\text{ mM}$ MOPS buffer, **pH 7.15**, containing **$5\text{ ppm Ca}^{2+}$**, $20\text{ ppm Na}^{+}$, and $0.002\%\text{ TWEEN-80}$ |
| **Pre- & Post-Stress Activity Assay** | **$1\%\text{ (w/v)}$ boiled corn starch** + $\sim 0.05\,\mu\text{g/L}$ enzyme in $50\,\mu\text{L}$ (384-well PP v-bottom MTP), incubated **30 min at $60\,^\circ\text{C}$** (1150 rpm) |
| **Reducing-End Readout** | $5\,\mu\text{L}$ reaction + $30\,\mu\text{L}$ **Pierce BCA reagent** heated at **$95\,^\circ\text{C}$ for 3 min**, cooled, and read at **$562\text{ nm}$** |
| **Residual Activity Calculation** | $\text{Thermostability} = \text{Activity}_{\text{after stress}} / \text{Activity}_{\text{before stress}}$ (unitless ratio, normalized to plate parent control) |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`align_amylase_stability_random_80_10_10_standardized.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/align_amylase_stability/align_amylase_stability_random_80_10_10_standardized.csv) partitions the `8,736` unique variants into **6,989 train / 874 val / 873 test** via a deterministic (seed 42) 10-decile stratified 80/10/10 random split keyed on unique sequence, with `z`-score standardization fit on the training split ($\mu_{\text{train}} = -0.1753$, $\sigma_{\text{train}} = 0.5495$).
- **Caveats & leakage analysis**:
  - **Zero exact sequence overlap**, with $>97.4\%$ pairwise sequence identity across all 8,736 variants of *B. subtilis* AmyE.
  - **Pooled single + multi-mutant regime**: Both `dataset=1` single mutants and `dataset=2` combinatorial multi-mutants are distributed across train, val, and test; use `source_dataset` in `data/interim/align/align_amylase.csv` if evaluating strict single-to-multi-mutant generalization.
