---
card_type: dataset
task_id: goase_variant_substrate
dataset_id: goase_variant_substrate
property_name: goase_variant_substrate
track: real_case
task_type: protein_ligand_global
label_kind: scalar_regression
layout: Layout B
doi: 10.1021/acscatal.4c04660
license: CC BY-NC-ND 4.0
source_pdfs:
  - docs/pdfs/goase_supekar2024_acscatal.pdf (main, pp. 1-11)
  - docs/pdfs/goase_supekar2024_suppinfo.pdf (si, pp. 1-36)
  - docs/pdfs/galox_birmingham2021_natcommun.pdf (review, pp. 1-10)
noise_regime: single_lab_replicated
label_noise_floor: Published 10-fold CV ceiling R^2 = 0.72-0.78 (RMSE = 0.90-1.01 ln units across 8 tree/kernel regressors in Supekar et al. 2024 Fig. 4a); plus 0.0635 log10 units RMS error across 46 colliding enantiomeric SMILES rows (S151/S152 and S175/S176)
scope_kind: combinatorial_matrix
computed_statistics_ref: data/processed/real_case/goase_variant_substrate/dataset_card.json#/statistics
---

# `goase_variant_substrate` — *Fusarium* galactose oxidase (`GOase` $\text{M}_{3\text{-}5}$) variant-by-substrate activity matrix (34 variants $\times$ 56 bulky secondary alcohols)

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/goase_variant_substrate/dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/goase_variant_substrate/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is a combinatorial enzyme-variant $\times$ substrate matrix (`Layout B`: 1,183 unique `(protein_sequence, smiles)` rows across 34 engineered 639-aa `GOase` variants and 54 unique canonical SMILES strings).** Supekar et al. (*ACS Catalysis* 2024) screened **34 purified variants** of the laboratory-evolved *Fusarium* NRRL 2903 galactose oxidase $\text{M}_{3\text{-}5}$ parent (`GOh1001b`, 27 single/double variants `GOh1002–GOh1052`, and 6 triple variants `GOh2001–GOh2006` built on `GOh1052` [`F217A, N268W`]; `SI Table S5`, p. 35) against **56 pharmaceutically relevant bulky secondary alcohol substrates** (`S001–S199`, `Table 1` & `SI Table S2`) using a 24-hour 96-well ABTS/HRP-coupled colorimetric oxidation assay at $405\text{ nm}$.
2. **Two pairs of opposite enantiomers (`S151/S152` and `S175/S176`) were assigned the exact same chiral SMILES string in `panel56.smi`, collapsing 1,229 raw rows into 1,183 unique rows.** In the authors' raw file `data/raw/goase/panel56.smi`, both `S151` and `S152` ((R)- and (S)-1-phenylbutan-1-ol) are written as the identical SMILES **`CCC[C@H](O)c1ccccc1`** (24 variants tested on both), and both `S175` and `S176` ((R)- and (S)-2-chloro-1-phenylethan-1-ol) are written as the identical SMILES **`O[C@@H](CCl)c1ccccc1`** (22 variants tested on both). When [`goase_supekar.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/goase_supekar.py#L43-L79) deduplicates by `(protein_sequence, smiles)`, those **46 enantiomeric pairs are averaged together**, reducing `1,229` raw rows in `dataset_56-subs_34-vars.csv` to **1,183 unique `(protein_sequence, smiles)` rows** across **54 unique canonical SMILES**.
3. **The benchmark target (`log10_activity`) is double-logarithmically transformed ($\log_{10}(\ln(\text{Rate} \times \text{MaxAbs} \times 100 + 1) + 0.001)$).** In `data/raw/goase/dataset_56-subs_34-vars.csv`, the column `labels` was *already* natural-log transformed by the paper authors as $\ln(\text{Absorbance} \times \text{Slope} \times 100 + 1)$ (`Main p. 8`), ranging from `0.0` (12 inactive pairs) to `9.4766`. During Enzymado ingestion, [`goase_supekar.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/goase_supekar.py#L82-L106) stores `labels` as `activity` and applies a **second** logarithmic transformation `log10_activity = log10(activity + 0.001)` (`offset = 1e-3`). This compresses the 1,171 active pairs into $[-0.2505, +0.9767]$ (standardized $[-0.1782, +1.1515]$) while mapping the **12 exact-zero rows (`activity = 0.0`) to `-3.0000` (standardized `-3.1577`)**, creating a sharp left-tail outlier cliff ($\text{skewness} = -1.033$).
4. **Under the point-random 80/10/10 split, 100% of the 34 variants and 100% of the 54 SMILES appear in `train`.** Because splitting is performed over matrix cells (`random_split_unit="point"`), every variant is observed $\sim 28$ times in `train` and every substrate is observed $\sim 18$ times in `train`. Thus `goase_variant_substrate_random_80_10_10_standardized` evaluates **matrix completion** (imputing missing entries in a $34 \times 54$ reactivity matrix), **not** unseen-protein or unseen-substrate generalization.

**Practical consequences for use:**
- **Never pass the intermediate column `activity` as a model feature:** Both `activity` ($\ln(\text{Rate}\times\text{MaxAbs}\times 100 + 1)$) and `log10_activity` ($\log_{10}(\text{activity} + 0.001)$) are stored in `goase_variant_substrate.csv`; only `protein_sequence` and `smiles` are valid inputs.
- **Beware of the 12 exact-zero rows (`-3.16 SD`) across splits:** `train` contains 10 exact-zero rows, `val` contains **0** exact-zero rows (plus 1 row where an active enantiomer was averaged with a zero enantiomer at `-2.79 SD`), and `test` contains **2** exact-zero rows (`-3.16 SD`). Always report Spearman $\rho$ and MAE alongside Pearson $R^2$ and RMSE.
- Never pass `dataset_id` (`goase_00001` … `goase_01183`) as a feature.

---

## 1. Label noise and the measurement floor

### 1.1 Experimental quality controls and noise anchors

| Property | Value | Evidence |
|---|---|---|
| **Enzyme purification & normalization** | All 34 GOase variants were Ni-NTA affinity purified (with $\text{CuSO}_4$ reconstitution), 30 kDa buffer-exchanged, BCA-quantified, and assayed at **$20\,\mu\text{g/mL}$** ($\sim 0.29\,\mu\text{M}$) | *"Substrate (1 mM), ABTS (2.5 mM), horseradish peroxidase (HRP (Toyobo), 25 µg/mL), and purified galactose oxidase (20 μg/mL) in a final volume of 200 μL with final concentration of dimethylsulfoxide (DMSO) at 5% v/v in 100mM sodium phosphate buffer (pH 7.0)."* `[Source: SI, Page 18]` |
| **Kinetic readout schedule** | 43 timepoints over 24 h at $405\text{ nm}$ (every 2 min for 1 h, every 10 min for 2 h, every 30 min for 5 h, every 2 h for 16 h) | *"HTP colorimetric assays were carried out using 96-well clear shallow well plates at room temperature and read at 405 nm using a microplate reader (Tecan Infinite M200 Pro) for the following timepoints: every 2 min for 1 h, every 10 min for next 2 h, every 30 min for next 5 h, every 2 h for next 16 h for a total of 24 h."* `[Source: SI, Page 18]` |
| **Published 10-fold CV regression ceiling** | **$R^2 = 0.72\text{–}0.78$**, $\text{RMSE} = 0.90\text{–}1.01$ (on the natural-log `activity` scale across Extra Trees, Random Forest, XGBoost, and Gradient Boosting in Fig. 4a) | *"Comparison of model performance showed that decision tree-based models, specifically gradient boosting (GB), random forest (RF), extreme gradient boosting (XGB), and extra trees (ET) regressors, generally outperformed other standard algorithms in regression (R2: 0.72−0.78; RMSE: 0.90−1.01; Figure 4a)."* `[Source: Main, Page 4]` |
| **Enantiomeric SMILES collision error (46 rows)** | Mean $|y_1 - y_2| = 0.449\text{ ln units}$ ($0.068\text{ log}_{10}\text{ units}$, RMS half-difference $= 0.0635\text{ log}_{10}\text{ units}$) across the 46 merged `S151/S152` and `S175/S176` pairs | Computed directly across the 46 duplicate `(protein_sequence, smiles)` pairs in `dataset_56-subs_34-vars.csv` |

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`combinatorial_matrix`):** **1,183 unique `(protein_sequence, smiles)` rows** spanning **34 engineered *Fusarium* NRRL 2903 galactose oxidase (`GOase`) variants** (639 aa, **EC 1.1.3.9**, radical-copper alcohol oxidase; wild-type reference structure PDB `1GOG`, UniProt `P0CS93`) and **54 unique canonical SMILES** (56 tested secondary alcohol substrates).
- **Protein panel composition (34 variants, 1–3 mutations from $\text{M}_{3\text{-}5}$):**
  - **Parent (`GOh1001b`):** Laboratory-evolved $\text{M}_{3\text{-}5}$ variant (`S10P, M70V, P136P, G195A, V494A, N535D` relative to WT *Fusarium* GOase, which unlocked broad activity on secondary benzylic alcohols; `Main p. 3`).
  - **27 single and double active-site mutants (`GOh1002–GOh1052`):** Mutating 14 first/second-shell pocket residues (`R132, C185, V193, A195, F217, S237, W239, N259, N268, Q269, Q294, Y325, Q330, H334`; `SI Table S5`, p. 35).
  - **6 triple mutants (`GOh2001–GOh2006`):** Built by adding `V193A`, `A195G`, `N259C`, `Q294A`, `Y325C`, or `Y325L` onto the best double mutant `GOh1052` (`F217A, N268W`), which enlarges the active-site cavity to accommodate diarylmethanols and bulky heterocyclic alcohols (`Main p. 3`, `SI Table S5`, p. 35).
  - All 34 sequences have exact length **639 aa** (100% canonical amino acids, $\ge 99.53\%$ pairwise identity, 1 MMseqs2 cluster at 50% identity).
- **Chemical scope (`has_ligand_smiles: true`, Layout B):** 54 unique canonical SMILES strings (56 substrate IDs `S001–S199`, MW $122\text{–}315\text{ Da}$; `Table 1` & `SI Table S2`) spanning substituted 1-phenylethanols, 1-phenylpropanols, 1-phenylbutanols, $\alpha$-tetralols, indanols, benzhydrols, and pyridyl/thiophenyl/naphthyl secondary alcohols.

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | 96-well microplate HRP/ABTS-coupled colorimetric oxidation assay at $405\text{ nm}$ | No | `reported` | *"Interaction between 2,2′-azino-bis(3-ethylbenzothiazoline-6-sulfonic acid) (ABTS), horseradish peroxidase (HRP), and hydrogen peroxidase (H2O2) generated from the reaction yielded green ABTS radical cations suitable for colorimetric assay (Figure S13).31"* `[Source: Main, Page 8]` |
| **Analytical readout** | $\text{Colorimetric Activity} = \text{Rate} \times \text{Maximum Absorbance} \times 100$ at $405\text{ nm}$ on a Tecan Infinite M200 Pro reader | No | `reported` | *"Colorimetric activities were calculated from the 2 parameters of maximum absorbance and rate measured via 405 nm absorbance data gathered by microplate reader (Tecan) according to the following formula: Colormetric Activity = Rate * Maximum Absorbance * 100"* `[Source: SI, Page 18]` |
| **Expression & purification** | Expressed in *E. coli* from `pET-30a(+)` in AIM-TB + $0.8\text{ mM CuSO}_4$ ($18\,^\circ\text{C}$, 16 h), Ni-NTA purified with $0.4\text{ mM CuSO}_4$, 30 kDa buffer-exchanged, BCA-normalized to **$20\,\mu\text{g/mL}$** | No | `reported` | *"After inoculation of 4% v/v overnight culture into 250 mL Auto-Induction Media Terrific Broth (AIM-TB, Formedium) containing 50 μg/mL kanamycin and 0.8 mM sterile copper sulfate in 1L baffled flasks, the expression cultures were grown at 37 °C, 200 rpm for 4 h before cooling down to 18 °C, 200 rpm for 16 h."* `[Source: SI, Page 16]` |
| **Expression decoupled from label** | **true** (all variants purified and assayed at uniform $20\,\mu\text{g/mL}$ concentration) | — | `reported` | SI p. 18 |
| **Temperature ($T$)** | **Room temperature** ($\sim 22\text{–}25\,^\circ\text{C}$) | No (constant) | `reported` | *"HTP colorimetric assays were carried out using 96-well clear shallow well plates at room temperature"* `[Source: SI, Page 18]` |
| **pH & buffer** | **pH 7.0**, **$100\text{ mM}$ sodium phosphate buffer** with **$5\%\text{ v/v}$ DMSO**, $2.5\text{ mM}$ ABTS, $25\,\mu\text{g/mL}$ HRP ($200\,\mu\text{L}$ final volume) | No (constant) | `reported` | *"Substrate (1 mM), ABTS (2.5 mM), horseradish peroxidase (HRP (Toyobo), 25 µg/mL), and purified galactose oxidase (20 μg/mL) in a final volume of 200 μL with final concentration of dimethylsulfoxide (DMSO) at 5% v/v in 100mM sodium phosphate buffer (pH 7.0)."* `[Source: SI, Page 18]` |
| **Cofactors & metals** | Mononuclear **$\text{Cu(II)}$** center + autocatalytic **$\text{Tyr272–Cys228}$ radical cofactor**; ambient $\text{O}_2$ cosubstrate | No (constant) | `reported` | SI p. 16–17; PDB `1GOG` |
| **Substrate loading** | **$1\text{ mM}$ secondary alcohol substrate** + $20\,\mu\text{g/mL}$ GOase + $2.5\text{ mM}$ ABTS + $25\,\mu\text{g/mL}$ HRP | **Yes (`smiles`)** | `reported` | SI p. 18 |
| **Reaction time** | **24 h** total monitoring across 43 timepoints | No (constant) | `reported` | *"for the following timepoints: every 2 min for 1 h, every 10 min for next 2 h, every 30 min for next 5 h, every 2 h for next 16 h for a total of 24 h."* `[Source: SI, Page 18]` |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation ($1,904 \to 1,325 \to 1,229 \to 1,183$)

1. **Theoretical grid:** $34\text{ variants} \times 56\text{ substrates} = \mathbf{1,904}$ pairs.
2. **Paper text vs. Zenodo CSV (`1,325` vs. `1,229` rows):** While Main p. 9 states *"Our activity data set based on 34 variants and 56 substrates comprised 1325 variant−substrate examples"*, the official CSV deposited by the authors (`data/raw/goase/dataset_56-subs_34-vars.csv`) contains **1,229 rows** across the 34 variants and 56 substrate codes (`S001–S199`).
3. **Enantiomeric SMILES collision in `panel56.smi` (`1,229` $\to$ `1,183` rows):** In `data/raw/goase/panel56.smi`, two pairs of opposite enantiomers were recorded with identical SMILES strings:
   - `S151` and `S152` both map to `CCC[C@H](O)c1ccccc1` (24 variants tested on both $\to$ 24 collisions);
   - `S175` and `S176` both map to `O[C@@H](CCl)c1ccccc1` (22 variants tested on both $\to$ 22 collisions).
   When [`goase_supekar.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/goase_supekar.py#L72-L77) calls `collapse_duplicate_rows(rows, key_columns=("protein_sequence", "smiles"), target_columns=("activity",))`, those $24 + 22 = \mathbf{46}$ duplicate pairs are averaged, yielding **1,183 unique `(protein_sequence, smiles)` rows**.

### 4.2 Double-logarithmic target transformation

$$\text{activity} = \ln(\text{Rate} \times \text{MaxAbs} \times 100 + 1), \qquad \text{log10\_activity} = \log_{10}(\text{activity} + 0.001)$$

followed by train-split z-score standardization ($\mu_{\text{train}} = -0.0859$, $\sigma_{\text{train}} = 0.9228$ across the 965 training rows). Because `activity` is already natural-log transformed, applying $\log_{10}(\text{activity} + 0.001)$ compresses the 1,171 active rows into $[-0.2505, +0.9767]$ while placing the 12 exact-zero rows at $-3.0000$ (`-3.1577` standardized).

---

## 5. Split semantics and leakage caveats

One split file is published: `goase_variant_substrate_random_80_10_10_standardized` ($965\text{ train} / 109\text{ val} / 109\text{ test} = 1,183\text{ total}$; see [`dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/goase_variant_substrate/dataset_card.json) for exact split moments).

1. **Matrix-completion split (`random_split_unit="point"`):** All 34 variants and all 54 unique SMILES appear in `train`. Evaluating out-of-distribution generalization to unseen variants or unseen substrates requires custom cold-protein or cold-ligand splits.
2. **Target-column leakage guard:** `goase_variant_substrate.csv` contains both `activity` and `log10_activity`; never pass `activity` as a model feature.
3. **Sequential `dataset_id` ordering:** `goase_00001` … `goase_01183` are ordered by variant and substrate ID and must never be passed as a feature.
