# Human Cytochrome P450 2C9 (CYP2C9) Click-seq Deep Mutational Scan — Catalytic Activity (`cyp2c9_activity`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `6,143` unique variants (`4,913` train / `615` val / `615` test)
> **Primary Citation:** Amorosi, C. J. *et al.* Massively parallel characterization of CYP2C9 variant enzyme activity and abundance. *The American Journal of Human Genetics* **108**, 1735–1751 (2021). [DOI: 10.1016/j.ajhg.2021.07.001](https://doi.org/10.1016/j.ajhg.2021.07.001)
> **Machine-Readable Contract:** [`cyp2c9_activity_dataset_card_enriched.json`](../../docs/data_cards/cyp2c9_activity_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/cyp2c9_activity/dataset_card.json`](../../data/processed/real_case/cyp2c9_activity/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Clinical Pharmacogene DMS via Click-seq:** `cyp2c9_activity` characterizes **6,142 single missense variants plus 1 wild-type sequence (`6,143` total)** across positions 2–490 of **human Cytochrome P450 2C9 (CYP2C9, 490 aa)**, the primary hepatic monooxygenase responsible for clearing $\sim 15\%$ of clinical drugs including the narrow-therapeutic-index anticoagulant *S*-warfarin and the anticonvulsant phenytoin.
- **Mechanism-Based Suicide Probe Readout (TAHA-ABP):** Expressed in a humanized *Saccharomyces cerevisiae* strain (`YMD4256`) carrying chromosomally integrated human cytochrome P450 reductase (CPR) and cytochrome $b_5$, catalytically active CYP2C9 oxidizes the tienilic acid hexynyl amide activity-based probe (**TAHA-ABP**) into a reactive thiophene sulfoxide/epoxide intermediate that covalently alkylates the enzyme. CuAAC click chemistry appends a CF488A fluorophore for 4-gate FACS sorting and UMI-deduplicated barcode sequencing.
- **Coupled Activity + Abundance Readout:** Because Click-seq measures total probe labeling per cell without normalizing for protein abundance, **steady-state protein abundance explains $\sim 56\%$ of the variance in `activity_score`** ($r = 0.748$, $R^2 = 0.56$ across 4,421 missense variants shared with [`cyp2c9_abundance`](../../docs/data_cards/cyp2c9_abundance.md)), while heme-coordinating residues (`Cys435`, `Glu354`, `Arg357`, `Arg97`, `Gly431`, `Arg433`, `Gly437`) selectively abolish activity without destabilizing the fold.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Low-noise 4-replicate UMI-barcoded FACS-seq (`5.8` mean barcodes/variant) | `main` p. 5, 7 |
| **Biological Replicates** | **$n = 4$ biological replicates** (94.1% of variants in all 4 sorts; pairwise **mean Pearson $r = 0.92$**, Spearman $\rho = 0.919$) | `main` p. 7 |
| **Empirical Noise Ceiling** | Median SE = `0.0332` (median SD = `0.0659`) vs. dataset $\sigma = 0.3645$, giving **$R^2_{\max} = 0.9711$** ($r_{\max} = 0.9854$) | Computed from MaveDB `se` column |
| **Gold-Standard Validation** | Monogenic ABPP flow cytometry **$r = 0.991$** ($n = 14$); microsomal LC-MS/MS *S*-warfarin 7-hydroxylation **$r = 0.874$** ($\rho = 0.895$) and phenytoin 4-hydroxylation **$r = 0.764$** ($\rho = 0.87$) | `main` p. 8 |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** *Homo sapiens* Cytochrome P450 2C9 (EC `1.14.14.1` / `1.14.14.51`, UniProt `P11712`, PDB `1R9O`, `1OG5`).
- **Construct:** Untagged 490-aa human CYP2C9 (yeast-codon-optimized, retaining the native N-terminal ER membrane anchor) expressed from a galactose-inducible `CEN/ARS` plasmid in humanized *S. cerevisiae* `YMD4256` (co-expressing human CPR and cytochrome $b_5$).

---

## 4. How the Dataset Was Built & Target Transformation

- **Upstream Definition:** Weighted average across 4 FACS bins ($w_1 = 0.05$, $w_2 = 0.2$, $w_3 = 0.25$, $w_4 = 1.0$, calibrated against 14 monogenic controls at $R^2 = 0.986$), linearly normalized per replicate so that the median synonymous variant equals `1.0` and the median nonsense variant (first 90% of protein) equals `0.0`.
- **Enzymado Ingestion ([`cyp2c9_dms.py`](../../scripts/data_prep/real_cases/cyp2c9_dms.py)):** Reads `data/raw/real_cases/cyp2c9_dms/mavedb_00000095_a_1_scores.csv` (`6,524` rows), maps `p.=` to the 490-aa wild-type sequence (`score = 1.0`), drops `131` nonsense (`Ter`) and `250` single-site synonymous (`=`) rows, retains all `6,142` missense rows (`6,143` total), and z-score standardizes using the training split ($\mu_{\text{train}} = 0.541096$, $\sigma_{\text{train}} = 0.365358$).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Host & Permeabilization** | Humanized *S. cerevisiae* `YMD4256` ($1\ \text{OD}_{600}/\text{mL}$) in $100\,\mu\text{L}$ **PBS (pH 7.4) + $0.1\%$ (w/v) saponin** | `main` p. 4 |
| **Cofactor Pre-Incubation** | **$2\text{ mM}$ NADPH** for $20\text{ min}$ at **$37\,^\circ\text{C}$** (plus endogenous heme $b$, human CPR, and cytochrome $b_5$) | `main` p. 4 |
| **Probe Reaction** | **$10\,\mu\text{M}$ TAHA-ABP** incubated with rotation at **$37\,^\circ\text{C}$ for 20 h** | `main` p. 4 |
| **Click Fluorophore Labeling** | CuAAC reaction ($5\,\mu\text{M}$ CF488A picolyl azide, $1\text{ mM}\ \text{CuSO}_4$, $2\text{ mM}$ THPTA, $3\text{ mM}$ ascorbic acid in PBS) for $30\text{ min}$ at RT | `main` p. 4 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`4,913` train / `615` val / `615` test).
- **Note on `used_in_runs` Metadata:** In `data/processed/real_case/cyp2c9_activity/dataset_card.json`, `used_in_runs.rows` was recorded as `1` (only WT) when the initial summary script intersected against an unpopulated `common_ids` cache; the split parquet files contain all **6,143 sequences**.
