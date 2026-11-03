# Dataset Card: `martinez_esterase_panel` — Metagenomic Esterase / Lipase Family Substrate Promiscuity Panel

> **json card**: [`martinez_esterase_panel_dataset_card_enriched.json`](./martinez_esterase_panel_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/martinez_esterase_panel/dataset_card.json`](../../data/processed/real_case/martinez_esterase_panel/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `martinez_esterase_panel` |
| **Property Name** | `log10_specific_activity` |
| **Biochemical Observable** | $\log_{10}(\text{specific activity in U g}^{-1} + 10^{-3})$ across 146 microbial serine ester hydrolases and 96 carboxylic esters via a 384-well Phenol Red pH-indicator assay at pH 8.0, $30\,^\circ\text{C}$ |
| **Primary Experimental Source** | Martínez-Martínez, M. et al. *Determinants and Prediction of Esterase Substrate Promiscuity Patterns*. *ACS Chemical Biology* **13**, 225–234 (2018), DOI: [`10.1021/acschembio.7b00996`](https://doi.org/10.1021/acschembio.7b00996) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/martinez_esterase.py`](../../scripts/data_prep/real_cases/martinez_esterase.py) |
| **Published Split** | `martinez_esterase_panel_random_80_10_10_standardized` (`11,388` train / `1,314` val / `1,314` test; `14,016` total rows) |

---

## 2. How to Think About This Dataset

1. **Complete $146 \times 96$ metagenomic esterase promiscuity matrix (`14,016` pairs)**: Profiles **146 diverse microbial esterases/lipases** ($<1,000$ aa, out of 145 environmental ester hydrolases `EH1–EH145` isolated from 28 global metagenomic libraries plus commercial *Pseudozyma aphidis* / *Candida antarctica* lipases `CalA` and `CalB`) against **96 structurally diverse carboxylic esters** (linear, branched, halogenated, aromatic, lactone, chiral, and triacylglycerol esters).
2. **Critical biochemical distinction — Table S3 measures whole wet-cell activity ($\text{U g}^{-1}$ wet cells) for `EH1–EH145`**: In the full $147 \times 96$ matrix (Table S3, digitized by Goldman et al.), `EH1–EH145` were assayed using harvested *E. coli* clones ($0.4\text{ mg}$ wet cells per well) and normalized per gram of **wet cell weight**, whereas `CalA` and `CalB` were normalized per gram of **total commercial protein** (and only the top 28 promiscuous EHs were subsequently purified as $\text{His}_6$-tagged proteins in Table S6; SI p. S5 & p. S20). Therefore, `is_expression_decoupled` is `false`: inter-enzyme activity differences conflate intrinsic catalytic efficiency with recombinant expression level and cell-envelope permeability in *E. coli*.
3. **Heavy zero-inflation (`78.05%` at the `-3.0` floor)**: Only reactions exceeding $>6\times$ background signal were scored as positive hits (SI p. S5). Exactly **10,940 of 14,016 rows (`78.05%`)** are inactive (`0.0 U/g`, mapped to $\log_{10}(0 + 10^{-3}) = -3.0$, standardized `-0.5038`), making the minimum, 25th percentile, median, and 75th percentile equal to `-0.5038` in every split.

---

## 3. Label Noise & Performance Ceiling

- **Replicate Precision**: All 384-well Phenol Red spectrophotometric assays were run in **triplicate** (`n = 3`) using a Tecan Freedom EVO liquid-handling robot, corrected against a negative-control library, with reported triplicate standard deviation **$\text{SD} < 1\%$** (Table S3 legend, SI p. S20).
- **Decomposed Noise & Distortion Sources**:
  1. **Whole-cell expression & permeability confounding (`EH1–EH145` vs. `CalA`/`CalB`)**: `U/g wet cells` scales with intracellular expression yield and substrate transport across the *E. coli* cell envelope.
  2. **Left-censoring below the $>6\times$ background cutoff (`78.05%` zeros)**: Collapses weak and non-cognate activities onto `-3.0` (`-0.5038` standardized).
  3. **Equal mass concentration ($1.14\text{ mg mL}^{-1}$) rather than equal molarity**: Low-MW esters are present at up to $\sim 5\times$ higher molar concentration than high-MW triglycerides.
  4. **Co-solvent partitioning ($4.5\%\text{ v/v}$ acetonitrile for 48 esters vs. DMSO for 48 esters)** and partial emulsion of hydrophobic esters.

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`146` proteins, lengths `196–694` aa)**: Spanning 14 lipolytic families ($\alpha/\beta$-hydrolases and serine $\beta$-lactamase-like esterases) with a median pairwise sequence identity of just **$13.9\%$** ($1.6\%$ to $99.7\%$). One sequence $\ge 1,000\text{ aa}$ from the 147-enzyme Table S3 was excluded by Enzymado's `<1,000 aa` filter.
- **Chemical Scope (`96` carboxylic esters)**: Assayed at $1.14\text{ mg mL}^{-1}$ across alkyl, aryl, halogenated, chiral, and bulky esters.

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Assay Format** | 384-well continuous spectrophotometric Phenol Red pH-indicator assay ($44\,\mu\text{L}$ total volume, $n = 3$ replicates) |
| **Readout** | Synergy HT reader at $550\text{ nm}$ ($\varepsilon = 8,450\text{ M}^{-1}\text{ cm}^{-1}$) every 10 min for 1 h and every 30 min up to 24 h |
| **Temperature** | $30\,^\circ\text{C}$ |
| **pH & Buffer** | pH $8.0$, $5\text{ mM}$ EPPS buffer, $0.45\text{ mM}$ Phenol Red, $4.5\%\text{ v/v}$ acetonitrile or DMSO |
| **Substrate & Biocatalyst Loading** | $1.14\text{ mg mL}^{-1}$ ester; $0.4\text{ mg}$ wet *E. coli* cells per well for `EH1–EH145` ($2\,\mu\text{g}$ protein for `CalA`/`CalB`) |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min / p25 / Median / p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 11,388 | 146 | 96 | `0.0000` | `1.0000` | `-0.5038` | `5.0439` |
| **Val** | 1,314 | 146 | 96 | `-0.0004` | `1.0019` | `-0.5038` | `5.0913` |
| **Test** | 1,314 | 146 | 96 | `0.0056` | `1.0102` | `-0.5038` | `5.1146` |

- **Caveat on `used_in_runs`**: `used_in_runs` is `0` rows (`0.0%`) for `martinez_esterase_panel`; do not filter by `used_in_runs == True`.
