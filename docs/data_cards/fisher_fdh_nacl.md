# Dataset Card: `fisher_fdh_nacl` — Flavin-Dependent Halogenase Family-Wide Chlorination Screen (NaCl)

> **json card**: [`fisher_fdh_nacl_dataset_card_enriched.json`](./fisher_fdh_nacl_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/fisher_fdh_nacl/dataset_card.json`](../../data/processed/real_case/fisher_fdh_nacl/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `fisher_fdh_nacl` |
| **Property Name** | `log10_conversion` |
| **Biochemical Observable** | $\log_{10}(\text{fractional conversion} + 10^{-3})$ for 18 h oxidative C–H chlorination of 62 aromatic/heteroaromatic substrates by 42 genome-mined flavin-dependent halogenases (FDHs) under $100\text{ mM NaCl}$ |
| **Primary Experimental Source** | Fisher, B. F., Snodgrass, H. M., Jones, K. A., Andorfer, M. C. & Lewis, J. C. *Site-Selective C–H Halogenation Using Flavin-Dependent Halogenases Identified via Family-Wide Activity Profiling*. *ACS Central Science* **5**, 1844–1856 (2019), DOI: [`10.1021/acscentsci.9b00835`](https://doi.org/10.1021/acscentsci.9b00835) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/fisher_fdh.py`](../../scripts/data_prep/real_cases/fisher_fdh.py) |
| **Published Split** | `fisher_fdh_nacl_random_80_10_10_standardized` (`2,100` train / `252` val / `252` test; `2,604` total rows) |

---

## 2. How to Think About This Dataset

1. **Complete $42 \times 62$ family-wide C–H chlorination matrix (`2,604` pairs)**: 42 catalytically active, genome-mined flavin-dependent halogenases (FDHs) are profiled against 62 diverse electron-rich aromatic and heteroaromatic substrates (12 small probe indoles/phenols/anilines `1–12` and 50 sterically demanding drugs, alkaloids, and synthetic intermediates `13–62`) in the presence of $100\text{ mM NaCl}$.
2. **Why `fisher_fdh_nacl` (chlorination) and `fisher_fdh_nabr` (bromination) are separate tasks**: Both tasks share the exact same 42 enzyme sequences and 62 organic substrate SMILES. Because the identity of the inorganic halide cosubstrate ($\text{Cl}^-$ vs. $\text{Br}^-$) is a reaction-buffer condition rather than part of the organic substrate SMILES—and because $\text{HOBr}$ is substantially more electrophilic than $\text{HOCl}$—Enzymado separates them into two distinct tasks so models do not receive conflicting target values for identical `(aa_sequence, ligand_smiles)` inputs.
3. **Extreme zero-inflation (`94.82%` at the `-3.0` floor) and `<8%` HTS cutoff**: Because C–H chlorination is thermodynamically more demanding than bromination and because Fisher et al. set all LC-MS conversions below $8\%$ to zero to suppress false positives (SI p. S33), **2,469 of 2,604 rows (`94.82%`)** equal zero (`log10(0 + 1e-3) = -3.0`, standardized `-0.2213`). Only **135 pairs (`5.18%`)** are active ($\ge 8\%$ conversion). Consequently, the minimum, 25th percentile, median, and 75th percentile of every split all equal `-0.2213`.

---

## 3. Label Noise & Performance Ceiling

- **Label-Free UPLC-ESI-MS EIC Readout**: Because synthesizing authentic chlorinated standards for 62 substrates was impractical, fractional conversion was estimated from extracted ion chromatogram (EIC) peak heights (summing $M$ and $M+2$ $^{35}\text{Cl}/^{37}\text{Cl}$ isotopes for monochlorinated and dichlorinated products divided by starting material + product peak heights).
- **Decomposed Noise & Distortion Sources**:
  1. **Equal-ionization assumption in ESI-MS**: Reliable when basic amine groups dominate protonation, but introduces large systematic errors on neutral/phenolic compounds where halogenation shifts ionization efficiency (e.g., carvacrol `6` + FDH `1F12` read as $30\%$ conversion by HTS LC-MS vs. $>95\%$ by calibrated UV-HPLC; SI p. S33).
  2. **Hard `<8%` conversion threshold (`94.8%` censored)**: Creates a discontinuous gap between `0.0` ($\log_{10}(10^{-3}) = -3.0$, standardized `-0.2213`) and `0.08` ($\log_{10}(0.081) = -1.091$, standardized `+2.85`).
  3. **Reduced enzyme loading on 4 low-yield FDHs**: `1B8`, `1C9`, `1E11`, and `1G7` were screened at $1\,\mu\text{M}$ instead of $5\,\mu\text{M}$ due to low purification yields (SI p. S30).
  4. **18-hour endpoint saturation and cofactor-recycling coupling**: Conflates initial turnover rate with multi-hour enzyme stability and $\text{FADH}_2$ regeneration efficiency (`GDH` + `MBP-RebF`).

---

## 4. Specificity, Breadth & Construct Scope

- **Upstream Selection Funnel**: `3,975` FDH sequences in UniProt SSN (`PF04820`) $\to$ `128` maximally diverse genome-mined sequences + `RebH` control $\to$ `87` soluble FDHs co-expressed with `pGro7` (`GroEL/ES`) in *E. coli* BL21(DE3) and purified on 96-well Ni-NTA filter plates $\to$ `42` active sequences ($<1,000$ aa, lengths `415–625` aa) curated in Goldman et al.'s `halogenase_NaCl_binary.csv`.
- **Chemical Scope (`62` substrates)**: 12 small probe substrates (tryptophan, tryptamine, indole derivatives, phenols, anilines) and 50 complex pharmaceuticals, natural products, and synthetic intermediates (carbazoles, $\beta$-carbolines, quinolines, tetrahydroisoquinolines, etc.).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Assay Format** | 384-well plate ($10\,\mu\text{L}$ total volume), shaken at $750\text{ rpm}$ for $18\text{ h}$ at room temperature ($\sim 22\,^\circ\text{C}$), quenched with MeOH |
| **Readout** | Agilent 1290 UPLC / 6130 single-quadrupole ESI-MS EIC peak heights ($M + M+2$ isotopes); conversions $<8\%$ set to $0$ |
| **pH & Buffer** | pH $7.4$, $25\text{ mM}$ HEPES buffer ($\sim 5\%\text{ v/v}$ DMSO from substrate stock) |
| **Halide & Cofactors** | $100\text{ mM NaCl}$ ($1,000\text{ equiv.}$), $100\,\mu\text{M}$ FAD, $100\,\mu\text{M}$ $\text{NAD}^+$, $20\text{ mM}$ glucose, $9\text{ U mL}^{-1}$ GDH, $2.5\,\mu\text{M}$ MBP-RebF, $\text{O}_2$ |
| **Substrate & Enzyme Loading** | $500\,\mu\text{M}$ organic substrate ($1\text{ equiv.}$), $5\,\mu\text{M}$ purified $\text{His}_6$-FDH ($1\,\mu\text{M}$ for `1B8`, `1C9`, `1E11`, `1G7`) |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min / p25 / Median / p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 2,100 | 42 | 62 | `0.0000` | `1.0000` | `-0.2213` | `5.7364` |
| **Val** | 252 | 42 | 62 | `0.0079` | `1.0161` | `-0.2213` | `5.6540` |
| **Test** | 252 | 42 | 62 | `-0.0229` | `0.9537` | `-0.2213` | `5.6761` |
