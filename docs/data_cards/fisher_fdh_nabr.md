# Dataset Card: `fisher_fdh_nabr` — Flavin-Dependent Halogenase Family-Wide Bromination Screen (NaBr)

> **json card**: [`fisher_fdh_nabr_dataset_card_enriched.json`](./fisher_fdh_nabr_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/fisher_fdh_nabr/dataset_card.json`](../../data/processed/real_case/fisher_fdh_nabr/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `fisher_fdh_nabr` |
| **Property Name** | `log10_conversion` |
| **Biochemical Observable** | $\log_{10}(\text{fractional conversion} + 10^{-3})$ for 18 h oxidative C–H bromination of 62 aromatic/heteroaromatic substrates by 42 genome-mined flavin-dependent halogenases (FDHs) under $10\text{ mM NaBr}$ |
| **Primary Experimental Source** | Fisher, B. F., Snodgrass, H. M., Jones, K. A., Andorfer, M. C. & Lewis, J. C. *Site-Selective C–H Halogenation Using Flavin-Dependent Halogenases Identified via Family-Wide Activity Profiling*. *ACS Central Science* **5**, 1844–1856 (2019), DOI: [`10.1021/acscentsci.9b00835`](https://doi.org/10.1021/acscentsci.9b00835) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/fisher_fdh.py`](../../scripts/data_prep/real_cases/fisher_fdh.py) |
| **Published Split** | `fisher_fdh_nabr_random_80_10_10_standardized` (`2,100` train / `252` val / `252` test; `2,604` total rows) |

---

## 2. How to Think About This Dataset

1. **Complete $42 \times 62$ family-wide C–H bromination matrix (`2,604` pairs)**: Profiles the exact same **42 genome-mined FDHs** and **62 aromatic/heteroaromatic substrates** as [`fisher_fdh_nacl`](./fisher_fdh_nacl.md), but using **$10\text{ mM NaBr}$** ($100\text{ equiv.}$) as the halide source instead of $100\text{ mM NaCl}$.
2. **Nearly $4\times$ higher active-pair density than chlorination (`19.39%` vs. `5.18%`)**: Because $\text{HOBr}$ is a much stronger electrophile than $\text{HOCl}$, **505 of 2,604 pairs (`19.39%`)** achieve $\ge 8\%$ bromination conversion (compared to only 135 pairs in `fisher_fdh_nacl`). For this reason, Goldman et al. (2022) used only the `NaBr` matrix (binarized at $\ge 8\%$) in their main-text modeling experiments, whereas Enzymado provides both `fisher_fdh_nabr` and `fisher_fdh_nacl` as continuous scalar regression benchmarks.
3. **Zero-inflation (`80.61%` at the `-3.0` floor) and `<8%` HTS cutoff**: All conversions below $8\%$ were set to `0.0` by Fisher et al. to eliminate false positives (SI p. S33), placing **2,099 of 2,604 rows (`80.61%`)** at $\log_{10}(0 + 10^{-3}) = -3.0$ (`-0.4671` standardized). Thus, the minimum, 25th percentile, median, and 75th percentile of every split are all `-0.4671`.

---

## 3. Label Noise & Performance Ceiling

- **Label-Free UPLC-ESI-MS EIC Readout**: Fractional bromination conversion was calculated from extracted ion chromatogram (EIC) peak heights summing $^{79}\text{Br}$ ($M$) and $^{81}\text{Br}$ ($M+2$) isotope peaks for monobrominated and dibrominated products:
  $$\text{Conversion} = \frac{\text{Br}_1 + \text{Br}_2}{\text{H} + \text{Cl}_1 + \text{Cl}_2 + \text{Br}_1 + \text{Br}_2}$$
- **Decomposed Noise & Distortion Sources**:
  1. **Equal-ionization assumption in ESI-MS**: Assumes starting material and brominated products ionize identically in positive-mode ESI-MS (SI p. S33).
  2. **Competing trace chlorination**: Even after chloride-free purification of FDHs, MBP-RebF, and GDH and lowering NaBr to $10\text{ mM}$, residual trace chloride in cofactor stocks caused minor competing chlorination ($\text{Cl}_1, \text{Cl}_2$), which was included in the denominator (SI p. S33).
  3. **Hard `<8%` conversion threshold (`80.6%` censored)**: Creates a discontinuous gap between `0.0` (`-3.0` raw log10, `-0.4671` standardized) and `0.08` (`-1.091` raw log10, `+1.62` standardized).
  4. **Reduced enzyme loading on 4 low-yield FDHs**: `1B8`, `1C9`, `1E11`, and `1G7` were assayed at $1\,\mu\text{M}$ instead of $5\,\mu\text{M}$ (SI p. S30).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`42` proteins, lengths `415–625` aa)**: Identical to `fisher_fdh_nacl` (42 active FDHs $<1,000$ aa from 87 soluble GroEL/ES-coexpressed FDHs out of 128 SSN-sampled sequences).
- **Chemical Scope (`62` substrates)**: Identical to `fisher_fdh_nacl` (12 probe indoles, phenols, and anilines `1–12` plus 50 sterically demanding pharmaceuticals, natural products, and synthetic intermediates `13–62`).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Assay Format** | 384-well plate ($10\,\mu\text{L}$ total volume), shaken at $750\text{ rpm}$ for $18\text{ h}$ at room temperature ($\sim 22\,^\circ\text{C}$), quenched with MeOH |
| **Readout** | Agilent 1290 UPLC / 6130 single-quadrupole ESI-MS EIC peak heights ($^{79}\text{Br}/^{81}\text{Br}$ $M + M+2$ isotopes); conversions $<8\%$ set to $0$ |
| **pH & Buffer** | pH $7.4$, chloride-free $25\text{ mM}$ HEPES buffer ($\sim 5\%\text{ v/v}$ DMSO from substrate stock) |
| **Halide & Cofactors** | $10\text{ mM NaBr}$ ($100\text{ equiv.}$), $100\,\mu\text{M}$ FAD, $100\,\mu\text{M}$ $\text{NAD}^+$, $20\text{ mM}$ glucose, $9\text{ U mL}^{-1}$ GDH, $2.5\,\mu\text{M}$ MBP-RebF, $\text{O}_2$ |
| **Substrate & Enzyme Loading** | $500\,\mu\text{M}$ organic substrate ($1\text{ equiv.}$), $5\,\mu\text{M}$ purified $\text{His}_6$-FDH ($1\,\mu\text{M}$ for `1B8`, `1C9`, `1E11`, `1G7`) |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min / p25 / Median / p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 2,100 | 42 | 62 | `0.0000` | `1.0000` | `-0.4671` | `2.8185` |
| **Val** | 252 | 41 | 62 | `0.0085` | `1.0235` | `-0.4671` | `2.8090` |
| **Test** | 252 | 42 | 62 | `-0.0252` | `0.9787` | `-0.4671` | `2.7945` |
