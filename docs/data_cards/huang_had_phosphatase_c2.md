# Dataset Card: `huang_had_phosphatase_c2` — HAD Superfamily Phosphatase Substrate Profiling (Cap Type C2 Clade)

> **json card**: [`huang_had_phosphatase_c2_dataset_card_enriched.json`](./huang_had_phosphatase_c2_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/huang_had_phosphatase_c2/dataset_card.json`](../../data/processed/real_case/huang_had_phosphatase_c2/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `huang_had_phosphatase_c2` |
| **Property Name** | `log10_conversion` |
| **Biochemical Observable** | $\log_{10}(\text{OD}_{650} + 10^{-3})$ from 30-min endpoint malachite-green (`BioMol Green`) inorganic phosphate release across 48 $\alpha/\beta$-capped (`C2`) HADSF phosphatases and 165 organophosphate substrates |
| **Primary Experimental Source** | Huang, H. et al. *Panoramic view of a superfamily of phosphatases through substrate profiling*. *PNAS* **112**(16), E1974–E1983 (2015), DOI: [`10.1073/pnas.1423570112`](https://doi.org/10.1073/pnas.1423570112) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/huang_phosphatase.py`](../../scripts/data_prep/real_cases/huang_phosphatase.py) |
| **Published Split** | `huang_had_phosphatase_c2_random_80_10_10_standardized` (`6,240` train / `837` val / `843` test; `7,920` total rows) |

---

## 2. How to Think About This Dataset

1. **Why the HADSF screen is partitioned into `c0`, `c1`, and `c2`**: The full 218-enzyme $\times$ 165-substrate HADSF matrix ($\sim 35,970$ pairs) exceeds Enzymado's 20,000-row single-task limit. [`huang_phosphatase.py`](../../scripts/data_prep/real_cases/huang_phosphatase.py) partitions the superfamily along its structural cap-domain insertion (`Dataset S1`, `Supplemental 2` sheet) into [`huang_had_phosphatase_c0`](./huang_had_phosphatase_c0.md) (`34` enzymes, `5,610` rows), [`huang_had_phosphatase_c1`](./huang_had_phosphatase_c1.md) (`96` `C1.5*` enzymes, `15,840` rows), and **`huang_had_phosphatase_c2`** (`48` **C2** enzymes $\times$ `165` substrates = `7,920` rows).
2. **Structural biology and high promiscuity of the C2 cap domain**: C2 HADSF members contain a large $\alpha/\beta$ cap domain (subtypes `C2A`, archetype NagD from *E. coli* PDB `2C4N`, and `C2B`, archetype $\alpha$-phosphomannomutase 1 from *H. sapiens* PDB `2FUC`, SI p. 51) inserted after strand 2 or 3 of the Rossmann-fold core. In Huang et al. (2015), C2 phosphatases display the highest substrate promiscuity of all three cap architectures (mean $39.07$ hits per enzyme in Table S2 vs. $24.88$ for C1 and $13.27$ for C0, SI p. 50), reflected here by the lowest zero-activity rate (`24.86%`) and a 75th percentile ($\text{OD}_{650} \approx 0.217$) that reaches the $\text{OD}_{650} > 0.2$ active-hit regime.
3. **Narrow kinetic differentiation window ($\text{OD}_{650} \in [0.2, 0.7]$)**: The raw target (`conversion` in Goldman's CSV) is the background-subtracted $650\text{ nm}$ absorbance from the 30-min BioMol Green assay, which detects weak activities down to $k_{\text{cat}} \ge 0.004\text{ s}^{-1}$ ($\text{OD}_{650} > 0.2 \pm 0.03$) and **saturates above $\text{OD}_{650} = 0.7$ ($k_{\text{cat}} \ge 0.014\text{ s}^{-1}$)**. Sub-threshold readings ($0 < \text{OD}_{650} < 0.2$) are retained continuously alongside **1,969 exact-zero cells (`24.86%`, `-1.5090` standardized)**.

---

## 3. Label Noise & Performance Ceiling

- **Duplicate Screening Precision (`n = 2`)**: Every enzyme–substrate pair and no-enzyme background control was assayed in **duplicate** in 384-well plates. Across the entire superfamily screen, the mean difference between duplicates was **$0.0307 \pm 0.0003\text{ OD}$** (SI p. 2).
- **Decomposed Noise & Distortion Sources**:
  1. **Malachite-green saturation above $\text{OD}_{650} = 0.7$ ($k_{\text{cat}} \ge 0.014\text{ s}^{-1}$)**: Compresses moderate and fast turnover onto the same upper plateau (main text p. E1976).
  2. **Log-stretching of sub-threshold baseline noise ($\text{OD}_{650} < 0.2$)**: $\log_{10}(\text{OD}_{650} + 10^{-3})$ stretches $\pm 0.03\text{ OD}$ baseline jitter across $-3.0$ to $-1.3$ log units.
  3. **Spontaneous hydrolysis of labile organophosphates** (e.g., acetyl phosphate, carbamoyl phosphate, pyrophosphates).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`48` C2 proteins, lengths `223–466` aa, median `268.5` aa)**: Selected from 79,778 prokaryotic HADSF sequences in SFLD $\to$ 1,319 cloned targets $\to$ 217 HisTrap + Superdex 200 SEC-purified enzymes $\to$ 48 C2 clade members. `18` of the 48 proteins (`2,970` rows, `37.50%`) are in `used_in_runs`.
- **Chemical Scope (`165` organophosphates)**: Sugar phosphates, nucleotides, deoxynucleotides, phosphonates, amino acid phosphates, and carboxylic-phosphoric anhydrides, with stereochemistry preserved and alkoxide `[O-]` neutralized to `O`.

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Assay Format** | 384-well microplate ($10\,\mu\text{L}$ reaction volume) in duplicate (`n = 2`), quenched after $30\text{ min}$ with $20\,\mu\text{L}$ BioMol Green reagent |
| **Readout** | PerkinElmer EnVision reader at $650\text{ nm}$ after $30\text{ min}$ color development, corrected against duplicate no-enzyme controls |
| **Temperature** | Room temperature ($\sim 25\,^\circ\text{C}$) |
| **pH & Buffer** | pH $7.5$, $25\text{ mM}$ (or $50\text{ mM}$) HEPES, $50\text{ mM NaCl}$, $5\text{ mM MgCl}_2$, $2\text{ mM}$ DTT, $0.01\%$ Triton X-100 |
| **Substrate & Enzyme Loading** | $1\text{ mM}$ organophosphate substrate; $0.5\,\mu\text{M}$ ($5\,\mu\text{M}$ in main text Materials & Methods) HisTrap+SEC-purified enzyme |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min | p25 | Median | p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 6,240 | 48 | 165 | `0.0000` | `1.0000` | `-1.5090` | `-1.5090` | `0.1563` | `0.8180` | `1.9765` |
| **Val** | 837 | 48 | 164 | `-0.0001` | `0.9999` | `-1.5090` | `-1.5090` | `0.1563` | `0.8180` | `1.9739` |
| **Test** | 843 | 48 | 165 | `0.0002` | `0.9994` | `-1.5090` | `-0.2738` | `0.1563` | `0.8199` | `1.9817` |
