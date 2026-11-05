# Dataset Card: `huang_had_phosphatase_c0` — HAD Superfamily Phosphatase Substrate Profiling (Cap Type C0 Clade)

> **json card**: [`huang_had_phosphatase_c0_dataset_card_enriched.json`](./huang_had_phosphatase_c0_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/huang_had_phosphatase_c0/dataset_card.json`](../../data/processed/real_case/huang_had_phosphatase_c0/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `huang_had_phosphatase_c0` |
| **Property Name** | `log10_conversion` |
| **Biochemical Observable** | $\log_{10}(\text{OD}_{650} + 10^{-3})$ from 30-min endpoint malachite-green (`BioMol Green`) inorganic phosphate release across 34 capless (`C0`) HADSF phosphatases and 165 organophosphate substrates |
| **Primary Experimental Source** | Huang, H. et al. *Panoramic view of a superfamily of phosphatases through substrate profiling*. *PNAS* **112**(16), E1974–E1983 (2015), DOI: [`10.1073/pnas.1423570112`](https://doi.org/10.1073/pnas.1423570112) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/huang_phosphatase.py`](../../scripts/data_prep/real_cases/huang_phosphatase.py) |
| **Published Split** | `huang_had_phosphatase_c0_random_80_10_10_standardized` (`4,420` train / `594` val / `596` test; `5,610` total rows) |

---

## 2. How to Think About This Dataset

1. **Why the HADSF screen is partitioned into `c0`, `c1`, and `c2`**: The full 218-enzyme $\times$ 165-substrate HADSF matrix ($\sim 35,970$ pairs) exceeds Enzymado's 20,000-row single-task limit. Rather than subsampling at random, [`huang_phosphatase.py`](../../scripts/data_prep/real_cases/huang_phosphatase.py) splits the superfamily along its fundamental structural and evolutionary boundary—the **cap domain insertion** (`Dataset S1`, `Supplemental 2` sheet)—into three disjoint tasks:
   - **`huang_had_phosphatase_c0`**: Capless / minimal-insert **C0** clade (`34` enzymes $\times$ `165` substrates = `5,610` rows).
   - **[`huang_had_phosphatase_c1`](./huang_had_phosphatase_c1.md)**: $\alpha$-helical **C1 (`C1.5*`)** cap clade (`96` enzymes $\times$ `165` substrates = `15,840` rows).
   - **[`huang_had_phosphatase_c2`](./huang_had_phosphatase_c2.md)**: $\alpha/\beta$ **C2 (`C2A`/`C2B`)** cap clade (`48` enzymes $\times$ `165` substrates = `7,920` rows).
2. **Structural biology of the C0 clade**: Because C0 phosphatases lack a mobile cap domain over the Rossmann-fold catalytic cleft, solvent exclusion relies on oligomeric interfaces or bulky substrates (Fig. S22, SI p. 72). Consequently, C0 enzymes have the narrowest small-molecule substrate breadth of the three clades (mean $13.27$ hits per enzyme in Table S2 vs. $24.88$ for C1 and $39.07$ for C2; SI p. 50).
3. **Narrow kinetic differentiation window ($\text{OD}_{650} \in [0.2, 0.7]$)**: The raw target (`conversion` in Goldman's CSV) is the background-subtracted $650\text{ nm}$ absorbance from the 30-min BioMol Green assay. Huang et al. show that the assay detects weak activities down to $k_{\text{cat}} \ge 0.004\text{ s}^{-1}$ ($\text{OD}_{650} > 0.2 \pm 0.03$) but **saturates above $\text{OD}_{650} = 0.7$ ($k_{\text{cat}} \ge 0.014\text{ s}^{-1}$)**, compressing all moderate and fast enzymes onto the same upper plateau. Meanwhile, sub-threshold readings ($0 < \text{OD}_{650} < 0.2$) are retained continuously alongside **1,908 exact-zero cells (`34.01%`, `-1.2127` standardized)**.

---

## 3. Label Noise & Performance Ceiling

- **Duplicate Screening Precision (`n = 2`)**: Every enzyme–substrate pair and no-enzyme background control was assayed in **duplicate** in 384-well plates. Across the entire superfamily screen, the mean difference between duplicates was **$0.0307 \pm 0.0003\text{ OD}$** (SI p. 2).
- **Decomposed Noise & Distortion Sources**:
  1. **Malachite-green saturation above $\text{OD}_{650} = 0.7$ ($k_{\text{cat}} \ge 0.014\text{ s}^{-1}$)**: Prevents differentiation of moderate vs. high catalytic rates (main text p. E1976).
  2. **Log-stretching of sub-threshold baseline noise ($\text{OD}_{650} < 0.2$)**: Because Goldman's CSV retains raw background-corrected $\text{OD}_{650}$ values down to $0.001$ alongside `0.0` (`34.01%`), $\log_{10}(\text{OD}_{650} + 10^{-3})$ stretches $\pm 0.03\text{ OD}$ baseline noise across $-3.0$ to $-1.3$ log units.
  3. **Spontaneous hydrolysis of labile organophosphates** (e.g., acetyl phosphate, carbamoyl phosphate, pyrophosphates).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`34` C0 proteins, lengths `136–584` aa, median `209.5` aa)**: Selected from 79,778 prokaryotic HADSF sequences in SFLD $\to$ 1,319 cloned targets $\to$ 217 HisTrap + Superdex 200 SEC-purified enzymes $\to$ 34 C0 clade members. `12` of the 34 proteins (`1,980` rows, `35.29%`) are in `used_in_runs`.
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

| Split | Rows | Proteins | Ligands | Mean | Std | Min / p25 | Median | p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 4,420 | 34 | 165 | `0.0000` | `1.0000` | `-1.2127` | `0.0442` | `0.7854` | `2.0734` |
| **Val** | 594 | 34 | 163 | `-0.0020` | `0.9888` | `-1.2127` | `0.0442` | `0.7745` | `2.0372` |
| **Test** | 596 | 34 | 161 | `-0.0004` | `0.9964` | `-1.2127` | `0.0442` | `0.7782` | `2.0085` |
