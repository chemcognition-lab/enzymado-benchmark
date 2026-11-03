# Dataset Card: `huang_had_phosphatase_c1` — HAD Superfamily Phosphatase Substrate Profiling (Cap Type C1 / `C1.5*` Clade)

> **json card**: [`huang_had_phosphatase_c1_dataset_card_enriched.json`](./huang_had_phosphatase_c1_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/huang_had_phosphatase_c1/dataset_card.json`](../../data/processed/real_case/huang_had_phosphatase_c1/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `huang_had_phosphatase_c1` |
| **Property Name** | `log10_conversion` |
| **Biochemical Observable** | $\log_{10}(\text{OD}_{650} + 10^{-3})$ from 30-min endpoint malachite-green (`BioMol Green`) inorganic phosphate release across 96 $\alpha$-helical cap (`C1.5*`) HADSF phosphatases and 165 organophosphate substrates |
| **Primary Experimental Source** | Huang, H. et al. *Panoramic view of a superfamily of phosphatases through substrate profiling*. *PNAS* **112**(16), E1974–E1983 (2015), DOI: [`10.1073/pnas.1423570112`](https://doi.org/10.1073/pnas.1423570112) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/huang_phosphatase.py`](../../scripts/data_prep/real_cases/huang_phosphatase.py) |
| **Published Split** | `huang_had_phosphatase_c1_random_80_10_10_standardized` (`12,480` train / `1,696` val / `1,664` test; `15,840` total rows) |

---

## 2. How to Think About This Dataset

1. **Why `huang_had_phosphatase_c1` selects SFLD Subgroup `C1.5*` (`96` enzymes, `15,840` rows)**: In Huang et al. (2015), Cap Type **C1** is the largest structural clade (`130` enzymes in Table S2). Because $130 \times 165 = 21,450$ rows would exceed Enzymado's 20,000-row single-task ceiling, [`huang_phosphatase.py`](../../scripts/data_prep/real_cases/huang_phosphatase.py) selects the dominant SFLD subgroup **`C1.5*`** (`96` enzymes $\times$ `165` substrates = `15,840` rows) from `Dataset S1` (`Supplemental 2` sheet), paired with [`huang_had_phosphatase_c0`](./huang_had_phosphatase_c0.md) (`34` enzymes, `5,610` rows) and [`huang_had_phosphatase_c2`](./huang_had_phosphatase_c2.md) (`48` enzymes, `7,920` rows).
2. **Structural biology of the C1 cap domain**: C1 HADSF members carry an $\alpha$-helical four-helix-bundle cap domain inserted in the middle of the flap motif after $\beta$-strand 1 of the Rossmann-like core (archetype $\beta$-phosphoglucomutase from *Lactococcus lactis*, PDB `1O08`, SI p. 51). Hinged domain closure over the active site enables C1 enzymes to desolvate and recognize diverse small-molecule phosphates (mean $24.88$ hits per enzyme in Table S2, intermediate between C0 [$13.27$] and C2 [$39.07$]; SI p. 50).
3. **Narrow kinetic differentiation window ($\text{OD}_{650} \in [0.2, 0.7]$)**: The raw target (`conversion` in Goldman's CSV) is the background-subtracted $650\text{ nm}$ absorbance from the 30-min BioMol Green assay, which detects weak activities down to $k_{\text{cat}} \ge 0.004\text{ s}^{-1}$ ($\text{OD}_{650} > 0.2 \pm 0.03$) and **saturates above $\text{OD}_{650} = 0.7$ ($k_{\text{cat}} \ge 0.014\text{ s}^{-1}$)**. Sub-threshold readings ($0 < \text{OD}_{650} < 0.2$) are retained continuously alongside **5,046 exact-zero cells (`31.86%`, `-1.2975` standardized)**.

---

## 3. Label Noise & Performance Ceiling

- **Duplicate Screening Precision (`n = 2`)**: Every enzyme–substrate pair and no-enzyme background control was assayed in **duplicate** in 384-well plates. Across the entire superfamily screen, the mean difference between duplicates was **$0.0307 \pm 0.0003\text{ OD}$** (SI p. 2).
- **Decomposed Noise & Distortion Sources**:
  1. **Malachite-green saturation above $\text{OD}_{650} = 0.7$ ($k_{\text{cat}} \ge 0.014\text{ s}^{-1}$)**: Compresses moderate and fast turnover onto the same upper plateau (main text p. E1976).
  2. **Log-stretching of sub-threshold baseline noise ($\text{OD}_{650} < 0.2$)**: $\log_{10}(\text{OD}_{650} + 10^{-3})$ stretches $\pm 0.03\text{ OD}$ baseline jitter across $-3.0$ to $-1.3$ log units.
  3. **Spontaneous hydrolysis of labile organophosphates** (e.g., acetyl phosphate, carbamoyl phosphate, pyrophosphates).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`96` `C1.5*` proteins, lengths `190–382` aa, median `230.0` aa)**: Selected from 79,778 prokaryotic HADSF sequences in SFLD $\to$ 1,319 cloned targets $\to$ 217 HisTrap + Superdex 200 SEC-purified enzymes $\to$ 96 `C1.5*` clade members. `25` of the 96 proteins (`4,125` rows, `26.04%`) are in `used_in_runs`.
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
| **Train** | 12,480 | 96 | 165 | `0.0000` | `1.0000` | `-1.2975` | `0.0715` | `0.7850` | `2.0757` |
| **Val** | 1,696 | 96 | 165 | `-0.0020` | `1.0028` | `-1.2975` | `0.0715` | `0.7798` | `2.0258` |
| **Test** | 1,664 | 96 | 165 | `0.0029` | `1.0000` | `-1.2975` | `0.0715` | `0.7889` | `2.0672` |
