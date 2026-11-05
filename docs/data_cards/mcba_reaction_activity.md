# Dataset Card: `mcba_reaction_activity` — McbA ATP-Dependent Amide Synthetase Engineering across 9 Pharmaceutical Amides

> **json card**: [`mcba_reaction_activity_dataset_card_enriched.json`](./mcba_reaction_activity_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/mcba_reaction_activity/dataset_card.json`](../../data/processed/real_case/mcba_reaction_activity/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

> **Note:** the identity table and split table on this card match the published CSV. The remaining text and the enriched JSON were written for an earlier version of this dataset and are pending review.

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `mcba_reaction_activity` |
| **Property Name** | `log10_activity` |
| **Biochemical Observable** | $\log_{10}(\text{activity} + 10^{-3})$, where `activity` is WT-normalized or DAD-normalized LC-MS amide product formation after $16\text{ h}$ at $37\,^\circ\text{C}$ across 1,932 cell-free expressed McbA variants and 9 pharmaceutical amidation reactions |
| **Primary Experimental Source** | Landwehr, G. M., Bogart, J. W., Magalhaes, C., Hammarlund, E. G., Karim, A. S. & Jewett, M. C. *Accelerated enzyme engineering by machine-learning guided cell-free expression*. *Nature Communications* **16**, 865 (2025), DOI: [`10.1038/s41467-024-55399-0`](https://doi.org/10.1038/s41467-024-55399-0) |
| **Raw Data Source** | Supplementary Data 1 (`41467_2024_55399_MOESM4_ESM.xlsx`) / [`grantlandwehr/McbA_engineering`](https://github.com/grantlandwehr/McbA_engineering) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/mcba_amide_synthetase.py`](../../scripts/data_prep/real_cases/mcba_amide_synthetase.py) |
| **Published Split** | `mcba_reaction_activity_random_80_10_10_standardized` (`8,484` train / `1,217` val / `1,217` test; `10,918` total rows) |

---

## 2. How to Think About This Dataset

1. **Active-site hot-spot saturation (`1,216` single mutants) + ML-guided multi-mutants (`716` variants) across `9` pharmaceutical amides (`12,068` rows)**: Using the crystal structure of the ATP-dependent amide synthetase **McbA** from *Marinactinospora thermotolerans* (`UniProt R4R1U5`, `PDB 6SQ8`, $495\text{ aa}$), Landwehr et al. performed full site-saturation mutagenesis across 64 residues within $\sim 10\,\text{\AA}$ of the active site and substrate tunnels ($64 \times 19 = 1,216$ single mutants) and combined beneficial substitutions via ridge regression and transfer learning to synthesize 9 diverse pharmaceutical amides (`moclobemide`, `metoclopramide`, `cinchocaine`, `itopride`, `declopramide`, `trimethobenzamide`, `sulpiride`, `procainamide`, and `troxipide`).
2. **Why there are `8` unique `ligand_smiles` but `9` unique `reaction_smiles`**: McbA catalyzes a bimolecular condensation ($\text{carboxylic acid} + \text{amine} + \text{ATP} \to \text{amide} + \text{AMP} + \text{PP}_i$). In Enzymado's schema, `ligand_smiles` stores the carboxylic acid substrate, while `reaction_smiles` stores the complete `acid.amine>>amide` reaction. Because **trimethobenzamide** and **troxipide** share the exact same carboxylic acid (`3,4,5-trimethoxybenzoic acid`, `COc1cc(C(=O)O)cc(OC)c1OC`) paired with two different amines, models that inspect only `ligand_smiles` cannot distinguish those two reactions.
3. **Unpurified cell-free protein synthesis (`CFPS`) and per-sheet normalization**: Variants were expressed in $10\,\mu\text{L}$ PANOx-SP cell-free reactions in 384-well plates from linear PCR templates (`LETs`) with an N-terminal `CSL-tag`, and $3\,\mu\text{L}$ of crude extract ($\sim 1\,\mu\text{M}$ enzyme) was transferred directly into the amidation reaction. Because values in Supplementary Data 1 are normalized per reaction (WT-normalized conversion by UV DAD at $245\text{ nm}$ or EIC normalized to the acid DAD peak), labels measure relative activity within each reaction rather than absolute turnover rates across reactions.

---

## 3. Label Noise & Performance Ceiling

- **Replicate Design (`n = 1` primary HSS screens; `n = 3` validation)**: The 1,216-member single-mutant Hot Spot Screens (`HSS`) were assayed at **`n = 1`** per reaction well (main text p. 5), whereas follow-up validation and purified enzyme assays were measured in triplicate (`n = 3`, main text p. 11).
- **Decomposed Noise & Distortion Sources**:
  1. **Unpurified CFPS expression & solubility coupling**: Fixed volume ($3\,\mu\text{L}$) of crude CFPS reaction is assayed without per-well protein concentration normalization.
  2. **Low signal-to-noise on MS-only trace reactions**: For sluggish reactions only detectable by single-quadrupole ESI-MS extracted ion chromatograms (`EIC`), the authors note that modest apparent improvements in HSS can be artifacts of low signal-to-noise (main text p. 6; SI p. 30).
  3. **Zero-conversion floor (`1,945` rows, `16.12%`)**: Mapped via $\log_{10}(0 + 10^{-3}) = -3.0$ to `-2.1190` standardized.

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`1,932` McbA variants, all `495` aa)**: All sequences are single or multi-site variants of wild-type *M. thermotolerans* McbA (`UniProt R4R1U5`, `PDB 6SQ8`). `1,216` proteins (`2,007` rows, `16.63%`) are in `used_in_runs`.
- **Reaction Scope (`9` amides / `8` carboxylic acids)**: Pharmaceutical benzamides and heteroaromatic amides covering antidepressants (`moclobemide`), antiemetics/prokinetics (`metoclopramide`, `itopride`, `trimethobenzamide`), local anesthetics (`cinchocaine`), antiarrhythmics (`procainamide`), antipsychotics (`sulpiride`), and gastroprotective agents (`troxipide`).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **CFPS Expression** | $10\,\mu\text{L}$ PANOx-SP *E. coli* cell-free reaction in 384-well plates from $1\,\mu\text{L}$ linear PCR template (with N-terminal `CSL-tag`), $30\,^\circ\text{C}$ for $16\text{ h}$ |
| **Amidation Assay** | $25\,\mu\text{L}$ reaction initiated with $3\,\mu\text{L}$ crude CFPS ($\sim 1\,\mu\text{M}$ McbA), incubated at $37\,^\circ\text{C}$ for $16\text{ h}$, quenched with $25\,\mu\text{L}$ methanol |
| **pH & Buffer** | pH $7.5$, $50\text{ mM}$ potassium phosphate buffer, $5\text{ mM MgCl}_2$, $10\%\text{ v/v}$ DMSO |
| **Substrate & ATP Loading** | $25\text{ mM}$ carboxylic acid, $25\text{ mM}$ amine ($20\text{ mM}$ each for cinchocaine; $10\text{ mM}$ each for procainamide), $25\text{ mM}$ ATP |
| **Analytical Readout** | Agilent G6125B Single Quadrupole LC/MSD (C18 column): UV DAD at $245\text{ nm}$ peak ratio $\text{amide}/(\text{amide} + \text{acid})$ or positive-mode ESI EIC divided by acid DAD peak area |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands (`acid`) | Mean | Std | Min | p25 | Median | p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,484 | 1,217 | 9 | `0.0000` | `1.0000` | `-2.0319` | `-0.3598` | `0.4596` | `0.7032` | `2.0498` |
| 1,217 | 1,217 | 9 | `0.0355` | `0.9646` | `-2.0319` | `-0.2741` | `0.4636` | `0.7035` | `1.9539` |
| 1,217 | 1,217 | 9 | `0.0257` | `0.9841` | `-2.0319` | `-0.3069` | `0.4586` | `0.7089` | `2.2881` |
