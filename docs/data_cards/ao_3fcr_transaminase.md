# Dataset Card: `ao_3fcr_transaminase` — Fold Type I Amine Transaminases (`3FCR` & `3HMU`) across 15 Enantiomeric Amine Pairs

> **json card**: [`ao_3fcr_transaminase_dataset_card_enriched.json`](./ao_3fcr_transaminase_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/ao_3fcr_transaminase/dataset_card.json`](../../data/processed/real_case/ao_3fcr_transaminase/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `ao_3fcr_transaminase` |
| **Property Name** | `ln_specific_activity` |
| **Biochemical Observable** | $\ln(\text{specific activity in mU mg}^{-1})$, measured via continuous UV ketone-formation time-courses ($245\text{–}310\text{ nm}$) in triplicate (`n = 3`) on IMAC-purified, PD-10-desalted enzymes across 76 engineered Fold Type I amine transaminase variants (`64` `3FCR` + `12` `3HMU`) and 30 enantiopure $(R)$- and $(S)$-primary aromatic amines (15 racemic pairs) |
| **Primary Experimental Source** | Ao, Y.-F., Pei, S., Xiang, C., Menke, M. J., Shen, L., Sun, C., Dörr, M., Born, S., Höhne, M. & Bornscheuer, U. T. *Structure- and Data-Driven Protein Engineering of Transaminases for Improving Activity and Stereoselectivity*. *Angew. Chem. Int. Ed.* **62**, e202301660 (2023), DOI: [`10.1002/anie.202301660`](https://doi.org/10.1002/anie.202301660) |
| **Raw Data Source** | Supporting Information Tables S2 & S4 (`data/raw/real_cases/ata_3fcr/ata_3fcr_data.xlsx`, sheets `Dataset of 3FCR variants` and `Dataset of 3HMU variants`) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/ata_3fcr.py`](../../scripts/data_prep/real_cases/ata_3fcr.py) |
| **Published Split** | `ao_3fcr_transaminase_random_80_10_10_standardized` (`1,520` train / `180` val / `180` test; `1,880` total rows) |

---

## 2. How to Think About This Dataset

1. **Gold-standard purified specific-activity kinetics (`mU/mg`, `n = 3`) with zero censoring absent**: Every entry in Ao et al. (2023) is the natural logarithm of the mean specific activity ($\text{mU mg}^{-1}$ of IMAC-purified, PD-10-desalted holoenzyme) measured by continuous UV spectrophotometric monitoring ($245\text{–}310\text{ nm}$) across three independent experiments (`n = 3`). Specific activities span nearly five orders of magnitude ($0.20\text{ mU mg}^{-1}$ to $\sim 14,000\text{ mU mg}^{-1}$, or $\ln \in [-1.61, +9.55]$) without an artificial zero spike.
2. **Strict test of stereochemical representation (`15` $(R)/(S)$ enantiomeric pairs = `30` chiral amines)**: The 30 substrates consist of 15 pairs of pure $(S)$- and $(R)$-primary aromatic/cyclic amines (`1–15`). Wild-type `3FCR` and `3HMU` are $(S)$-selective, whereas engineered pocket variants invert stereoselectivity to $(R)$-selective (`>99% ee`). **Chirality-blind 2D molecular descriptors assign identical vectors to $(R)$- and $(S)$-enantiomers** and therefore cannot capture enantioselectivity inversions.
3. **Note the CIP nomenclature inversion on substrate pair `3` (`4-chromanamine`)**: As highlighted in Fig. 3 of Ao et al. (main text p. 4), **`(R)-3` (`(R)-4-chromanamine`) has the exact same active-site binding mode as all other `(S)`-amines** (`(S)-1`, `(S)-2`, `(S)-4`–`15`), because the ring oxygen atom changes the Cahn–Ingold–Prelog priority order of the substituents around the stereocenter.
4. **Two homologous scaffolds (`3FCR` vs. `3HMU`) sharing 7 targeted active-site residues**: Variants mutate 4 small-pocket residues (`Y87`, `S19`, `Y152`, `S155` in `3FCR`) and 3 large-pocket residues (`T231`, `Y59`, `L58` in `3FCR`; corresponding to `91, 25, 156, 159, 233, 63, 62` in `3HMU`).

---

## 3. Label Noise & Performance Ceiling

- **Triplicate Purified Kinetic Precision (`n = 3`)**: All 1,880 datapoints are means of three independent experiments on purified, PLP-reconstituted enzymes (main text p. 6; SI p. 4), supporting out-of-sample $R^2$ up to $0.88$ in Ao et al. (2023).
- **Decomposed Noise & Distortion Sources**:
  1. **Minor rounding at the lower detection limit ($0.2\text{–}0.5\text{ mU mg}^{-1}$)** for poorly accepted enantiomers ($\ln(0.2) = -1.6094$, standardized `-2.6470`).
  2. **Non-additive scaffold epistatic effects between `3FCR` (`467` aa) and `3HMU` (`469` aa)**: Identical 7-residue active-site pocket motifs exhibit distinct catalytic baselines in `3HMU` vs. `3FCR` due to second-shell and PLP–protein interactions (main text p. 6–7).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`76` variants: `64` of `3FCR` [`467` aa, `1,520` rows] + `12` of `3HMU` [`469` aa, `360` rows])**: Parent enzymes from *Ruegeria* sp. TM1040 (`PDB 3FCR`, `UniProt Q1GD43`) and *Ruegeria pomeroyi* DSS-3 (`PDB 3HMU`, `UniProt Q5LMU1`), carrying C-terminal His-tags (`EGSHHHHHH`). `15` of the 76 proteins (`435` rows, `23.14%`) are in `used_in_runs`.
- **Chemical Scope (`30` enantiopure primary amines = `15` $(R)/(S)$ pairs)**: Cyclic arylalkylamines (`1-aminotetralin`, `1-aminoindane`, `4-chromanamine`, `4-thiochromanamine`), alkyl-chain homologues (`1-phenylethylamine`, `1-phenylpropylamine`, `1-phenylbutylamine`), and ring-substituted `1-phenylethylamines` (`p-F`, `p-Cl`, `p-Br`, `p-Me`, `p-OMe`, `p-NO2`, `m-OMe`, `p-CF3`).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Expression & Purification** | *E. coli* BL21(DE3) / pGASTON induced with $0.2\%$ L-rhamnose ($20\,^\circ\text{C}$, $20\text{ h}$), purified by $\text{Ni}^{2+}$-NTA chromatography and desalted on PD-10 columns in $50\text{ mM}$ HEPES pH $7.5 + 0.1\text{ mM}$ PLP |
| **Assay Format & Readout** | Continuous UV ketone absorbance time-course ($245\text{–}310\text{ nm}$, TECAN Infinite 200 PRO in UV-Star 96-well plates) in triplicate (`n = 3`); $1\text{ U} = 1\,\mu\text{mol}$ ketone $\text{min}^{-1}$ |
| **Temperature** | $30\,^\circ\text{C}$ |
| **pH & Buffer** | pH $9.0$, $50\text{ mM}$ CHES buffer, $1.25\text{–}2.5\%$ DMSO |
| **Substrate & Cofactor Loading** | $2.5\text{ mM}$ enantiopure $(R)$- or $(S)$-amine donor, $2.5\text{ mM}$ pyruvate amino acceptor, PLP holoenzyme |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min | p25 | Median | p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 1,520 | 76 | 30 | `0.0000` | `1.0000` | `-2.6470` | `-0.6836` | `0.0752` | `0.8315` | `2.2389` |
| **Val** | 180 | 72 | 30 | `-0.0005` | `0.9854` | `-2.6470` | `-0.6746` | `0.0736` | `0.8065` | `2.1640` |
| **Test** | 180 | 70 | 30 | `-0.0009` | `1.0080` | `-2.6470` | `-0.6881` | `0.0886` | `0.8351` | `2.2016` |
