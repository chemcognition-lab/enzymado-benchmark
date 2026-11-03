# Dataset Card: `cleeveland_galox` — Fungal AA5_2 Copper Radical Oxidase Substrate Specificity Survey

> ** JSON Companion**: [`cleeveland_galox_dataset_card_enriched.json`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/cleeveland_galox_dataset_card_enriched.json)
> ** Computed Statistics**: [`data/processed/real_case/cleeveland_galox/dataset_card.json#/statistics`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/cleeveland_galox/dataset_card.json)

---

## 1. Identity & Provenance

| Field | Value |
| :--- | :--- |
| **`task_id` / `dataset_id`** | `cleeveland_galox` / `cleeveland` |
| **Property (`property_name`)** | `log10_specific_activity` ($\log_{10}(\text{specific\_activity}\;[\mu\text{mol}\cdot\text{min}^{-1}\cdot\text{mg}^{-1}])$, offset $= 0.0$) |
| **Task / Label / Layout** | `protein_ligand_global` / `scalar_regression` / **Layout B + numeric conditions** (`protein_sequence`, `smiles`, `substrate_concentration`) |
| **Enzyme Family & EC Numbers** | Auxiliary Activity Family 5 Subfamily 2 (AA5_2) fungal copper radical oxidases (CROs) — **EC `1.1.3.9`** (galactose 6-oxidase), **EC `1.1.3.13`** (alcohol oxidase), **EC `1.1.3.7`** (aryl-alcohol oxidase) |
| **Source Organisms** | 11 fungal species across Ascomycota & Basidiomycota (12 recombinant enzymes: `ExeGalOx`, `MreGalOx`, `PfeGalOx`, `FoxGalOxB`, `UmaRafOx`, `PhuRafOx`, `AflAlcOx`, `PruAlcOx` [`PruAA5_2A`], `FoxAlcOx`, `AsyAlcOx`, `PorAlcOx`, `CglAlcOx`) |
| **Primary Survey Paper (`review` / `si`)** | Cleveland, M. E. et al. (2021). "A survey of substrate specificity among Auxiliary Activity Family 5 copper radical oxidases." *Cellular and Molecular Life Sciences*, 78, 8187–8208. DOI: `10.1007/s00018-021-03981-w` ([`cleveland_galox2021_cmls.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/cleveland_galox2021_cmls.pdf), [`cleveland_galox2021_cmls_suppinfo.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/cleveland_galox2021_cmls_suppinfo.pdf) Tables S5–S10) |
| **Companion Paper (`main` / `si`)** | Cleveland, M. et al. (2021). "Two *Fusarium* copper radical oxidases with high activity on aryl alcohols." *Biotechnology for Biofuels*, 14, 138. DOI: `10.1186/s13068-021-01984-0` ([`cleveland_galox2021_biotechbiofuels.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/cleveland_galox2021_biotechbiofuels.pdf), [`cleveland_galox2021_suppinfo.pdf`](file:///home/beangoben/github/enzymado-benchmark/docs/pdfs/cleveland_galox2021_suppinfo.pdf)) |
| **Processed Dataset Size** | **385** active enzyme–substrate pairs (**12** unique proteins $\times$ **40** unique substrates; **350** rows $\le 1024$ aa; **316** train / **34** val / **35** test) |

---

## 2. How to Think About This Dataset

- **What a row represents**: Each row in [`cleeveland_galox.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/cleeveland_galox/cleeveland_galox.csv) represents one catalytically active **(AA5_2 enzyme, small-molecule substrate)** pair (`protein_sequence` and canonical isomeric `smiles`), the assay `substrate_concentration` (in $\text{mM}$: `300.0`, `30.0`, `5.0`, or `2.5`), and the base-10 log-transformed initial-rate specific activity (`log10_specific_activity`, where raw specific activity is in $\mu\text{mol}\cdot\text{min}^{-1}\cdot\text{mg}^{-1}$ of purified enzyme).
- **Incomplete $12 \times 40$ screening matrix & positive-only survivorship filtering**:
  - In Cleveland et al. (2021) *Cell. Mol. Life Sci.* (`review` Fig. 2, p. 7; `si` Tables S5–S10, pp. 9–16), 12 purified AA5_2 homologs were screened in triplicate across a panel of 40 substrates ($12 \times 40 = 480$ cells).
  - Out of those 480 cells, **385 cells** exhibited measurable specific activity ($\ge 0.01\,\mu\text{mol}\cdot\text{min}^{-1}\cdot\text{mg}^{-1}$, up to $705.0\,\mu\text{mol}\cdot\text{min}^{-1}\cdot\text{mg}^{-1}$), whereas **94 cells** fell below the limit of detection ($9 \times 10^{-4}\,\mu\text{mol}\cdot\text{min}^{-1}\cdot\text{mg}^{-1}$, marked `n.m.` in Tables S5–S10) and **1 cell** was not assayed (`AsyAlcOx` on glycolaldehyde dimer, marked `NA` in Table S9).
  - When `data/raw/cleeveland/og_dataset_clev.csv` was compiled, only the **385 measurable (`> LOD`) entries** were retained (ranging from `22` substrates for `MreGalOx` to `39` substrates for `PfeGalOx`). Thus, `cleeveland_galox` models quantitative turnover rate conditional on detectable activity rather than binary substrate acceptance.
- **Matrix completion under the point split & concentration collinearity**:
  - Under the default observation-level 80/10/10 split (`cleeveland_random_80_10_10_standardized`), all 12 enzymes and nearly all 40 substrates appear in the training set. High test accuracy therefore demonstrates **bilinear enzyme–substrate matrix completion** across seen proteins and seen ligands, not unseen-scaffold generalization.
  - Furthermore, `substrate_concentration` (`300.0`, `30.0`, `5.0`, or `2.5` mM) was assigned strictly by substrate solubility/class (`300 mM` for carbohydrates/polyols/diols/short primary alcohols; `30 mM` for benzyl alcohol and galactitol; `5 mM` for methyl glyoxal, glycolaldehyde dimer, and coniferyl alcohol; `2.5 mM` for aryl alcohols, hexanol/octanol, and furans), making it 100% collinear with `smiles`.

---

## 3. Label Noise & Performance Ceiling

- **Direct analytical noise ceiling from published triplicate standard deviations**:
  - Every cell in the 12-enzyme $\times$ 40-substrate screen was measured in **technical triplicate** ($n = 3$) in a single laboratory using the 96-well HRP/ABTS microplate assay (`review` p. 7).
  - Crucially, `data/raw/cleeveland/og_dataset_clev.csv` and `data/interim/cleeveland.csv` preserve the exact triplicate standard deviation ($\pm\text{SD}$, stored in the column named `variance`) from Supplementary Tables S5–S10!
  - Across all 385 rows, the **median coefficient of variation is $\text{CV} = 10.0\%$** (mean $\text{CV} = 19.3\%$), corresponding to a **median single-replicate $\log_{10}$ standard deviation of $0.0434\;\log_{10}\text{ units}$** ($1.105\text{-fold}$) and a **median triplicate-mean standard error ($\text{SEM}$) of $0.0251\;\log_{10}\text{ units}$** ($1.06\text{-fold}$).
  - Comparing this technical error variance against the total dataset target variance ($\text{Var}(\log_{10} y) = 1.5561$, $\text{SD} = 1.2474\;\log_{10}\text{ units}$) yields an **analytical within-lab technical noise ceiling of $R^2_{\max} = 0.987$** (single-replicate noise) to **$0.996$** (triplicate-mean SEM)!
- **Secondary noise & modeling bottlenecks**:
  - **Two-decimal rounding floor at `0.01` ($\log_{10} = -2.000$)**: Because Tables S5–S10 round specific activities to $0.01\,\mu\text{mol}\cdot\text{min}^{-1}\cdot\text{mg}^{-1}$, 54 trace-activity pairs sit at `-2.000` (`-1.6824` standardized).
  - **Enzyme-specific optimal pH (`pH 6.0–8.5`)**: Each of the 12 enzymes was screened at its own pH optimum (`review` p. 17, Table 2 p. 6); because pH is fixed per protein, sequence representations can absorb it, though pH is not passed as a separate feature column.

---

## 4. Specificity & Breadth

- **Protein diversity**: **12** fungal AA5_2 copper radical oxidases (`506` to `1,068` aa, median `672` aa; **8** MMseqs2 clusters at 50% sequence identity) selected from a 623-sequence SSN across 38 phylogenetic clades:
  - **4 Galactose 6-oxidases**: `ExeGalOx` (823 aa, 29 rows), `MreGalOx` (828 aa, 22 rows), `PfeGalOx` (667 aa, 39 rows), `FoxGalOxB` (**1,068 aa**, 35 rows — the single protein $>1024$ aa, leaving `350` rows in `used_in_runs`).
  - **2 Raffinose oxidases**: `UmaRafOx` (630 aa, 34 rows), `PhuRafOx` (629 aa, 32 rows).
  - **4 Broad-specificity primary alcohol oxidases**: `AflAlcOx` (695 aa, 32 rows), `PruAlcOx` (`PruAA5_2A`, 657 aa, 34 rows), `FoxAlcOx` (679 aa, 34 rows), `AsyAlcOx` (677 aa, 27 rows).
  - **2 Non-carbohydrate alcohol oxidases**: `PorAlcOx` (611 aa, 36 rows), `CglAlcOx` (506 aa, 31 rows).
- **Ligand & substrate diversity**: **40** unique canonical isomeric SMILES spanning 7 chemical classes: 11 carbohydrates, 4 polyols, 5 diols, 2 aldehydes (hydrated *gem*-diols), 5 primary aliphatic alcohols, 8 aryl alcohols, and 5 furans (HMF, HMFCA, DFF, FFCA, furfural).
- **Experimental condition diversity**: `substrate_concentration` (`300.0`, `30.0`, `5.0`, or `2.5` mM) is included as a numeric condition feature (`Layout B + numeric conditions`).

---

## 5. How the Dataset Was Built & Cleaned

1. **Provenance reconciliation (`10.1007/s00018-021-03981-w` vs. `10.1186/s13068-021-01984-0`)**: While [`scripts/data_prep/card_metadata.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/card_metadata.py) cites the companion *Fusarium* AAO paper by the same first and last authors (`Cleveland et al. 2021 Biotechnol. Biofuels 14:138`, `main`), the 12 proteins in `proteins.csv` and 385 specific activity rows in `og_dataset_clev.csv` are extracted directly from **Supplementary Tables S5–S10** of the 12-enzyme survey paper (`Cleveland et al. 2021 Cell. Mol. Life Sci. 78:8187–8208`, `review` / `si`).
2. **Joining & validation**: In [`scripts/data_prep/real_cases/cleveland_galox.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/cleveland_galox.py), [`build_cleeveland_rows`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/cleveland_galox.py#L64-L162) maps `protein_name` to `proteins.csv` and `substrate_name` to `substrates.csv`, parses `substrate notes` into `substrate_concentration` ($\text{mM}$), verifies zero missing or duplicate triplets, writes `data/interim/cleeveland.csv` (preserving `variance` and `units`), and transforms specific activity via $\log_{10}(\text{value})$ (`offset = 0.0`).

---

## 6. Experimental Measurement & STRENDA Summary

| Parameter | Experimental Condition (Cleveland et al. 2021 *CMLS* / *Biotechnol. Biofuels*) |
| :--- | :--- |
| **Expression & Purification** | Secreted in *Pichia pastoris* X33 (`pPICZalphaA`, C-terminal $6\times\text{His}$ tag) in BMMY media + **$0.5\text{ mM CuSO}_4$** at $16\,^\circ\text{C}$ for 3 days; purified by Ni-NTA affinity + Sephadex G-25 desalting; quantified by $A_{280}$ (ProtParam $\varepsilon_{280}$) |
| **Assay Principle** | Colorimetric **HRP / ABTS coupled initial-rate assay** in 96-well plates (BioTek Epoch reader, validated against 1 mL cuvettes on a Cary 60 UV-Vis spectrometer; $2\text{ mol ABTS}$ oxidized at $414\text{ nm}$ [$\varepsilon = 36,000\text{ M}^{-1}\text{cm}^{-1}$] per $1\text{ mol H}_2\text{O}_2$ generated) |
| **Buffer, pH & Temperature** | **$100\text{ mM}$ sodium phosphate buffer** at each enzyme's optimal pH (**pH 6.0 to 8.5**), **$0.46\text{ mM}$ ABTS**, **$21\text{ U/mL}$ HRP**, at **room temperature ($\sim 22\text{–}25\,^\circ\text{C}$)** |
| **Substrate Concentration Tiers** | **$300\text{ mM}$** (carbohydrates, polyols, diols, short primary alcohols), **$30\text{ mM}$** (benzyl alcohol, galactitol), **$5\text{ mM}$** (methyl glyoxal, glycolaldehyde dimer, coniferyl alcohol), **$2.5\text{ mM}$** (aryl alcohols, hexanol/octanol, furans); initiated with $30\text{ pmol–}0.26\text{ nmol}$ purified enzyme |

---

## 7. Split Semantics, Leakage & Caveats

- **Standardized split**: [`cleeveland_random_80_10_10_standardized.csv`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/cleeveland_galox/cleeveland_random_80_10_10_standardized.csv) partitions the `385` active pairs into **316 train / 34 val / 35 test** via a deterministic (seed 42) 10-decile stratified 80/10/10 random split at the observation (`point`) level, standardizing both `substrate_concentration` and `log10_specific_activity` on the training split ($\mu_{\text{train}} = 0.0978$, $\sigma_{\text{train}} = 1.2464$).
- **Caveats & leakage analysis**:
  - **Seen-protein & seen-ligand matrix completion**: All 12 enzymes and nearly all 40 substrates appear in the training split; do not claim zero-shot protein or ligand generalization from the default split.
  - **`FoxGalOxB` length $>1024$ aa**: `FoxGalOxB` (`1,068` aa, 35 rows) exceeds 1,024 residues (`used_in_runs = 350` rows across 11 enzymes).
