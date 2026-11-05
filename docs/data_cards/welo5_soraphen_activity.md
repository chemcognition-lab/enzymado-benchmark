# WelO5* Aliphatic Halogenase Combinatorial Library Activity on Soraphen A (`welo5_soraphen_activity`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `504` unique variants (`404` train / `50` val / `50` test)
> **Primary Citation:** Büchler, J. *et al.* Algorithm-aided engineering of aliphatic halogenase WelO5* for the asymmetric late-stage functionalization of soraphens. *Nature Communications* **13**, 371 (2022). [DOI: 10.1038/s41467-022-27999-1](https://doi.org/10.1038/s41467-022-27999-1)
> **Machine-Readable Contract:** [`welo5_soraphen_activity_dataset_card_enriched.json`](../../docs/data_cards/welo5_soraphen_activity_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/welo5_soraphen_activity/dataset_card.json`](../../data/processed/real_case/welo5_soraphen_activity/dataset_card.json)

---

## 1. How to Think About This Dataset

- **3-Site Active-Site Combinatorial Landscape ($20^3 = 8,000$ Design Space):** `welo5_soraphen_activity` measures **504 Sanger-sequenced variants** (1 wild-type `VIA`, 8 single mutants, 71 double mutants, and 424 triple mutants) of the $\text{Fe(II)}/\alpha\text{-ketoglutarate}$-dependent aliphatic halogenase **WelO5*** from *Hapalosiphon welwitschii* IC-52-3 (290 aa). Mutations are restricted strictly to three first-shell active-site positions (**V81, A88, I161**) identified by docking the bulky macrolide fungicide **Soraphen A** ($520.6\text{ Da}$) into the WelO5* active site.
- **Heavily Zero-Inflated Hurdle Distribution (70.0% Inactive):** Installing a late-stage chlorine onto an unactivated $sp^3$ carbon of a non-native macrolide is sterically demanding. Out of 504 variants, **353 variants (70.0%) show zero detectable chlorination** (`activity = 0.0`, mapped to $\log_{10}(0 + 10^{-3}) = -3.0$, or `-0.6313` standardized), while the **151 active variants (30.0%)** span nearly three orders of magnitude from `0.0236` to `10.1568`-fold relative to the reference variant **WelO5* GAP** (`V81G/A88A/I161P`).
- **Coupled Lysate Expression & Catalytic Conversion:** Screening was conducted in chemically lysed *E. coli* BL21(DE3) cells in 96-well plates without normalizing for soluble holoenzyme concentration. The activity readout therefore reflects both active-site steric accommodation of Soraphen A and soluble expression/folding of tagless WelO5*.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Moderate single-replicate 96-well lysate screen with a hard 70% zero floor | `main` p. 5, 8 |
| **Biological Replicates** | $n = 1$ well per library variant, normalized per plate to parent control **WelO5* GAP** (hit variants validated at $n = 3$) | `main` p. 5, 8 |
| **Published ML Anchor** | **5-fold CV out-of-fold $R^2 = 0.745$** using Gaussian Process regression with 7 physicochemical descriptors per residue | `main` p. 9 |
| **Zero-Floor Fraction** | **353 / 504 (70.0%)** at raw `activity = 0.0` (`log10_activity = -3.0`, standardized `-0.6313`) | Computed from raw `WelO5_data.xlsx` |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** *Hapalosiphon welwitschii* IC-52-3 aliphatic halogenase WelO5* / AmbO5 (EC `1.14.11.-`, GenBank `KY499613`, PDB `7Q5F` [WT], `7Q5G` [GAP], `7Q6B` [SLP], `7Q6C` [WVS]).
- **Construct:** Full-length 290-aa polypeptide subcloned **without a His-tag** into a modified `pET28b(+)` vector.
- **Mutational Breadth:** All 504 sequences have length 290 and differ only at positions **81**, **88**, and **161** ($\ge 98.97\%$ pairwise sequence identity).

---

## 4. How the Dataset Was Built & Target Transformation

- **Upstream Definition:** Total chlorination conversion is quantified by UPLC-MS in Single Ion Monitoring (SIM) mode from the peak areas of monochlorinated isomers **1a** and **1b** ($m/z\ 523.2$), hydroxylated byproduct **1c** ($m/z\ 505.2$), and unreacted Soraphen A **1** ($m/z\ 489.3$):
  $$\text{tot. Cl conversion} = \frac{\text{SIM}_{1a} + \text{SIM}_{1b}}{\text{SIM}_{1a} + \text{SIM}_{1b} + \text{SIM}_{1c} + \text{SIM}_{1}}, \qquad A = \frac{\text{tot. Cl conversion}_{\text{mutant}}}{\text{tot. Cl conversion}_{\text{GAP}}}$$
- **Enzymado Ingestion ([`welo5_soraphen.py`](../../scripts/data_prep/real_cases/welo5_soraphen.py)):** Reads all 504 rows from `data/raw/real_cases/welo5_soraphen/WelO5_data.xlsx` (`Data` sheet), substitutes positions 81, 88, and 161 on the 290-aa wild-type sequence (`welo5_wt.fasta`), and computes:
  $$y_{\text{raw}} = \log_{10}(A + 10^{-3})$$
  followed by z-score standardization using the training split mean ($\mu_{\text{train}} = -2.265699$) and standard deviation ($\sigma_{\text{train}} = 1.163178$).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Expression Host & Media** | *E. coli* BL21(DE3) in $0.9\text{ mL}$ Zymo5052 autoinduction medium ($50\,\mu\text{g/mL}$ kanamycin), 24 h at $20\,^\circ\text{C}$, $300\text{ rpm}$ | `main` p. 8 |
| **Cell Lysis** | Chemical lysis in $100\,\mu\text{L}$ of $50\text{ mM}$ sodium phosphate (pH 8.0) with $1\text{ mg/mL}$ lysozyme, $0.5\text{ mg/mL}$ polymyxin B, $0.01\text{ mg/mL}$ DNase | `main` p. 8 |
| **Reaction Buffer & pH** | $50\text{ mM}$ sodium phosphate buffer, **pH 8.0** ($100\,\mu\text{L}$ lysate + $100\,\mu\text{L}$ $2\times$ reaction buffer = $200\,\mu\text{L}$ total) | `main` p. 8 |
| **Temperature & Time** | **$20\,^\circ\text{C}$**, overnight shaking ($850\text{ rpm}$) under breathable membrane, quenched with $800\,\mu\text{L}$ $\text{MeOH/H}_2\text{O}$ (5:3) | `main` p. 8 |
| **Substrate & Cofactors (Final)** | $1\text{ mM}$ Soraphen A, $110\text{ mM}$ $\alpha$-ketoglutarate, $106\text{ mM}$ sodium ascorbate, $500\text{ mM}$ NaCl, $1.3\text{ mM}$ $\text{Fe(NH}_4)_2(\text{SO}_4)_2$ | `main` p. 8 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`404` train / `50` val / `50` test; `125` non-zero in train, `12` in val, `14` in test).
- **Combinatorial Interpolation:** Because all 504 sequences vary only at positions 81, 88, and 161, the split tests **multi-mutant combinatorial interpolation** across the three active-site residues rather than generalization to unseen sequence positions.
