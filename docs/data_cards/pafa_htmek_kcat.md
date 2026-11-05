# PafA Alkaline Phosphatase HT-MEK Valine/Glycine Scan — Turnover Number $k_{\text{cat}}$ (`pafa_htmek_kcat`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `1,018` unique variants (`818` train / `100` val / `100` test)
> **Primary Citation:** Markin, C. J. *et al.* Revealing enzyme functional architecture via high-throughput microfluidic enzyme kinetics. *Science* **373**, eabf8761 (2021). [DOI: 10.1126/science.abf8761](https://doi.org/10.1126/science.abf8761)
> **Machine-Readable Contract:** [`pafa_htmek_kcat_dataset_card_enriched.json`](../../docs/data_cards/pafa_htmek_kcat_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/pafa_htmek_kcat/dataset_card.json`](../../data/processed/real_case/pafa_htmek_kcat/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Gold-Standard Microfluidic Michaelis–Menten $k_{\text{cat}}$ Landscape:** `pafa_htmek_kcat` provides direct steady-state unimolecular turnover numbers ($\log_{10} k_{\text{cat}}$ in $\text{s}^{-1}$) for **1,018 purified variants** of *Elizabethkingia meningoseptica* (*Flavobacterium meningosepticum*) phosphate-irrepressible alkaline phosphatase **PafA** (529-aa construct) hydrolyzing the fluorogenic aryl phosphate monoester **cMUP**.
- **Expression-Normalized, Yet Modulated by Kinetic Misfolding Traps ($f_a$):** Every microfluidic chamber measures immobilized PafA-eGFP concentration ($\text{[E]}$) via an eGFP calibration curve before dividing $V_{\max}$ by $\text{[E]}$. However, Markin *et al.* (2021) discovered that **232 mutations promote kinetic partitioning during $37\,^\circ\text{C}$ cell-free expression into a long-lived, catalytically inactive misfolded state (`M`)** whose C-terminal eGFP tag still folds and fluoresces normally. Thus, apparent $k_{\text{cat}}$ reflects the product of intrinsic catalytic turnover and the natively folded fraction $f_a$.
- **Paired Multi-Parameter Architecture:** This dataset is directly paired with [`pafa_htmek_km`](../../docs/data_cards/pafa_htmek_km.md) ($N = 1,018$) and [`pafa_htmek_kcat_km`](../../docs/data_cards/pafa_htmek_kcat_km.md) ($N = 1,042$) on the exact same enzyme and substrate.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Low-noise multi-replicate microfluidic Michaelis–Menten fitting | `main` p. 7; `si` p. 12 |
| **Replicate Design** | Median of **9 replicates** (Valine scan) and **7 replicates** (Glycine scan) across 16 three-tier experiments | `main` p. 7 |
| **Empirical Noise Ceiling** | Median relative SE = **13.0%** ($0.0564\ \log_{10}\text{ units}$ vs. dataset $\sigma = 0.612$), giving **$R^2_{\max} = 0.986$** ($r_{\max} = 0.993$) | Computed from `kcat_cMUP_s_inv_err` |
| **Censored / Limit Rows** | **0 / 1,018** (`kcat_cMUP_limit == 'none'` for all 1,018 variants; 24 non-saturating `NaN` rows excluded) | `pafa_htmek_summary.csv` |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** *Elizabethkingia meningoseptica* PafA (EC `3.1.3.1`, UniProt `Q935C6`, PDB `5TJ3`), a bimetallo $\text{Zn}^{2+}$ alkaline phosphatase superfamily member with an additional distal structural $\text{Zn}^{2+}$ site.
- **Construct & Numbering Offset:** Mature PafA (527 aa, lacking the 17-aa native signal peptide) extended by C-terminal linker residues `GR` (**529 aa total**) fused to `GGGSGGGGSG-eGFP`. Preprotein residue numbers in the paper are mapped to 1-based mature sequence coordinates using `offset = -17` (e.g., active-site nucleophile `T79` $\rightarrow$ mature index `62`).
- **Library Composition:** 505 Valine-scan single mutants, 507 Glycine-scan single mutants (with Alanine substituted at native Val/Gly positions), 5 single active-site mutants (`T79S`, `N100A`, `K162A`, `R164A`, plus scanning overlaps), and 1 double active-site mutant (`N100A/R164A`).

---

## 4. How the Dataset Was Built & Target Transformation

- **Enzymado Ingestion ([`pafa_htmek.py`](../../scripts/data_prep/real_cases/pafa_htmek.py)):** Reads `data/raw/real_cases/pafa_htmek/pafa_htmek_summary.csv` (`1,042` total rows), drops the **24 severely impaired/misfolded variants** (`D38V`, `G64V`, `T79G`, `T79V`, `G82V`, `G89V`, `G96V`, `I97G`, `W102G`, `S133V`, `S160V`, `I167G`, `G171V`, `W179G`, `S190V`, `Y193G`, `D305V`, `G312V`, `D326V`, `D352V`, `H353G`, `G354V`, `S464V`, `H495G`) that lack separate $k_{\text{cat}}$ and $K_M$ fits, and computes:
  $$y_{\text{raw}} = \log_{10}\!\left(k_{\text{cat, cMUP}}\ [\text{s}^{-1}]\right)$$
  spanning $-1.7959$ ($0.016\text{ s}^{-1}$) to $+2.5051$ ($320.0\text{ s}^{-1}$), standardized with $\mu_{\text{train}} = 1.697428$, $\sigma_{\text{train}} = 0.611389$.

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Expression System** | On-chip NEB PURExpress *E. coli* cell-free TX-TL with $100\,\mu\text{M}\ \text{ZnCl}_2$ for 45 min at **$37\,^\circ\text{C}$**, plus 90 min RT eGFP maturation | `si` p. 7 |
| **Purification & Capture** | In-situ microfluidic affinity capture beneath anti-eGFP antibody-patterned Button valves ($\sim 1.5\text{ nM}$ Tier 1; $\sim 15\text{ nM}$ Tiers 2–3) | `main` p. 6–7; `si` p. 7 |
| **Assay Buffer & pH** | $100\text{ mM}$ MOPS, $500\text{ mM}$ NaCl, $100\,\mu\text{M}\ \text{ZnCl}_2$, **pH 8.0** | `si` p. 11 |
| **Assay Temperature** | **$23\,^\circ\text{C}$** | `main` p. 9 |
| **Substrate Loading** | 7–8 concentrations of **cMUP**: $10, 25, 50, 100, 200, 500, 1000\,\mu\text{M}$ (plus $2\,\mu\text{M}$ in a subset of runs) | `si` p. 11 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`818` train / `100` val / `100` test).
- **Same-Site Partner Sharing:** Because almost every residue is mutated to both Valine and Glycine, ~80% of test variants have their alternate amino acid substitution at the same sequence position present in the training set.
