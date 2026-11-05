# PafA Alkaline Phosphatase HT-MEK Valine/Glycine Scan — Catalytic Efficiency $k_{\text{cat}}/K_M$ (`pafa_htmek_kcat_km`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `1,042` unique variants (`832` train / `104` val / `106` test)
> **Primary Citation:** Markin, C. J. *et al.* Revealing enzyme functional architecture via high-throughput microfluidic enzyme kinetics. *Science* **373**, eabf8761 (2021). [DOI: 10.1126/science.abf8761](https://doi.org/10.1126/science.abf8761)
> **Machine-Readable Contract:** [`pafa_htmek_kcat_km_dataset_card_enriched.json`](../../docs/data_cards/pafa_htmek_kcat_km_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/pafa_htmek_kcat_km/dataset_card.json`](../../data/processed/real_case/pafa_htmek_kcat_km/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Complete, Uncensored 1,042-Variant HT-MEK Catalytic Efficiency Landscape:** `pafa_htmek_kcat_km` measures the apparent second-order rate constant ($\log_{10}(k_{\text{cat}}/K_M)$ in $\text{M}^{-1}\text{s}^{-1}$) for **all 1,042 variants** of *Elizabethkingia meningoseptica* PafA hydrolyzing the fluorogenic aryl phosphate monoester **cMUP**. Unlike [`pafa_htmek_kcat`](../../docs/data_cards/pafa_htmek_kcat.md) and [`pafa_htmek_km`](../../docs/data_cards/pafa_htmek_km.md) ($N = 1,018$), which drop 24 non-saturating loss-of-function variants, `pafa_htmek_kcat_km` retains all 24 severely compromised variants via their lower-limit-of-detection (LLoD) upper bounds.
- **Composite Readout of Catalysis and Kinetic Folding Partitioning ($f_a$):** Although immobilized enzyme concentration $\text{[E]}$ is calibrated and divided out in every chamber via C-terminal eGFP fluorescence, mutations at **232 positions promote kinetic partitioning during $37\,^\circ\text{C}$ cell-free expression into a long-lived, catalytically inactive misfolded state (`M`)** whose eGFP tag remains fluorescent. Apparent $k_{\text{cat}}/K_M$ therefore equals $(k_{\text{cat}}/K_M)_{\text{chem}} \times f_a$.
- **Partially Binding-Limited Near Wild Type:** Because cMUP is a fast, activated aryl monoester, substrate binding is partially rate-limiting for wild-type PafA ($\sim 1.4 \times 10^6\text{ M}^{-1}\text{s}^{-1}$); small destabilizations of the chemical transition state are buffered on cMUP until the chemical barrier exceeds the binding barrier.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Exceptionally high-precision multi-replicate microfluidic Michaelis–Menten kinetics | `main` p. 7; `si` p. 12–14 |
| **Replicate Design** | Median of **9 replicates** (Valine scan) and **7 replicates** (Glycine scan) across 16 three-tier experiments | `main` p. 7 |
| **Empirical Noise Ceiling** | Median relative SE = **12.7%** ($0.0552\ \log_{10}\text{ units}$ vs. dataset $\sigma = 0.6801\ \log_{10}\text{ units}$), giving **$R^2_{\max} = 0.990$** ($r_{\max} = 0.995$) | Computed from `kcat_over_km_M_s_err` |
| **Limit Annotations** | `1,010` exact (`none`), **`24` upper limits** (`upper`, LLoD-bounded dead/misfolded variants), and **`8` lower limits** (`lower`, tight-$K_M$ variants) | `pafa_htmek_summary.csv` |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme & Construct:** *Elizabethkingia meningoseptica* PafA (EC `3.1.3.1`, UniProt `Q935C6`, PDB `5TJ3`), 529-aa mature+linker sequence C-terminally fused to `GGGSGGGGSG-eGFP`, with preprotein mutation tokens shifted by `offset = -17`.
- **Complete 1,042-Variant Library:** 519 Valine-scan single mutants, 517 Glycine-scan single mutants (with Alanine at native Val/Gly positions), 5 single active-site controls, and 1 double active-site mutant (`N100A/R164A`).

---

## 4. How the Dataset Was Built & Target Transformation

- **Enzymado Ingestion ([`pafa_htmek.py`](../../scripts/data_prep/real_cases/pafa_htmek.py)):** Reads all 1,042 rows of `data/raw/real_cases/pafa_htmek/pafa_htmek_summary.csv` and computes:
  $$y_{\text{raw}} = \log_{10}\!\left(\frac{k_{\text{cat}}}{K_M}\ [\text{M}^{-1}\text{s}^{-1}]\right)$$
  spanning $2.3222$ ($210\text{ M}^{-1}\text{s}^{-1}$) to $6.7243$ ($5.3 \times 10^6\text{ M}^{-1}\text{s}^{-1}$), standardized with $\mu_{\text{train}} = 5.798685$, $\sigma_{\text{train}} = 0.685639$.

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Expression & Capture** | On-chip NEB PURExpress TX-TL ($45\text{ min}$ at $37\,^\circ\text{C}$, $100\,\mu\text{M}\ \text{ZnCl}_2$), immobilized on anti-eGFP Button valves ($\sim 1.5\text{ nM}$ Tier 1; $\sim 15\text{ nM}$ Tiers 2–3) | `si` p. 7 |
| **Assay Buffer & pH** | $100\text{ mM}$ MOPS, $500\text{ mM}$ NaCl, $100\,\mu\text{M}\ \text{ZnCl}_2$, **pH 8.0** at **$23\,^\circ\text{C}$** | `main` p. 9; `si` p. 11 |
| **Substrate Titrations** | $10, 25, 50, 100, 200, 500, 1000\,\mu\text{M}$ **cMUP** (plus $2\,\mu\text{M}$ in a subset of runs) | `si` p. 11 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`832` train / `104` val / `106` test).
- **Split Alignment Warning:** Because `pafa_htmek_kcat_km` has $1,042$ rows whereas `pafa_htmek_kcat` and `pafa_htmek_km` have $1,018$ rows, always align predictions across the three PafA tasks by sequence hash ID (`sha256(sequence)[:16]`) rather than row index.
