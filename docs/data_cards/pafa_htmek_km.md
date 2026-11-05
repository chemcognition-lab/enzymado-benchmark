# PafA Alkaline Phosphatase HT-MEK Valine/Glycine Scan — Michaelis Constant $K_M$ (`pafa_htmek_km`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `1,018` unique variants (`818` train / `100` val / `100` test)
> **Primary Citation:** Markin, C. J. *et al.* Revealing enzyme functional architecture via high-throughput microfluidic enzyme kinetics. *Science* **373**, eabf8761 (2021). [DOI: 10.1126/science.abf8761](https://doi.org/10.1126/science.abf8761)
> **Machine-Readable Contract:** [`pafa_htmek_km_dataset_card_enriched.json`](../../docs/data_cards/pafa_htmek_km_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/pafa_htmek_km/dataset_card.json`](../../data/processed/real_case/pafa_htmek_km/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Misfolding-Invariant, Expression-Independent $K_M$ Landscape:** Unlike $k_{\text{cat}}$ and $k_{\text{cat}}/K_M$, the Michaelis constant $K_M$ ($\log_{10} K_M$ in $\mu\text{M}$) is determined purely by the midpoint of the 7–8 point cMUP substrate saturation curve ($10\text{–}1000\,\mu\text{M}$). Consequently, **`pafa_htmek_km` is completely invariant to total enzyme concentration $\text{[E]}$ and to the kinetically trapped misfolded state (`M`)** that affects 232 positions during $37\,^\circ\text{C}$ expression.
- **Compact Bulk Distribution with Sharp Mechanistic Outliers:** Across the 1,018 variants, $K_M$ spans roughly two orders of magnitude ($5.0\,\mu\text{M}$ to $580.0\,\mu\text{M}$, median $74.0\,\mu\text{M}$, $\sigma_{\log_{10}} = 0.187$). Most distal mutations leave $K_M$ virtually unchanged (IQR $64\text{–}86\,\mu\text{M}$), whereas mutations in the active site, the T79 nucleophile helix (**FC2** ground-state electrostatic destabilization), or the K162/R164 phosphoryl-binding pocket (**FC3**) produce sharp shifts in both directions.
- **Sign Convention:** Lower $\log_{10} K_M$ (negative standardized values) indicates **tighter apparent substrate affinity** (down to the $5.0\,\mu\text{M}$ lower titration limit, `-6.09` standardized), whereas higher $\log_{10} K_M$ (positive standardized values) indicates weaker apparent affinity (up to $580\,\mu\text{M}$, `+4.96` standardized).

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Low titration error ($\sim 12.9\%$ median relative SE) over a compact biological range ($\sigma_{\log_{10}} = 0.188$) | `main` p. 7; `pafa_htmek_summary.csv` |
| **Replicate Design** | Median of **9 replicates** (Valine scan) and **7 replicates** (Glycine scan) across 16 experiments | `main` p. 7 |
| **Empirical Noise Ceiling** | Median SE = $0.0561\ \log_{10}\text{ units}$ vs. dataset $\sigma = 0.1884\ \log_{10}\text{ units}$, giving **$R^2_{\max} = 0.876$** ($r_{\max} = 0.936$) | Computed from `km_cMUP_uM_err` |
| **Censored / Limit Rows** | **8 / 1,018** flagged with `KM_cMUP_limit == 'upper'` ($K_M < [\text{cMUP}]_{\text{lowest}}/2$; 7 pinned at $5.0\,\mu\text{M}$, 1 at $8.15\,\mu\text{M}$) | `si` p. 12; `pafa_htmek_summary.csv` |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme & Construct:** Identical 1,018-variant cohort to [`pafa_htmek_kcat`](../../docs/data_cards/pafa_htmek_kcat.md): *Elizabethkingia meningoseptica* PafA (EC `3.1.3.1`, UniProt `Q935C6`, PDB `5TJ3`), 529-aa mature+linker sequence C-terminally fused to `GGGSGGGGSG-eGFP`, with preprotein mutation tokens shifted by `offset = -17`.

---

## 4. How the Dataset Was Built & Target Transformation

- **Enzymado Ingestion ([`pafa_htmek.py`](../../scripts/data_prep/real_cases/pafa_htmek.py)):** Filters `pafa_htmek_summary.csv` (`1,042` rows) for the 1,018 rows with finite `km_cMUP_uM` (excluding the same 24 non-saturating variants as `pafa_htmek_kcat`) and computes:
  $$y_{\text{raw}} = \log_{10}\!\left(K_{M,\text{ cMUP}}\ [\mu\text{M}]\right)$$
  spanning $0.6990$ ($5.0\,\mu\text{M}$) to $2.7634$ ($580.0\,\mu\text{M}$), standardized with $\mu_{\text{train}} = 1.837218$, $\sigma_{\text{train}} = 0.186899$.

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Expression & Capture** | On-chip NEB PURExpress TX-TL ($45\text{ min}$ at $37\,^\circ\text{C}$, $100\,\mu\text{M}\ \text{ZnCl}_2$), immobilized on anti-eGFP Button valves | `si` p. 7 |
| **Assay Buffer & pH** | $100\text{ mM}$ MOPS, $500\text{ mM}$ NaCl, $100\,\mu\text{M}\ \text{ZnCl}_2$, **pH 8.0** at **$23\,^\circ\text{C}$** | `main` p. 9; `si` p. 11 |
| **Substrate Titrations** | $10, 25, 50, 100, 200, 500, 1000\,\mu\text{M}$ **cMUP** (plus $2\,\mu\text{M}$ in a subset of runs) | `si` p. 11 |

---

## 6. Split Semantics & Caveats

- **Identical Split to `pafa_htmek_kcat`:** Uses the exact same 80/10/10 split (`818` train / `100` val / `100` test), enabling direct multi-task evaluation of $k_{\text{cat}}$ and $K_M$.
- **Extreme Left-Tail Leverage:** The 7 active-site variants pinned at the $5.0\,\mu\text{M}$ upper limit (`T79S`, `N100A`, `N100G`, `N100V`, `R164A`, `R164G`, `N100A/R164A`) sit at `-6.09` standard deviations below the training mean and exert high leverage on RMSE and Pearson $r$.
