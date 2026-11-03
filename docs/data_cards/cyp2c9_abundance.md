# Human Cytochrome P450 2C9 (CYP2C9) VAMP-seq Deep Mutational Scan — Steady-State Cellular Abundance (`cyp2c9_abundance`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `6,371` unique variants (`5,091` train / `640` val / `640` test)
> **Primary Citation:** Amorosi, C. J. *et al.* Massively parallel characterization of CYP2C9 variant enzyme activity and abundance. *The American Journal of Human Genetics* **108**, 1735–1751 (2021). [DOI: 10.1016/j.ajhg.2021.07.001](https://doi.org/10.1016/j.ajhg.2021.07.001)
> **Machine-Readable Contract:** [`cyp2c9_abundance_dataset_card_enriched.json`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/cyp2c9_abundance_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/cyp2c9_abundance/dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/cyp2c9_abundance/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Mammalian Single-Copy VAMP-seq Abundance Landscape:** `cyp2c9_abundance` measures **6,370 single missense variants plus 1 wild-type sequence (`6,371` total)** across positions 2–490 of **human Cytochrome P450 2C9 (CYP2C9, 490 aa)** in cultured human `HEK293T` cells using Variant Abundance by Massively Parallel Sequencing (**VAMP-seq**).
- **Ratiometric ERAD Quality-Control Readout:** Each variant is integrated as a single copy into a Bxb1 recombinase landing pad as a C-terminal **CYP2C9–eGFP** fusion co-expressing **mCherry** via an IRES. Thermodynamically destabilized or misfolded variants (such as clinical allele `CYP2C9*11` [`p.Arg335Trp`]) are degraded by mammalian ER-associated degradation (ERAD) and the proteasome, lowering the cellular `eGFP:mCherry` ratio across 4 quartile FACS bins.
- **Mechanistic Counterpart to [`cyp2c9_activity`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/cyp2c9_activity.md):** Whereas $64.9\%$ of missense variants reduce Click-seq activity, only **$36.8\%$ (2,347 variants) significantly reduce VAMP-seq abundance**. Active-site and BC-loop residues remain highly abundant when mutated (median abundance scores `0.91` and `0.88`), allowing models to separate global folding stability from heme/substrate catalysis.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Moderate-to-low noise 3-replicate barcoded VAMP-seq (`5.9` mean barcodes/variant) | `main` p. 5, 8 |
| **Biological Replicates** | **$n = 3$ replicate sorts** from 2 transfections (89.2% of variants in all 3 sorts; pairwise **mean Pearson $r = 0.789$**, Spearman $\rho = 0.754$) | `main` p. 8 |
| **Empirical Noise Ceiling** | Median SE = `0.0701` (median SD = `0.1192`) vs. dataset $\sigma = 0.3290$, giving **$R^2_{\max} = 0.9017$** ($r_{\max} = 0.9496$) | Computed from MaveDB `se` column |
| **Monogenic Validation** | Individual variant HEK293T `eGFP:mCherry` flow cytometry **Pearson $r = 0.942$** ($n = 12$ variants) | `main` p. 8 |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** *Homo sapiens* Cytochrome P450 2C9 (EC `1.14.14.1` / `1.14.14.51`, UniProt `P11712`, PDB `1R9O`, `1OG5`).
- **Construct:** Full-length 490-aa human CYP2C9 fused C-terminally to **eGFP** (preserving the N-terminal ER membrane anchor) with bicistronic **IRES–mCherry**, integrated into the single-copy Bxb1 landing pad of `HEK293T-LLP-iCasp9` cells.

---

## 4. How the Dataset Was Built & Target Transformation

- **Upstream Definition:** Weighted average across 4 equal quartile FACS bins ($w_1 = 0.25$, $w_2 = 0.50$, $w_3 = 0.75$, $w_4 = 1.0$), linearly normalized per replicate so that the median synonymous variant equals `1.0` and the median nonsense variant (middle 80% of positions, excluding the first and last 10%) equals `0.0`.
- **Enzymado Ingestion ([`cyp2c9_dms.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/cyp2c9_dms.py)):** Reads `data/raw/real_cases/cyp2c9_dms/mavedb_00000095_b_1_scores.csv` (`6,821` rows), maps `p.=` to the 490-aa wild-type sequence (`score = 1.0`), drops `189` nonsense (`Ter`) and `261` single-site synonymous (`=`) rows, retains all `6,370` missense rows (`6,371` total), and z-score standardizes using the training split ($\mu_{\text{train}} = 0.686626$, $\sigma_{\text{train}} = 0.327677$).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Host & Culture** | Human `HEK293T-LLP-iCasp9` cells in DMEM + $10\%$ FBS at **$37\,^\circ\text{C}$, $5\%\ \text{CO}_2$** (**pH 7.4**) | `main` p. 4, 8; `review` p. 11 |
| **Induction & Selection** | **$2.5\,\mu\text{g/mL}$ doxycycline** (48 h post-transfection) followed 24 h later by **$10\text{ nM}$ AP1903** selection | `main` p. 4, 8 |
| **Readout** | Quartile FACS sorting of live cells by **`eGFP:mCherry` fluorescence ratio** (no catalytic substrate added) | `main` p. 8 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`5,091` train / `640` val / `640` test).
- **Note on `used_in_runs` Metadata:** As with `cyp2c9_activity`, `used_in_runs.rows` in `dataset_card.json` was recorded as `1` (only WT) when the initial summary script intersected against an unpopulated `common_ids` cache; the split parquet files contain all **6,371 sequences**.
