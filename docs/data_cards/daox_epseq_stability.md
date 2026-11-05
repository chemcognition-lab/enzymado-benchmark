# *Rhodotorula gracilis* D-Amino Acid Oxidase (RgDAOx) EP-Seq Deep Mutational Scan — Surface Display Expression / Stability (`daox_epseq_stability`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `6,434` unique missense variants (`5,150` train / `642` val / `642` test)
> **Primary Citation:** Vanella, R., Küng, C., Schoepfer, A. A., Doffini, V., Ren, J. & Nash, M. A. Understanding activity-stability tradeoffs in biocatalysts by enzyme proximity sequencing. *Nature Communications* **15**, 1807 (2024). [DOI: 10.1038/s41467-024-45630-3](https://doi.org/10.1038/s41467-024-45630-3)
> **Machine-Readable Contract:** [`daox_epseq_stability_dataset_card_enriched.json`](../../docs/data_cards/daox_epseq_stability_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/daox_epseq_stability/dataset_card.json`](../../data/processed/real_case/daox_epseq_stability/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Near-Complete Yeast Surface Display Stability/Abundance DMS:** `daox_epseq_stability` measures **6,434 single missense variants** ($93.0\%$ of all $6,916$ possible single substitutions across positions 2–365) of **D-amino acid oxidase from *Rhodotorula gracilis* (RgDAOx, 365 aa)** displayed on *Saccharomyces cerevisiae* EBY100.
- **Eukaryotic Secretory Quality Control as a Stability Proxy:** Because the yeast endoplasmic reticulum quality-control system retains and degrades misfolded or thermally unstable polypeptides, cell-surface display abundance quantified via immunofluorescence of the C-terminal `6xHis` tag (Alexa Fluor 594 across 4 FACS gates) tracks folding stability (correlating with FoldX $\Delta\Delta G$ at Spearman $\rho = -0.51$ and identifying thermally stabilized variants confirmed by purified DSF).
- **Direct Biophysical Companion to [`daox_epseq_activity`](../../docs/data_cards/daox_epseq_activity.md):** Comparing `log_stability` ($N = 6,434$) against `log_activity` ($N = 6,404$; $6,399$ shared variants) separates global Rossmann-core destabilization from specific catalytic, FAD-binding, or dimer-interface perturbations.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Low-noise pooled 4-gate FACS-seq with high biological replicate concordance | `main` p. 3 |
| **Biological Replicates** | **$n = 2$ independent biological replicates**, **Pearson $r = 0.94$** ($n = 7,069$, $p < 0.0001$), implying **$R^2_{\max} \approx 0.92\text{–}0.94$** ($r_{\max} \approx 0.96\text{–}0.97$) | `main` p. 3 |
| **Internal Controls** | Synonymous WT variants centered at **$0.00 \pm 0.057$** ($n = 301$, vs. missense $\sigma = 0.197$); nonsense variants at **$-0.52 \pm 0.11$** ($n = 334$) | `main` p. 3 |
| **Monogenic Validation** | **Pearson $r = 0.89$** ($p < 0.0001$) across 11 individually cloned and flow-cytometry-assayed mutants | `main` p. 3 |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** *Rhodotorula gracilis* (*Rhododosporidium toruloides*) D-amino acid oxidase (EC `1.4.3.3`, UniProt `P80324`, PDB `1C0P`).
- **Surface Display Construct:** `pYD1` N-terminal **Aga2p** fusion + 13-residue linker (`GTPTPTPTPTGEF`) + 365-aa RgDAOx (lacking C-terminal `SKL`) + C-terminal `6xHis` tag.

---

## 4. How the Dataset Was Built & Target Transformation

- **Upstream Definition:** Weighted average FACS gate score $\beta_v$ across 4 anti-6xHis Alexa Fluor 594 sorting bins ($\ge 100$ reads per variant), normalized to tile-internal synonymous wild-type variants $\beta_{\text{wt}}$:
  $$F_{\text{exp}} = \log_2\!\left(\frac{\beta_v}{\beta_{\text{wt}}}\right)$$
- **Enzymado Ingestion ([`rgdaox_epseq.py`](../../scripts/data_prep/real_cases/rgdaox_epseq.py)):** Extracts column `Expression score (log)` from `data/raw/real_cases/rgdaox_epseq/DMS_scores_DAOX.csv` for all `6,434` finite missense rows (excluding 334 nonsense `*` rows) as `log_stability`, and z-score standardizes using the training split ($\mu_{\text{train}} = -0.179484$, $\sigma_{\text{train}} = 0.196273$).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Expression Induction** | *S. cerevisiae* EBY100 in galactose medium + $100\text{ mM}$ citrate-phosphate buffer (**pH 7.0**), **48 h at $20\,^\circ\text{C}$** ($200\text{ rpm}$) | `main` p. 10 |
| **Antibody Staining** | Anti-6xHis mouse primary antibody ($1:500$) + Alexa Fluor 594 goat anti-mouse secondary antibody ($1:500$) in $1\times\text{PBS} + 0.1\%$ (w/v) BSA | `main` p. 10 |
| **Substrate Loading** | None (surface display abundance measured in the absence of D-alanine) | `main` p. 9–10 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`5,150` train / `642` val / `642` test).
- **Split Alignment Warning:** Because `daox_epseq_stability` has $6,434$ rows whereas `daox_epseq_activity` has $6,404$ rows, always align variants across the two tasks using `sha256(sequence)[:16]`.
