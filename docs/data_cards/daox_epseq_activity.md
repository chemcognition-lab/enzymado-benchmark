# *Rhodotorula gracilis* D-Amino Acid Oxidase (RgDAOx) EP-Seq Deep Mutational Scan — Catalytic Activity (`daox_epseq_activity`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `6,404` unique missense variants (`5,124` train / `640` val / `640` test)
> **Primary Citation:** Vanella, R., Küng, C., Schoepfer, A. A., Doffini, V., Ren, J. & Nash, M. A. Understanding activity-stability tradeoffs in biocatalysts by enzyme proximity sequencing. *Nature Communications* **15**, 1807 (2024). [DOI: 10.1038/s41467-024-45630-3](https://doi.org/10.1038/s41467-024-45630-3)
> **Machine-Readable Contract:** [`daox_epseq_activity_dataset_card_enriched.json`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/daox_epseq_activity_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/daox_epseq_activity/dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/daox_epseq_activity/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Near-Complete Flavoenzyme DMS via Enzyme Proximity Sequencing (EP-Seq):** `daox_epseq_activity` measures **6,404 single missense variants** ($92.5\%$ of all $6,916$ possible single amino acid substitutions across positions 2–365) of the homodimeric, FAD-dependent peroxisomal flavoenzyme **D-amino acid oxidase from *Rhodotorula gracilis* (RgDAOx, 365 aa)** displayed on *Saccharomyces cerevisiae*.
- **Peroxide-Driven Tyramide Proximity Labeling:** Upon incubating yeast-displayed RgDAOx with $130\text{ mM}$ **D-alanine**, $\text{H}_2\text{O}_2$ released by active enzyme variants fuels exogenous horseradish peroxidase (HRP) to convert **Alexa Fluor 488 tyramide** into short-lived phenoxyl radicals that covalently tag the surface of the producing yeast cell in viscous $0.75\%$ sodium alginate medium, followed by 4-gate FACS sorting and NovaSeq deep sequencing.
- **Coupled Display & Catalysis Readout:** Because the activity sort is performed on the full yeast library without pre-gating on C-terminal His-tag surface display, a variant must fold, retain its non-covalent FAD cofactor, dimerize, and export to the yeast surface to generate signal. Consequently, `log_activity` correlates strongly with [`daox_epseq_stability`](file:///home/beangoben/github/enzymado-benchmark/docs/data_cards/daox_epseq_stability.md) ($r \approx 0.81$ across 6,399 shared variants) and FoldX $\Delta\Delta G$ ($\rho = -0.59$), while additionally penalizing mutations in the FAD dinucleotide motif, active site, and $\beta$-hairpin dimerization loop.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Low-noise pooled 4-gate FACS-seq with high biological replicate concordance | `main` p. 3 |
| **Biological Replicates** | **$n = 2$ independent biological replicates**, **Pearson $r = 0.96$** ($n = 7,036$, $p < 0.0001$), implying **$R^2_{\max} \approx 0.96$** ($r_{\max} \approx 0.98$) | `main` p. 3 |
| **Internal Controls** | Synonymous WT variants centered at **$0.00 \pm 0.056$** ($n = 300$, vs. missense $\sigma = 0.275$); nonsense variants at **$-0.61 \pm 0.14$** ($n = 332$) | `main` p. 3 |
| **Monogenic Validation** | **Pearson $r = 0.95$** ($p < 0.0001$) across 11 individually cloned and assayed mutants | `main` p. 3 |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** *Rhodotorula gracilis* (*Rhododosporidium toruloides*) D-amino acid oxidase (EC `1.4.3.3`, UniProt `P80324`, PDB `1C0P`).
- **Surface Display Construct:** Cloned into `pYD1` in *S. cerevisiae* EBY100 as an N-terminal **Aga2p** fusion connected via a 13-residue linker (`GTPTPTPTPTGEF`) to the 365-aa RgDAOx sequence (native 368-aa sequence lacking the C-terminal peroxisomal targeting tripeptide `SKL`) with a C-terminal `6xHis` tag.
- **Mutational Breadth:** 6,404 single missense variants spanning positions 2–365 ($\sim 17.6$ substitutions per site).

---

## 4. How the Dataset Was Built & Target Transformation

- **Upstream Definition:** Weighted average FACS gate score $\beta_v$ across 4 sorting bins ($\ge 100$ reads per variant), normalized to tile-internal synonymous wild-type variants $\beta_{\text{wt}}$:
  $$F_{\text{act}} = \log_2\!\left(\frac{\beta_v}{\beta_{\text{wt}}}\right)$$
- **Enzymado Ingestion ([`rgdaox_epseq.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/rgdaox_epseq.py)):** Reconstructs the 365-aa wild-type sequence from `data/raw/real_cases/rgdaox_epseq/DMS_scores_DAOX.csv` (`7,644` rows), drops nonsense (`*`) and `NaN` rows, retains all `6,404` finite missense rows from column `Activity score (log)`, and z-score standardizes using the training split ($\mu_{\text{train}} = -0.336467$, $\sigma_{\text{train}} = 0.275295$).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Expression Induction** | *S. cerevisiae* EBY100 in galactose medium + $100\text{ mM}$ citrate-phosphate buffer (**pH 7.0**), **48 h at $20\,^\circ\text{C}$** ($200\text{ rpm}$) | `main` p. 10 |
| **Activity Assay Buffer** | $1\times\text{PBS}$ (**pH 7.4**) + **$0.75\%$ (w/v) sodium alginate** at $500\text{ cells}/\mu\text{L}$ ($2.5 \times 10^7\text{ cells}$) | `main` p. 10 |
| **Temperature & Time** | **$25\,^\circ\text{C}$ for 20 min** | `main` p. 10 |
| **Substrate & Coupling System** | **$130\text{ mM}$ D-alanine**, $56.8\,\mu\text{M}$ HRP, $1:200$ Alexa Fluor 488 Tyramide Reagent, endogenous FAD cofactor | `main` p. 10 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`5,124` train / `640` val / `640` test).
- **Dense Positional Coverage:** Every mutated residue (2–365) has $\sim 14$ substitutions in the training set, so random-split evaluation primarily probes amino-acid substitution effects at observed positions.
