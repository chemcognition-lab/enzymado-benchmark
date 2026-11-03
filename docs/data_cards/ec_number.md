---
task_id: swissprot_ec
dataset_id: swissprot_ec
property_name: ec_number
task_type: protein_global
label_type: hierarchical_multilabel
layout: Layout A
computed_statistics_ref: data/processed/properties/ec_number/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkae1010
domain_review_doi: https://doi.org/10.1093/nar/gkn582
---

# Dataset Card: `swissprot_ec` (Enzyme Commission Number Classification)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/ec_number/dataset_card.json`](../../data/processed/properties/ec_number/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `swissprot_ec`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Papers**:
  - The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2025.* **Nucleic Acids Research** 53, D609–D617 (2025). DOI: [`10.1093/nar/gkae1010`](https://doi.org/10.1093/nar/gkae1010) ([main PDF](../pdfs/uniprot_consortium2025_nar.pdf)).
  - The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2023.* **Nucleic Acids Research** 51, D523–D531 (2023). DOI: [`10.1093/nar/gkac1052`](https://doi.org/10.1093/nar/gkac1052) ([2023 PDF](../pdfs/uniprot_consortium2023_nar.pdf)).
- **Domain Review / Nomenclature Paper**: McDonald, A. G., Boyce, S. & Tipton, K. F. *ExplorEnz: the primary source of the IUBMB enzyme list.* **Nucleic Acids Research** 37, D593–D597 (2009). DOI: [`10.1093/nar/gkn582`](https://doi.org/10.1093/nar/gkn582) ([review PDF](../pdfs/rev_ec_mcdonald2023_nar.pdf)).
- **Upstream Repositories**:
  - **UniProtKB/Swiss-Prot (`data/raw/swissprot/swissprot_ec.tsv`)**: Manually reviewed protein knowledgebase queried via `scripts/data_prep/download_swissprot_ec.sh` (`(reviewed:true) AND (ec:*)`).
  - **IUBMB Enzyme Nomenclature (ExplorEnz) & Rhea**: Formal four-part hierarchical reaction classification and ChEBI-linked biochemical reaction knowledgebase used by Swiss-Prot curators.
- **License**: UniProtKB data and articles are distributed under CC BY 4.0 (main PDF p. 1).

### Reconciliation: Upstream Swiss-Prot vs. Enzymado Cleaned Counts

| Metric | UniProt 2025 Paper (p. 2) | Raw Swiss-Prot Export (`swissprot_ec.tsv`) | Enzymado `overall` (`swissprot_ec.csv`) | Enzymado `used_in_runs` | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Reviewed Swiss-Prot enzyme entries** | 231,709 (Rhea-linked) | 279,501 (all `ec:*`) | **237,526** | **237,255** | Sequence deduplication (`process_swissprot_ec`) + removal of **4,581** UniProt fragments (`remove_uniprot_fragments.py`) |
| **Distinct `protein_sequence` strings** | — | — | **237,474** | **237,203** | `271` ultra-long sequences ($>2,999$ aa, up to `7,182` aa) excluded in `used_in_runs` by PLM length cap |
| **Unique `uniprot_id` accessions** | — | 279,501 | **237,526** | **237,255** | `52` identical sequences shared across distinct Swiss-Prot accessions with different EC annotations |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`swissprot_ec` is a **sequence-only protein-global hierarchical multi-label classification** task (`Layout A`). Given an enzyme amino-acid sequence (`protein_sequence`), the model predicts its four-component IUBMB Enzyme Commission identifier(s) (`ec_number`, formatted as semicolon-separated strings such as `'3.2.1.39'` or `'3.2.2.-; 3.2.2.6'`).
- **Four-level hierarchy**:
  1. **Class (1st digit, 1–7)**: `1` Oxidoreductases, `2` Transferases, `3` Hydrolases, `4` Lyases, `5` Isomerases, `6` Ligases, `7` Translocases.
  2. **Subclass (2nd digit)**: Type of group/bond/donor acted upon.
  3. **Sub-subclass (3rd digit)**: Acceptor or more specific reaction sub-type.
  4. **Serial number (4th digit)**: Specific substrate/reaction within the sub-subclass.
- **Multi-label & partial EC annotations**: Bifunctional fusion enzymes and catalytically promiscuous enzymes carry multiple semicolon-separated EC numbers, and incompletely characterized specificities carry wildcard dash tokens (`-`, e.g., `'3.2.2.-'`).

### Reaction Taxonomy vs. Sequence Homology
Because the IUBMB EC list classifies **overall chemical reactions** rather than protein structures (McDonald et al. 2009, p. 1), two evolutionary phenomena challenge sequence-to-EC models:
1. **Non-homologous isofunctional enzymes (convergent evolution)**: Unrelated structural folds that evolved independently to catalyse the same reaction share the exact same 4-digit EC number.
2. **Divergent mechanistically diverse superfamilies**: Homologous enzymes sharing the same fold and $>30\%$ sequence identity (e.g., enolase, radical SAM, or amidohydrolase superfamilies) often catalyse reactions in different EC classes or subclasses.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Curator-reviewed homology propagation (UniRule / InterPro / PANTHER)**: While Swiss-Prot is manually reviewed, the majority of its 237,526 enzyme entries are bacterial and eukaryotic orthologs annotated by expert rule propagation (`UniRule`) from experimentally characterized model enzymes. Within conserved housekeeping families, this makes labels highly consistent across orthologs, but can over-propagate 4th-digit substrate specificity onto paralogs whose substrate preference has shifted.
2. **Extreme long-tail sparsity at Level 4**: While Level 1 has only 7 classes with tens of thousands of proteins each, Level 4 spans thousands of serial numbers, many of which contain $<5$ sequences across all of Swiss-Prot.

---

## 3. Specificity & Breadth of the Dataset

- **Dense Orthologous Clustering (`ratio_unique_variants = 0.1488`)**: Across **237,526 rows** (`237,474` distinct sequences; `237,255` rows and `237,203` sequences in `used_in_runs`), MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) groups the 237,474 sequences into **35,342 clusters** (`35,193` in `used_in_runs`), with **21,671 singletons** (61.3% of clusters), a mean cluster size of **6.719**, and a maximum cluster size of **1,039**. This low `ratio_unique_variants` (`0.1488`) reflects Swiss-Prot's deep sampling of conserved orthologous enzymes across hundreds of reference proteomes.
- **Sequence Lengths**: In `overall`, lengths span **14 to 7,182 aa** (mean `402.07`, std `232.97`, median `359.0`, IQR `259–484`), with `4,295` rows $>1,024$ aa and `322` rows containing non-standard amino acids (`U`, `O`, `X`, `B`, `Z`). In `used_in_runs`, `271` ultra-long sequences ($>2,999$ aa) are excluded, bounding lengths to **14–2,999 aa** (mean `398.18`, std `210.71`, median `358.0`; `4,024` rows $>1,024$ aa).

---

## 4. How the Dataset Was Built

1. **Swiss-Prot Query (`scripts/data_prep/download_swissprot_ec.sh`)**:
   - Downloaded all `279,501` reviewed UniProtKB/Swiss-Prot entries matching `(reviewed:true) AND (ec:*)` to `data/raw/swissprot/swissprot_ec.tsv`.
2. **Sequence & EC Validation (`src/enzymado/propertyprep/swissprot.py`)**:
   - Validated sequences (`common.is_valid_sequence`), parsed EC numbers (`parsing.parse_ec_numbers`), and deduplicated sequences.
3. **Fragment Removal (`scripts/data_prep/remove_uniprot_fragments.py`)**:
   - Removed **4,581** UniProt sequence fragments (`3,667` train, `458` val, `456` test), leaving **237,526** rows.

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Official IUBMB EC assignment requires published biochemical evidence of the catalysed reaction; individual Swiss-Prot records combine direct experimental literature curation (`ECO:0000269`) and reviewed homology/rule transfer (`UniRule`).
- **STRENDA Reporting Gaps**: Evidence codes (`ECO`) are not exported into `swissprot_ec.csv`, so direct experimental characterisations and curator-reviewed orthologous transfers are weighted equally.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Rows (Unique Seqs) | Val Rows (Unique Seqs) | Test Rows (Unique Seqs) | Sequences Straddling Splits |
| :--- | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 190,023 (189,979) | 23,752 (23,747) | 23,751 (23,748) | `0` |
| **`similarity_mmseqs2_80_10_10`** | 190,020 (189,979) | 23,753 (23,747) | 23,753 (23,748) | `0` |

1. **Massive Ortholog Redundancy in `random_80_10_10` vs. True Remote-Homology Generalisation in `similarity_mmseqs2_80_10_10`**: Because the average 50%-identity cluster in `swissprot_ec` contains `6.72` orthologous sequences (and up to `1,039` sequences for ubiquitous housekeeping enzymes), `random_80_10_10` places close orthologs in both `train` and `test`, allowing trivial homology lookup to achieve near-perfect accuracy. Always prioritize **`similarity_mmseqs2_80_10_10`** (`--min-seq-id 0.3`) when evaluating de novo enzyme function prediction.
2. **Zero Sequence Leakage**: Both splits group by `protein_sequence`, keeping the `52` identical-sequence paralog pairs inside the same split.

