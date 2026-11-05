---
task_id: swissprot_binding
dataset_id: swissprot_binding
property_name: binding_site
task_type: protein_residue_level
label_type: residue_binary
layout: Layout A
computed_statistics_ref: data/processed/properties/binding_site/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkae1010
domain_review_doi: https://doi.org/10.1186/s13321-024-00907-z
---

# Dataset Card: `swissprot_binding` (Substrate, Cofactor & Metal Binding-Site Residue Identification)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/binding_site/dataset_card.json`](../../data/processed/properties/binding_site/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `swissprot_binding`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Papers**:
  - The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2025.* **Nucleic Acids Research** 53, D609–D617 (2025). DOI: [`10.1093/nar/gkae1010`](https://doi.org/10.1093/nar/gkae1010) (main PDF).
  - The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2023.* **Nucleic Acids Research** 51, D523–D531 (2023). DOI: [`10.1093/nar/gkac1052`](https://doi.org/10.1093/nar/gkac1052) (2023 PDF).
- **Domain Review / Structural Pocket Analysis Paper**: Vavra, O. et al. *Large-scale annotation of biochemically relevant pockets and tunnels in cognate enzyme–ligand complexes.* **Journal of Cheminformatics** 16, 114 (2024). DOI: [`10.1186/s13321-024-00907-z`](https://doi.org/10.1186/s13321-024-00907-z) (review PDF).
- **Upstream Repositories**:
  - **UniProtKB/Swiss-Prot (`data/raw/swissprot/swissprot_ec.tsv`, `Binding site` / `BINDING` column)**: Reviewed enzyme sequence records annotated with ChEBI-standardized substrate, cofactor, and metal ion binding positions.
  - **ChEBI, PDBe/SIFTS & InterPro / UniRule**: Chemical ontology and homology rule systems used by Swiss-Prot to annotate and propagate binding-pocket residues.
- **License**: Both UniProtKB and Vavra et al. (2024) are published under CC BY 4.0.

### Reconciliation: Upstream Swiss-Prot vs. Enzymado Cleaned Counts

| Metric | Raw Swiss-Prot Export (`swissprot_ec.tsv`) | Enzymado `overall` (`swissprot_binding_site.csv`) | Enzymado `used_in_runs` | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Total rows (proteins)** | 185,292 (non-null `Binding site`) | **157,433** | **157,264** | Parsed valid integer `BINDING` positions/ranges, deduplicated by `protein_sequence`, and removed **804** UniProt fragments |
| **Distinct `protein_sequence` strings** | — | **157,433** | **157,264** | `169` ultra-long sequences ($>2,999$ aa, up to `7,096` aa) excluded in `used_in_runs` by PLM length cap |
| **Unique `uniprot_id` accessions** | 185,292 | **157,433** | **157,264** | Strict 1:1 correspondence between `uniprot_id` and `protein_sequence` |
| **Per-residue positive rate** | — | **0.0263** (2.63%) | **0.0263** (2.63%) | ~10.6 binding residues per ~403-aa enzyme chain (`6.4×` denser than `active_sites`) |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`swissprot_binding` is a **sequence-only per-residue binary classification** task (`Layout A`). Given an enzyme amino-acid sequence of length $L$ (`protein_sequence`), the model predicts a binary indicator $y_i \in \{0, 1\}$ for every residue $i \in \{0, \dots, L-1\}$, where $y_i = 1$ iff 0-indexed position $i$ appears in `binding_positions` (e.g., `'[504, 576, 578, 582, 660]'`).

### Physical & Mechanistic Meaning of `BINDING` vs. `ACT_SITE`
Whereas `swissprot_active` (`ACT_SITE`) marks only the 1–4 catalytic residues that make or break covalent bonds (`0.41%` positive residues), `swissprot_binding` (`BINDING`) marks **all residues annotated as binding physiological substrates, organic cofactors ($\text{ATP}$, $\text{NAD(P)}^+$, $\text{FAD}$, $\text{PLP}$, $\text{CoA}$, heme), catalytic or structural metal ions ($\text{Zn}^{2+}$, $\text{Mg}^{2+}$, $\text{Mn}^{2+}$, $\text{Fe}^{2+/3+}$, $\text{Ca}^{2+}$), or allosteric effectors**.
- **Contiguous motif intervals**: Unlike `ACT_SITE` (isolated single positions), `BINDING` frequently includes contiguous sequence intervals (e.g., `BINDING 12..18` for P-loop / Walker-A or Rossmann dinucleotide-binding motifs), which `parsing.parse_positions_from_tokens` expands inclusively into consecutive 0-indexed integers.
- **Per-residue positive prevalence (`2.63%`)**: Across all $\sim 63.4\text{ million}$ residues in `swissprot_binding`, **2.63% (`0.0263`)** are positive binding residues (`97.37%` negative)—still strongly imbalanced, requiring **AUPRC**, **$F_1$**, and **MCC** rather than accuracy.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Direct hydrogen/coordination bonds vs. full 3D cavity shells**: UniProt `BINDING` records emphasize residues making direct hydrogen bonds, salt bridges, metal coordination bonds, or conserved motif contacts rather than the complete geometric shell of van der Waals pocket-lining residues. In a large-scale comparison across 11,046 annotated cognate enzyme–ligand PDB structures, Vavra et al. (2024, Table 1 & p. 6) found that **76% (`8,350 / 11,046`)** matched Fpocket-selected cognate pockets to UniProt/CSA essential residues, while 24% did not overlap due to multi-subpocket spanning or surface-exposed binding.
2. **Induced-fit pocket plasticity**: Comparing 2,904 pairs of ligand-bound (holo) and ligand-free (apo) enzyme structures, Vavra et al. (2024, p. 6) showed that **24%** of pairs shared no common top-5 CAVER tunnels due to pocket collapse/narrowing in the apo state.
3. **Ligand-agnostic union**: Because no ligand structure is passed as input, `binding_positions` merges cofactor, metal, and substrate binding sites into a single binary mask.

---

## 3. Specificity & Breadth of the Dataset

- **Highest Orthologous Concentration in Enzymado (`ratio_unique_variants = 0.1143`)**: Across **157,433 proteins** (`157,264` in `used_in_runs`), MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) yields **17,999 clusters** (`17,909` in `used_in_runs`), with **10,636 singletons** (59.1% of clusters), a mean cluster size of **8.747**, and a maximum cluster size of **1,039**.
- **Sequence Lengths**: Lengths span **16 to 7,096 aa** in `overall` (mean `402.83`, std `220.49`, median `359.0`, IQR `267–481`; `2,428` rows $>1,024$ aa; `245` rows with non-standard amino acids) and **16 to 2,999 aa** in `used_in_runs` (mean `399.28`, std `197.70`, median `358.0`; `2,259` rows $>1,024$ aa).

---

## 4. How the Dataset Was Built

1. **BINDING Extraction & Range Expansion (`src/enzymado/propertyprep/swissprot.py`)**:
   - Filtered `data/raw/swissprot/swissprot_ec.tsv` (`185,292` non-null `Binding site` rows) to valid amino-acid sequences.
   - Parsed 1-indexed `BINDING` single positions and `start..end` ranges via `parsing.parse_positions_from_tokens(text, "BINDING")`, expanded ranges inclusively, and subtracted `1` to store sorted **0-indexed** integer lists in `binding_positions`.
   - Dropped rows with empty parsed lists and deduplicated on `protein_sequence`.
2. **Fragment Removal (`scripts/data_prep/remove_uniprot_fragments.py`)**:
   - Removed **804** UniProt sequence fragments (`640` train, `81` val, `83` test), leaving **157,433** full-length proteins.

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Derived from X-ray, cryo-EM, and NMR complex structures (cross-linked via PDBe/SIFTS and standardized to ChEBI ligand identifiers) and propagated across reviewed orthologs via `UniRule`.
- **STRENDA Reporting Gaps**: Specific ChEBI ligand identifiers (`/ligand="ATP"`, `/ligand="Zn(2+)"`) and ECO evidence codes are collapsed into a single ligand-agnostic residue mask.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Proteins (Pos Rate) | Val Proteins (Pos Rate) | Test Proteins (Pos Rate) | Sequences / Accessions Straddling Splits |
| :--- | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 125,947 (`0.0263`) | 15,743 (`0.0265`) | 15,743 (`0.0263`) | `0` |
| **`similarity_mmseqs2_80_10_10`** | 125,946 (`0.0263`) | 15,743 (`0.0265`) | 15,744 (`0.0264`) | `0` |

1. **Orthologous Motif Sharing in `random_80_10_10` vs. Cross-Family Generalisation in `similarity_mmseqs2_80_10_10`**: With an average of `8.75` orthologs per 50%-identity cluster, `random_80_10_10` places nearly identical cofactor- and metal-binding pockets across train and test. **`similarity_mmseqs2_80_10_10`** (`--min-seq-id 0.3`) isolates entire homologous families into single splits.
2. **0-Indexed Positions**: All indices in `binding_positions` are **0-indexed** (`0 <= i < L`).

