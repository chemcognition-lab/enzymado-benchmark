---
task_id: swissprot_active
dataset_id: swissprot_active
property_name: active_sites
task_type: protein_residue_level
label_type: residue_binary
layout: Layout A
computed_statistics_ref: data/processed/properties/active_sites/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkae1010
domain_review_doi: https://doi.org/10.1093/nar/gkx1012
---

# Dataset Card: `swissprot_active` (Catalytic Active-Site Residue Identification)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/active_sites/dataset_card.json`](../../data/processed/properties/active_sites/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `swissprot_active`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Papers**:
  - The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2025.* **Nucleic Acids Research** 53, D609–D617 (2025). DOI: [`10.1093/nar/gkae1010`](https://doi.org/10.1093/nar/gkae1010) ([main PDF](../pdfs/uniprot_consortium2025_nar.pdf)).
  - The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2023.* **Nucleic Acids Research** 51, D523–D531 (2023). DOI: [`10.1093/nar/gkac1052`](https://doi.org/10.1093/nar/gkac1052) ([2023 PDF](../pdfs/uniprot_consortium2023_nar.pdf)).
- **Domain Review / Mechanistic Atlas Paper**: Ribeiro, A. J. M. et al. *Mechanism and Catalytic Site Atlas (M-CSA): a database of enzyme reaction mechanisms and active sites.* **Nucleic Acids Research** 46, D618–D623 (2018). DOI: [`10.1093/nar/gkx1012`](https://doi.org/10.1093/nar/gkx1012) ([review PDF](../pdfs/rev_active_sites_mcsa_ribeiro2018_nar.pdf), [SI PDF](../pdfs/rev_active_sites_mcsa_ribeiro2018_suppinfo.pdf)).
- **Upstream Repositories**:
  - **UniProtKB/Swiss-Prot (`data/raw/swissprot/swissprot_ec.tsv`, `Active site` / `ACT_SITE` column)**: Reviewed enzyme sequence records annotated with catalytic residue coordinates.
  - **M-CSA & InterPro / UniRule**: Primary mechanistic literature database (`961` curated reference enzyme mechanisms/sites) and rule-based orthologous transfer pipelines underpinning Swiss-Prot `ACT_SITE` annotations.
- **License**: UniProtKB and M-CSA articles and data are published under CC BY 4.0.

### Reconciliation: Upstream Swiss-Prot vs. Enzymado Cleaned Counts

| Metric | Raw Swiss-Prot Export (`swissprot_ec.tsv`) | Enzymado `overall` (`swissprot_active_site.csv`) | Enzymado `used_in_runs` | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Total rows (proteins)** | 104,779 (non-null `Active site`) | **89,725** | **89,611** | Parsed valid integer `ACT_SITE` positions, deduplicated by `protein_sequence`, and removed **487** UniProt fragments |
| **Distinct `protein_sequence` strings** | — | **89,725** | **89,611** | `114` ultra-long sequences ($>2,999$ aa, up to `7,096` aa) excluded in `used_in_runs` by PLM length cap |
| **Unique `uniprot_id` accessions** | 104,779 | **89,725** | **89,611** | Strict 1:1 correspondence between `uniprot_id` and `protein_sequence` |
| **Per-residue positive rate** | — | **0.0041** (0.41%) | **0.0041** (0.41%) | ~1.7 catalytic residues per ~415-aa enzyme chain |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`swissprot_active` is a **sequence-only per-residue binary classification** task (`Layout A`). Given an enzyme amino-acid sequence of length $L$ (`protein_sequence`), the model predicts a binary indicator $y_i \in \{0, 1\}$ for every residue $i \in \{0, \dots, L-1\}$, where $y_i = 1$ iff 0-indexed position $i$ is listed in `active_site` (e.g., `'[500, 578, 582]'`).

### Physical & Mechanistic Meaning of `ACT_SITE` vs. `BINDING`
In UniProtKB/Swiss-Prot and M-CSA, **catalytic active-site residues (`ACT_SITE`, modeled in `swissprot_active`) are strictly distinct from substrate/cofactor binding-site residues (`BINDING`, modeled in `swissprot_binding`)**:
- `ACT_SITE` marks only the 1–4 amino acids that directly participate in bond making/breaking during catalysis (general acids/bases, covalent nucleophiles, proton shuttles, or transition-state electrostatic stabilizers).
- As Ribeiro et al. (2018, Fig. 1, p. 3) show across M-CSA, catalytic residues are heavily skewed toward ionizable and nucleophilic side chains (**His, Cys, Asp, Glu, Lys, Arg, Ser, Thr, Tyr**—with **Histidine 7× more frequent** than random sequence background), whereas hydrophobic residues (Ala, Leu, Gly, Val) rarely play a direct catalytic role.

### Extreme Per-Residue Class Imbalance (`0.41%` Positive)
Across the $\sim 37.3\text{ million}$ amino-acid tokens in `swissprot_active`, only **0.41% (`0.0041`)** are positive catalytic residues (`99.59%` negative). A trivial majority-class classifier achieves `99.59%` per-residue accuracy; evaluation must therefore rely on **AUPRC**, **macro/positive-class $F_1$**, and **MCC**.

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Partial catalytic triad/dyad annotation in UniRule transfers**: Many Swiss-Prot records annotate only the single covalent nucleophile or primary general acid/base (1 residue) matched by a PROSITE/InterPro signature, omitting 1–2 partner residues that complete the catalytic apparatus in 3D space.
2. **Homology-propagated consistency vs. rare active-site migration**: Because most of the 89,725 records are orthologous transfers from ~1,000 experimentally characterized reference enzymes, labels are highly consistent across orthologs (`random_80_10_10`) but can miss rare active-site migrations in divergent superfamily members.

---

## 3. Specificity & Breadth of the Dataset

- **High Orthologous Redundancy (`ratio_unique_variants = 0.1219`)**: Across **89,725 proteins** (`89,611` in `used_in_runs`), MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) yields **10,934 clusters** (`10,874` in `used_in_runs`), with **6,419 singletons** (58.7% of clusters), a mean cluster size of **8.206**, and a maximum cluster size of **738**.
- **Sequence Lengths**: Lengths span **19 to 7,096 aa** in `overall` (mean `415.27`, std `232.02`, median `366.0`, IQR `276–491`; `1,554` rows $>1,024$ aa; `109` rows with non-standard amino acids such as catalytic selenocysteine `U`) and **19 to 2,999 aa** in `used_in_runs` (mean `410.48`, std `203.24`, median `366.0`; `1,440` rows $>1,024$ aa).

---

## 4. How the Dataset Was Built

1. **ACT_SITE Extraction & 1-to-0 Index Conversion (`src/enzymado/propertyprep/swissprot.py`)**:
   - Filtered `data/raw/swissprot/swissprot_ec.tsv` (`104,779` non-null `Active site` rows) to valid amino-acid sequences.
   - Parsed 1-indexed `ACT_SITE` position tokens and ranges via `parsing.parse_positions_from_tokens(text, "ACT_SITE")` and subtracted `1` to store sorted **0-indexed** integer lists in `active_site`.
   - Dropped rows with empty parsed lists and deduplicated on `protein_sequence`.
2. **Fragment Removal (`scripts/data_prep/remove_uniprot_fragments.py`)**:
   - Removed **487** UniProt sequence fragments (`391` train, `50` val, `46` test), leaving **89,725** full-length proteins.

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Reference catalytic residues are established via site-directed mutagenesis (loss of $k_{\text{cat}}$), X-ray/cryo-EM complex structures, covalent mechanism-based inhibitors, pH-rate profiles, and QM/MM calculations (Ribeiro et al. 2018, p. 2), and propagated to reviewed orthologs via `UniRule`.
- **STRENDA Reporting Gaps**: Specific catalytic role qualifiers (`/note="Nucleophile"`, `/note="Proton donor"`) and ECO evidence codes are collapsed into a single binary residue indicator.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Proteins (Pos Rate) | Val Proteins (Pos Rate) | Test Proteins (Pos Rate) | Sequences / Accessions Straddling Splits |
| :--- | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 71,780 (`0.0041`) | 8,972 (`0.0041`) | 8,973 (`0.0041`) | `0` |
| **`similarity_mmseqs2_80_10_10`** | 71,780 (`0.0041`) | 8,972 (`0.0041`) | 8,973 (`0.0041`) | `0` |

1. **Conserved Motif Memorization in `random_80_10_10` vs. Remote-Homology Generalisation in `similarity_mmseqs2_80_10_10`**: Because each 50%-identity cluster contains an average of `8.21` orthologs sharing conserved active-site sequence motifs, `random_80_10_10` allows local motif memorization across train and test. **`similarity_mmseqs2_80_10_10`** (`--min-seq-id 0.3`) isolates entire homologous families into single splits, testing whether a model can identify catalytic residues on unseen folds.
2. **0-Indexed Positions**: All indices in `active_site` are **0-indexed** (`0 <= i < L`), shifted by `-1` relative to 1-indexed UniProt feature tables.

