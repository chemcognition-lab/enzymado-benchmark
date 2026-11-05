---
task_id: biolip_binding
dataset_id: biolip_binding
property_name: binding
task_type: protein_ligand_global
label_type: binary
layout: Layout B
computed_statistics_ref: data/processed/properties/binding/dataset_card.json#/statistics
primary_paper_doi: https://doi.org/10.1093/nar/gkad630
domain_review_doi: https://doi.org/10.1093/bib/bbab476
---

# Dataset Card: `biolip_binding` (Enzyme–Ligand Binary Binding Classification)

> ** Computed Quantitative Profile**: Numerical dataset distributions, sequence-length quantiles, amino-acid frequencies, and MMseqs2 clustering statistics are machine-generated in [`data/processed/properties/binding/dataset_card.json`](../../data/processed/properties/binding/dataset_card.json) (`#/statistics`). This document provides the experimental, mechanistic, and methodological context required to interpret those statistics and evaluate models on `biolip_binding`.

---

## 0. Provenance & Upstream-vs-Enzymado Comparison

- **Primary Database Papers**:
  - Zhang, C., Zhang, X., Freddolino, L. & Zhang, Y. *BioLiP2: an updated structure database for biologically relevant ligand–protein interactions.* **Nucleic Acids Research** 52, D404–D412 (2024). DOI: [`10.1093/nar/gkad630`](https://doi.org/10.1093/nar/gkad630) (main PDF).
  - Yang, J., Roy, A. & Zhang, Y. *BioLiP: a semi-manually curated database for biologically relevant ligand–protein interactions.* **Nucleic Acids Research** 41, D1096–D1103 (2013). DOI: [`10.1093/nar/gks966`](https://doi.org/10.1093/nar/gks966) (SI/predecessor PDF).
- **Domain Review Paper**: Dhakal, A., McKay, C., Tanner, J. J. & Cheng, J. *Artificial intelligence in the prediction of protein–ligand interactions: recent advances and future directions.* **Briefings in Bioinformatics** 23(1), bbab476 (2022). DOI: [`10.1093/bib/bbab476`](https://doi.org/10.1093/bib/bbab476) (review PDF).
- **Upstream Repositories**:
  - **BioLiP Non-Redundant Set (`data/raw/biolip/BioLiP_nr.txt.gz`)**: Weekly-updated archive of biologically relevant PDB protein–ligand complexes clustered by receptor sequence identity.
  - **wwPDB Chemical Component Dictionary (`Components-smiles-cactvs.smi`)**: Resolves 3-letter PDB ligand identifiers (`ccd_id` / `ligand_id`) to SMILES strings.
  - **SIFTS & UniProtKB**: Maps PDB chains to `uniprot_id` and EC numbers (`scripts/data_prep/remove_uniprot_fragments.py` removes UniProt fragments).
- **License**: BioLiP2 article published under CC BY-NC 4.0 (main PDF p. 1); curation code released under the BSD license (`https://github.com/kad-ecoli/mmCIF2BioLiP`).

### Reconciliation: Upstream BioLiP2 vs. Enzymado Cleaned Counts

| Metric | BioLiP2 Paper (Zhang et al. 2024, p. 3) | Enzymado `overall` (`biolip_binding.csv`) | Enzymado `used_in_runs` | Notes & Cause of Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Total rows (interactions)** | 781,684 (all redundant PDB ligands) | **28,342** (`14,171` pos + `14,171` neg) | **28,316** | Starts from `BioLiP_nr`, filters to explicit EC enzymes, strips DNA/RNA/peptides & monoatomic ions, removes **124** UniProt fragments, and pairs 1:1 with negatives |
| **Positive interactions (`binding = 1`)** | 407,951 (redundant regular small molecules) | **14,171** (50.00%) | **14,158** (50.00%) | Deduplicated on `[protein_sequence, smiles]` |
| **Negative interactions (`binding = 0`)** | — (positives-only database) | **14,171** (50.00%) | **14,158** (50.00%) | Assigned 1:1 per positive from `biolip_negative_assignments.csv` |
| **Distinct `protein_sequence` strings** | 385,160 (redundant PDB chains) | **11,245** | **11,233** | `12` ultra-long chains ($>2,652$ aa, up to `4,426` aa) excluded in `used_in_runs` by PLM length cap |
| **Unique `uniprot_id` accessions** | — | **10,120** | **10,109** | PDB chain constructs (`11,245`) outnumber UniProt IDs (`10,120`) due to crystallographic truncations/mutations |

---

## 1. How to Think About This Dataset

### Task Formulation & Input Modalities
`biolip_binding` is a **protein–ligand global binary classification** task (`Layout B`). Each row pairs an enzyme PDB chain amino-acid sequence (`protein_sequence`) with a canonical small-molecule SMILES string (`smiles`, plus metadata `ligand_id` and `ligand_name`) and asks the model to predict `binding` $\in \{0, 1\}$. The dataset is **strictly 50% positive / 50% negative** across the full dataset and within every split.

### Physical & Mechanistic Meaning of Positive vs. Negative Labels
- **Positives (`binding = 1`)**: Structurally resolved PDB enzyme–ligand complexes where the small molecule makes at least two inter-atomic contacts ($d \le r_{\text{vdW},i} + r_{\text{vdW},j} + 0.5\text{ \AA}$) to $\ge 2$ protein residues and passes BioLiP2's biological-relevance pipeline (filtering out 463 common crystallization buffers/additives such as PEG, glycerol, Tris, and sulfate unless rescued by stoichiometry, binding-site topology, and PubMed abstract text validation). Positives span cognate substrates, products, organic cofactors ($\text{NAD}^+$, $\text{FAD}$, $\text{PLP}$, $\text{CoA}$, $\text{ATP}$, heme), and synthetic inhibitors.
- **Negatives (`binding = 0`)**: Decoy enzyme–ligand pairs added 1:1 via `src/enzymado/propertyprep/biolip.py` (`add_negatives`) from `biolip_negative_assignments.csv`, pairing each enzyme construct with a BioLiP small molecule not observed to bind that enzyme.
- **Construct sequences, not canonical UniProt sequences**: Because `protein_sequence` comes directly from the PDB chain sequence in `BioLiP_nr.txt.gz` (restricted by BioLiP2 to chains with $\ge 30$ amino acids), different crystal structures of the same `uniprot_id` with different N/C-terminal truncations appear as distinct `protein_sequence` entries (`11,245` unique sequences vs. `10,120` unique `uniprot_id`).

---

## 2. Label Noise & Theoretical Performance Ceiling

### Primary Sources of Aleatoric & Epistatic Noise
1. **Untested false negatives in decoy pairing**: Because negative pairs (`binding = 0`) are generated by cross-pairing rather than experimental non-binding measurements, broad-specificity enzymes or cofactor-binding domains may physically bind a sampled negative ligand in solution.
2. **Residual crystallization additives among positives**: BioLiP's semi-manual filter achieves $\sim 95\text{–}97\%$ precision, meaning a small fraction of non-physiological buffer molecules or surrogate ligands mentioned in a paper's abstract remain labeled as `binding = 1`.
3. **Discretization of a 10-order-of-magnitude affinity continuum**: Both nanomolar tight-binding inhibitors ($K_d \sim 10^{-9}\text{ M}$) and millimolar crystallographic soaked fragments ($K_d \sim 10^{-2}\text{ M}$) receive identical `binding = 1` labels.

---

## 3. Specificity & Breadth of the Dataset

- **Protein Diversity & MMseqs2 Clustering**: Across **28,342 rows** (`11,245` unique sequences; `28,316` rows and `11,233` sequences in `used_in_runs`), MMseqs2 clustering at 50% sequence identity and 80% coverage (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`) yields **5,193 clusters** (`ratio_unique_variants = 0.4618`; `5,187` clusters in `used_in_runs`), with **3,234 singletons** (62.3% of clusters) and a maximum cluster size of **44** (mean `2.165`).
- **Sequence Lengths & Non-Standard Residues**: Sequence lengths span **30 to 4,426 aa** in `overall` (mean `378.41`, std `185.64`, median `345.0`, IQR `261–459`; `210` rows $>1,024$ aa) and **30 to 2,652 aa** in `used_in_runs` (mean `376.19`, std `176.26`, median `345.0`). The sharp minimum length of **30 aa** reflects BioLiP2's cutoff separating protein chains ($\ge 30$ aa) from peptide ligands ($< 30$ aa). Additionally, **450 rows** (`448` in `used_in_runs`) contain non-standard amino-acid characters (`X`, `U`, `O`) arising from selenomethionine (`MSE`) or unresolved/modified crystallographic residues in PDB chains.

---

## 4. How the Dataset Was Built

1. **Raw BioLiP Download & Tabular Conversion (`scripts/data_prep/download_biolip.sh` & `convert_biolip_to_csv.py`)**:
   - Downloaded `BioLiP_nr.txt.gz` and PDB CCD SMILES (`Components-smiles-cactvs.smi`).
2. **Enzyme Filtering, Biopolymer/Ion Removal & SMILES Canonicalisation (`src/enzymado/propertyprep/biolip.py`)**:
   - Retained rows with non-null `uniprot_id`, `ligand_id`, and an explicit EC number (`has_explicit_ec_number`).
   - Excluded biopolymer ligand codes (`dna`, `rna`, `pep`, `peptide`, `protein`, `polypeptide`) and monoatomic metal/halide ions (`is_ion_smiles`, 31 element symbols with 1 heavy atom).
   - Mapped `ligand_id` to canonical RDKit SMILES and deduplicated positives on `['protein_sequence', 'smiles']`.
3. **UniProt Fragment Removal & 1:1 Negative Assignment (`scripts/data_prep/remove_uniprot_fragments.py`)**:
   - Removed **124** positive rows linked to UniProt sequence fragments, leaving **14,171** positives.
   - Appended **14,171** non-overlapping negative pairs from `biolip_negative_assignments.csv` (`add_negatives`), yielding **28,342** balanced rows.

---

## 5. Experimental Measurement & STRENDA Compliance

- **Measurement Protocol**: Atomic contact detection ($d \le r_{\text{vdW},1} + r_{\text{vdW},2} + 0.5\text{ \AA}$) in X-ray, cryo-EM, and NMR structures from the wwPDB, coupled with BioLiP2 artifact-list filtering and PubMed abstract validation.
- **STRENDA Reporting Gaps**: Crystallization pH, temperature, soaking concentration, and thermodynamic dissociation constants ($K_d / K_i$) are not included in `biolip_binding.csv`.

---

## 6. Split Semantics, Leakage Controls & Caveats

| Split Scheme | Train Rows (Pos Prevalence / Proteins) | Val Rows (Pos Prevalence / Proteins) | Test Rows (Pos Prevalence / Proteins) | `uniprot_id` Accessions Straddling Splits |
| :--- | :---: | :---: | :---: | :---: |
| **`random_80_10_10`** | 22,740 (`0.5000` / 8,996) | 2,844 (`0.5000` / 1,124) | 2,758 (`0.5000` / 1,125) | `327 / 10,120` (3.23%) |
| **`similarity_mmseqs2_80_10_10`** | 22,674 (`0.5000` / 9,003) | 2,834 (`0.5000` / 1,111) | 2,834 (`0.5000` / 1,131) | `47 / 10,120` (0.46%) |

1. **Exact 50.0% Positive Prevalence in Every Split**: Because positive and negative pairs are balanced per protein construct and splits are grouped by `protein_sequence`, `train`, `val`, and `test` each have an exact `0.5000` positive rate in both `random_80_10_10` and `similarity_mmseqs2_80_10_10`.
2. **Construct Sequence Grouping vs. Multi-Domain `uniprot_id` Straddling**: Grouping by `protein_sequence` (`11,245` unique PDB construct strings) guarantees zero exact sequence overlap across splits. In `random_80_10_10`, `327` `uniprot_id` accessions have slightly different crystallographic constructs in different splits; in `similarity_mmseqs2_80_10_10` (`--min-seq-id 0.3`), MMseqs2 groups all overlapping constructs of the same protein into a single cluster, leaving only `47` `uniprot_id` accessions straddling splits (where separate, non-overlapping domains of a multidomain enzyme were crystallised independently and share $<30\%$ sequence identity).

