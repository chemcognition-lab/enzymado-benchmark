---
task_id: proteinea_solubility
dataset_id: proteinea_solubility
property_name: solubility
task_type: protein_global
label_type: binary
layout: Layout A
schema_version: 0.1.0
generated_at: 2026-04-20T00:00:00Z
---

# `proteinea_solubility` (`solubility`) — Dataset Card

## 0. At-a-Glance Summary

| Field | Value |
|---|---|
| **Task / Dataset ID** | `proteinea_solubility` (`data/processed/properties/solubility/`) |
| **Property** | Binary heterologous recombinant enzyme solubility (`1` = soluble, `0` = insoluble) |
| **Task / Label Type** | `protein_global` / `binary` (`Layout A`: single protein sequence $\to$ binary class in $\{0, 1\}$) |
| **Primary Upstream Sources** | **Ankh / Proteinea** ([Elnaggar et al. 2023](https://doi.org/10.48550/arXiv.2301.06568), `Proteinea/solubility`), **DeepSol** ([Khurana et al. 2018](https://doi.org/10.1093/bioinformatics/bty166)), **PROSO II / PepcDB** (Smialowski et al. 2012; Chang et al. 2014), intersected with **UniProtKB/Swiss-Prot** reviewed enzymes (`swissprot_ec.tsv`) |
| **Total Cleaned Rows** | **3,665** rows across **3,665** unique enzyme sequences (`100%` retained in `used_in_runs`; `0` duplicate sequences) |
| **Positive Class Balance** | **55.99% soluble** (`2,052` soluble [`1`] vs. `1,613` insoluble [`0`]; shifted from `41.97%` soluble in the raw `71,421`-protein DeepSol corpus due to Swiss-Prot enzyme intersection) |
| **Sequence Length (aa)** | `39` – `1,196` aa (mean `355.02 ± 176.29`, median `328.0`, IQR `227.0–446.0`; `24` sequences `> 1,024` aa; `0` non-standard AAs; bounded by DeepSol's $L = 1,200$ cap) |
| **Sequence Diversity (MMseqs2 50% id)** | **3,167 clusters** across `3,665` sequences (`ratio_unique_variants = 0.8641`; **2,844 singletons** [`89.8%` of clusters]; mean cluster size `1.157`, max `9`) |
| **Standardized Splits** | `random_80_10_10` (`2,931` / `367` / `367`) and `similarity_mmseqs2_80_10_10` (`2,932` / `367` / `366`) |
| **Recommended Reporting Split** | `similarity_mmseqs2_80_10_10` (`0` shared 50%-identity clusters across train/val/test; `246` clusters leaked in `random_80_10_10`) |
| **Computed Statistics Reference** | `data/processed/properties/solubility/dataset_card.json#/statistics` |

---

## 1. Provenance & Curation Pipeline

### 1.1 Upstream Databases: PepcDB/TargetTrack $\to$ PROSO II $\to$ DeepSol $\to$ Ankh (`Proteinea/solubility`)
The `proteinea_solubility` benchmark traces through four stages of curation:
1. **PROSO II & Chang et al. Experimental Compilation (Smialowski et al. 2012; Chang et al. 2014):** The foundational training pool was compiled for **PROSO II** (*FEBS J.* 279, 2192–2200, 2012) from the Protein Structure Initiative (PSI) **PepcDB / TargetTrack** repository and UniProtKB, comprising `58,689` soluble and `70,954` insoluble proteins (`129,643` total). An independent benchmark test set of `2,001` *Escherichia coli*-expressed proteins (`1,000` soluble and `1,001` insoluble) was curated by Chang et al. (2014).
2. **DeepSol Redundancy Reduction & Decontamination ([Khurana et al. 2018](https://doi.org/10.1093/bioinformatics/bty166)):** To eliminate high-identity sequence bias, Khurana et al. (*Bioinformatics* 34, 2605–2613, 2018) clustered the PROSO II training set with CD-HIT at `90%` global sequence identity, removed any training sequence sharing $\ge 30\%$ sequence identity with any sequence in the `2,001`-protein Chang et al. test set, excluded sequences containing non-standard amino acids, and capped sequence length at $L = 1,200$ residues. This yielded `69,420` training/validation sequences (`28,972` soluble [`41.73%`] and `40,448` insoluble [`58.27%`]) plus the `2,001` independent test sequences (`71,421` total sequences).
3. **Ankh / Proteinea Distribution ([Elnaggar et al. 2023](https://doi.org/10.48550/arXiv.2301.06568)):** For the **Ankh** protein language model benchmark, Elnaggar et al. (2023) packaged the DeepSol corpus on HuggingFace (`Proteinea/solubility`) with `62,478` training, `6,942` validation, and `2,001` test rows (`71,421` total rows, downloaded to `data/raw/proteinea/proteinea_solubility_raw.csv` by `scripts/data_prep/download_proteinea_datasets.py`).

### 1.2 Enzymado Enzyme-Specific Intersection & Fragment Removal
Because `Proteinea/solubility` spans all protein classes (including non-catalytic structural targets, viral proteins, and uncharacterized open reading frames) and contains only raw amino-acid sequences and binary solubility labels without UniProt accessions or Enzyme Commission (EC) annotations, Enzymado applies a strict two-stage enzyme curation filter:
1. **Exact Sequence Inner Join Against Reviewed Swiss-Prot Enzymes (`src/enzymado/propertyprep/proteinea.py:process_proteinea_solubility`):** Every sequence in `proteinea_solubility_raw.csv` (`71,421` rows) is matched by exact amino-acid identity against `data/raw/swissprot/swissprot_ec.tsv` (`279,501` reviewed UniProtKB/Swiss-Prot enzymes carrying explicit EC numbers). This inner join filters out `67,739` non-enzyme or unreviewed sequences and retains **3,682** verified Swiss-Prot enzymes annotated with canonical `uniprot_id` and `ec` numbers.
2. **UniProt Fragment Removal (`scripts/data_prep/remove_uniprot_fragments.py`):** Cross-referencing `uniprot_id` against UniProtKB fragment flags removes **17** incomplete sequence fragments (`13` from the upstream train split, `3` from validation, and `1` from test), producing **`3,665` complete enzyme sequences** in `data/raw/proteinea/proteinea_solubility.csv`.
3. **Standardized Benchmark Splits (`scripts/data_prep/propertyprep.py` & `dataprep.py:split_data`):** Maps `solubility -> label` and partitions all `3,665` enzymes into 80/10/10 `random_80_10_10` (`2,931` / `367` / `367`) and `similarity_mmseqs2_80_10_10` (`2,932` / `367` / `366`) split files.

### 1.3 Verbatim Supporting Quotes

> "We adopted the solubility database utilized in the development of DeepSol [30] with the same dataset splits, where any protein with ≥ 30% sequences identity to any protein in the testing subset is removed from the training and evaluation subsets."
> — **Elnaggar et al. (2023)**, `ankh_elnaggar2023_arxiv.pdf`, p. 10 (Section 5.4.4)

> "Solubility Prediction (SolP): This classification task evaluates the binary label of a set of dissimilar proteins as soluble or not. Solubility is an indispensable design parameter for effective proteins, especially for pharmaceuticals [29]."
> — **Elnaggar et al. (2023)**, `ankh_elnaggar2023_arxiv.pdf`, p. 10 (Section 4.3)

> "We used an initial training dataset consisting of 58,689 soluble and 70,954 insoluble protein sequences originally compiled in (Smialowski et al., 2012)."
> — **Khurana et al. (2018)**, `rev_solubility_deepsol_khurana2018_bioinf.pdf`, p. 3 (Section 2.1)

> "The final training data included 28,972 soluble and 40,448 insoluble proteins. The independent test set consisted of 1,000 soluble and 1,001 insoluble protein sequences first collected by (Chang et al., 2014)."
> — **Khurana et al. (2018)**, `rev_solubility_deepsol_khurana2018_bioinf.pdf`, p. 3 (Section 2.1)

> "Comparison of the mean prediction performance of DeepSol S1, S2 and S3 with that of PROSO II using 10-fold cross-validation on the training set comprising 69, 420 protein sequences."
> — **Khurana et al. (2018)**, `rev_solubility_deepsol_khurana2018_suppinfo.pdf`, p. 3 (Supplementary Table S2)

---

## 2. Measurement Protocol & Label Definition

### 2.1 Experimental Basis of Binary Solubility Labels
In PepcDB/TargetTrack and the Chang et al. (2014) validation set, recombinant proteins expressed heterologously (predominantly in *Escherichia coli*) are tracked through standardized structural-genomics pipeline stages (`cloned` $\to$ `expressed` $\to$ `soluble` $\to$ `purified` $\to$ `crystallized` $\to$ `structure`):
- **Soluble (`label = 1`):** Proteins confirmed to partition into the soluble supernatant following cell lysis and centrifugation (reaching the `soluble`, `purified`, `crystallized`, or `structure` milestones in TargetTrack, or curated as soluble enzymes in UniProtKB/Swiss-Prot by Smialowski et al. 2012).
- **Insoluble (`label = 0`):** Proteins that were successfully cloned and expressed in *E. coli* (or failed to advance past `cloned`/`expressed` milestones in TargetTrack) and formed insoluble inclusion bodies or failed to yield recoverable soluble protein under standard high-throughput screening conditions.

> "The heterologous expression of proteins in standard host cells, such as Escherichia coli, often render the proteins insoluble reducing their manufacturability (Chang et al., 2014)."
> — **Khurana et al. (2018)**, `rev_solubility_deepsol_khurana2018_bioinf.pdf`, p. 1 (Section 1 Introduction)

> "The length of all protein sequences in the training, validation and test set was fixed to L = 1, 200, which is the length of the longest protein sequence in the dataset."
> — **Khurana et al. (2018)**, `rev_solubility_deepsol_khurana2018_bioinf.pdf`, p. 5 (Section 2.2.1)

### 2.2 Assay Conditions & Missing Metadata
No per-entry experimental condition metadata are preserved in `Proteinea/solubility` or DeepSol. Differences in expression vector backbones, N/C-terminal affinity or solubility tags (e.g., $\text{His}_6$, GST, MBP, SUMO), induction temperature ($18\,^\circ\text{C}$ vs. $37\,^\circ\text{C}$), IPTG concentration, host strain genotype, and lysis buffer pH/ionic strength are collapsed into a single sequence-level binary label (`0` or `1`).

---

## 3. Data Cleaning, Filtering & Reconciliation

### 3.1 Exact Reconciliation of Paper, Raw HuggingFace, and Cleaned Enzymado Counts

| Pipeline Stage | Total Rows | Soluble (`1`) | Insoluble (`0`) | % Soluble | Notes |
|---|---:|---:|---:|---:|---|
| **PROSO II Initial Training Pool** (Smialowski et al. 2012) | `129,643` | `58,689` | `70,954` | `45.27%` | Raw PepcDB/TargetTrack + Swiss-Prot compilation |
| **DeepSol Filtered Train+Val Pool** (Khurana et al. 2018, p. 3) | `69,420` | `28,972` | `40,448` | `41.73%` | CD-HIT 90% internal clustering + `<30%` id to test + $L \le 1,200$ |
| **DeepSol Independent Test Set** (Chang et al. 2014) | `2,001` | `1,000` | `1,001` | `49.98%` | Independent *E. coli* expression dataset (printed as `1,999` in Table 12 of Elnaggar et al. 2023 due to a 2-row table typo) |
| **Raw `Proteinea/solubility`** (`proteinea_solubility_raw.csv`) | `71,421` | `29,972` | `41,449` | `41.97%` | `62,478` train + `6,942` val + `2,001` test on HuggingFace |
| **After Swiss-Prot Enzyme Match** (`proteinea.py`) | `3,682` | `2,063` | `1,619` | `56.03%` | Exact `protein_sequence` inner join with `swissprot_ec.tsv` (`67,739` non-enzymes/unreviewed dropped) |
| **Final Cleaned Enzymado Dataset** (`proteinea_solubility.csv`) | **`3,665`** | **`2,052`** | **`1,613`** | **`55.99%`** | `remove_uniprot_fragments.py` drops `17` UniProt fragments (`13` train, `3` val, `1` test) |

### 3.2 Class Balance Shift Upon Swiss-Prot Enzyme Filtering
Intersecting `Proteinea/solubility` with reviewed Swiss-Prot enzymes (`swissprot_ec.tsv`) flips the majority class from **insoluble (`58.03%` insoluble in raw DeepSol)** to **soluble (`55.99%` soluble in Enzymado)**. This `14.02`-percentage-point shift arises because reviewed UniProtKB/Swiss-Prot entries with experimentally assigned EC numbers are heavily enriched for well-behaved, soluble cytoplasmic and metabolic enzymes, whereas the uncharacterized structural-genomics ORFs removed during filtering (`67,739` sequences) had a high failure rate in *E. coli* high-throughput expression trials.

### 3.3 Known Noise Sources
1. **TargetTrack "Negative-by-Default" Censoring Noise:** Projects in PepcDB/TargetTrack that stopped at `cloned` or `expressed` were labeled insoluble (`0`) in PROSO II / DeepSol, even when termination occurred for administrative reasons (e.g., solved structure of an ortholog, funding reallocation, or construct redesign) rather than biophysical inclusion-body aggregation.
2. **Swiss-Prot Annotation Confounding:** In PROSO II (Smialowski et al. 2012), the positive class was augmented with proteins from Swiss-Prot carrying subcellular annotations. Models may partially exploit sequence signatures of well-annotated model-organism enzymes vs. structural-genomics targets.
3. **Host & Construct Heterogeneity:** Eukaryotic enzymes requiring disulfide bond formation, glycosylation, or specialized chaperones often misfold in standard *E. coli* cytoplasm (`label = 0`) despite being soluble in their native host.

---

## 4. Splits, Diversity & Leakage Audit

### 4.1 High Intrinsic Sequence Diversity (`ratio_unique_variants = 0.8641`)
Because Khurana et al. (2018) pre-clustered the DeepSol corpus with CD-HIT at `90%` sequence identity, `proteinea_solubility` exhibits much lower sequence redundancy than raw functional databases:
- **MMseqs2 50%-Identity Clustering (`--min-seq-id 0.5 -c 0.8 --cov-mode 0`):** Partitions the `3,665` enzyme sequences into **`3,167` clusters** (`ratio_unique_variants = 0.8641`), of which **`2,844` (`89.80%`) are singletons** and the largest cluster contains only `9` sequences (mean cluster size `1.157`).

### 4.2 Split Comparison (`random_80_10_10` vs. `similarity_mmseqs2_80_10_10`)

| Split Name | Train / Val / Test Rows | Train / Val / Test Positive Rate (`% Soluble`) | Exact Sequence / UniProt Overlap | Shared MMseqs2 50%-ID Clusters Across Splits |
|---|---|---|---|---|
| `random_80_10_10` | `2,931` / `367` / `367` | `55.96%` / `56.40%` / `55.86%` | `0.0` (`0` shared sequences) | **`246` clusters** (`7.77%` of `3,167` clusters span multiple splits) |
| **`similarity_mmseqs2_80_10_10`** *(Recommended)* | `2,932` / `367` / `366` | `55.97%` / `55.86%` / `56.13%` | `0.0` (`0` shared sequences) | **`0` clusters** (`2,533` train / `320` val / `314` test clusters) |

> "DeepSol outperformed all known sequence-based state-of-the-art solubility prediction methods and attained an accuracy of 0.77 and Matthew’s correlation coefficient of 0.55."
> — **Khurana et al. (2018)**, `rev_solubility_deepsol_khurana2018_bioinf.pdf`, p. 1 (Abstract)

### 4.3 Leakage Risks & Comparison Caveats
- **Re-splitting vs. Upstream DeepSol Test Set:** Of the `3,665` reviewed Swiss-Prot enzymes in Enzymado, `3,125` originated from the upstream `Proteinea/solubility` train split, `377` from validation, and `163` from the `2,001`-protein Chang et al. test split. Because `163` test enzymes would represent only `4.4%` of the dataset, Enzymado re-partitions all `3,665` enzymes into standardized 80/10/10 splits (`random_80_10_10` and `similarity_mmseqs2_80_10_10`). Consequently, metrics on `proteinea_solubility` should not be compared directly against whole-proteome `2,001`-sequence test scores reported in Khurana et al. (2018) or Elnaggar et al. (2023).
- **Pretraining Exposure:** Every sequence in `proteinea_solubility` is a reviewed UniProtKB/Swiss-Prot enzyme included in standard PLM pretraining corpora (UniRef50/UniRef90/UniRef100), although binary TargetTrack solubility labels are not part of unsupervised pretraining.

---

## 5. Intended Use & Limitations

- **Intended Use:** Evaluating protein language models and sequence embeddings on global cross-scaffold heterologous enzyme solubility classification (`proteinea_solubility`), and comparing model generalization across random (`random_80_10_10`) vs. homology-aware (`similarity_mmseqs2_80_10_10`) splits.
- **Out-of-Scope Use:**
  1. Predicting quantitative thermodynamic equilibrium solubility ($\text{mg}\cdot\text{mL}^{-1}$) or high-concentration aggregation kinetics of purified enzymes.
  2. Ranking single-point mutations ($\Delta\text{Sol}$) within a fixed wild-type enzyme scaffold (due to 90% CD-HIT deduplication and `89.8%` singleton clusters, local mutational series are absent; use mutational stability/expression datasets instead).
- **Key Limitations:** `94.87%` of raw `Proteinea/solubility` sequences are dropped by the reviewed Swiss-Prot enzyme filter (`71,421 -> 3,665`), shifting the soluble class rate from `41.97%` to `55.99%`; host strain, temperature, and fusion-tag metadata are unrecorded.

---

## 6. Provenance Gaps

| Gap Field | Description & Impact |
|---|---|
| `measurement_protocol.assay_conditions` | Per-protein expression conditions (host strain, induction temperature, expression vector, N/C-terminal solubility tag, lysis buffer pH/ionic strength) are absent in both `Proteinea/solubility` and DeepSol/PROSO II. |
| `provenance.upstream_databases[1].targettrack_id` | Original PepcDB/TargetTrack target identifiers were stripped in the upstream PROSO II / DeepSol sequence distributions; Enzymado recovers `uniprot_id` and `ec` annotations via exact amino-acid sequence matching against `swissprot_ec.tsv`. |

