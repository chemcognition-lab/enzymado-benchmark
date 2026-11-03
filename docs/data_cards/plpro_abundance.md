---
card_type: dataset
task_id: plpro_abundance
dataset_id: plpro
property_name: plpro_abundance
track: real_case
task_type: protein_global
label_kind: scalar_regression
layout: Layout A
doi: 10.1038/s41467-024-50566-9
license: CC BY 4.0
source_pdfs:
  - docs/pdfs/plpro_flynn2024_natcommun.pdf (main, pp. 1-18)
  - docs/pdfs/plpro_flynn2024_suppinfo.pdf (si, pp. 1-29)
noise_regime: single_lab_replicated
label_noise_floor: 0.1195 normalized fitness units (median scaled DiMSum sigma across 5,335 variants; analytical R^2_max = 0.8341)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/plpro_abundance/dataset_card.json#/statistics
---

# `plpro_abundance` — SARS-CoV-2 papain-like protease (PLpro) intracellular abundance DMS

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/plpro_abundance/dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/plpro_abundance/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is a VAMP-seq-style intracellular steady-state protein abundance landscape across 5,335 single-site missense variants of SARS-CoV-2 PLpro.** The 315-residue SARS-CoV-2 Nsp3 PLpro domain (Nsp3 residues 746–1060, numbered 1–315) was fused C-terminally to `mClover3` via a 14-residue flexible linker (`PVGGSGGGGSGGGG`), expressed in parental human HEK293T cells, and FACS-sorted by direct `mClover3` fluorescence into `mClover3-high` (folded/abundant) and `mClover3-low` (misfolded/degraded) populations.
2. **It is the orthogonal folding/stability counterpart to `plpro_activity` on the exact same 5,335 variants.** Both tasks are ingested by [`scripts/data_prep/real_cases/plpro_dms.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/plpro_dms.py) from the same `"Fig. 3"` worksheet of `plpro_flynn2024_MOESM9_ESM.xlsx`. Whereas `plpro_activity` drops whenever either catalysis or folding fails, `plpro_abundance` drops only when folding/cellular accumulation fails—so catalytically dead active-site mutants (e.g., `C111`, `H272`, `D286`, `W106`, `Y268`, `Q269`) retain wild-type-like abundance near `1.0`.
3. **Its 2-replicate transduction design sets an analytical $R^2_{\max}$ ceiling of `0.8341` (vs. `0.9713` for `plpro_activity`).** Whereas `plpro_activity` combined 7 independent lentiviral transductions (median DiMSum $\sigma = 0.0588$), `plpro_abundance` combined **2 independent transductions from 1 viral stock** (with an average of $16.8$ barcodes per variant mapped by Oxford Nanopore R10.4.1 sequencing), yielding a $2.03\times$ larger median DiMSum error ($\sigma = 0.1195$, or $0.3245$ standardized SD units).
4. **Under the point-stratified 80/10/10 split, every sequence position is sampled $\sim 13.6$ times in `train`.** Because splitting is performed at the variant level (`random_split_unit="point"`), all 315 positions appear in `train`; the split therefore evaluates **within-position substitution interpolation** for cellular protein abundance rather than unseen-site or unseen-scaffold extrapolation.

**Practical consequences for use:**
- Always normalize model $R^2$ expectations against the analytical noise ceiling **$R^2_{\max} = 0.8341$**: a model achieving $R^2 = 0.70$ on `plpro_abundance` accounts for $\sim 84\%$ of explainable variance.
- Expect zero-shot evolutionary/MSA log-likelihoods to produce systematic **false negatives at the catalytic active site**: active-site residues (`Cys111`, `His272`, `Asp286`, `Trp106`, blocking loop `Tyr268`/`Gln269`) are strictly conserved in nature for catalysis (`plpro_activity` $\approx 0$), yet tolerate substitutions with high cellular abundance (`plpro_abundance` $> 0.75$).
- Note that while the experimental construct carries the C-terminal `-PVGGSGGGGSGGGG-mClover3` tag, `protein_sequence` in Enzymado stores the 315-aa untagged PLpro sequence so that embeddings are shared 1:1 with `plpro_activity`.
- Never pass `dataset_id` (`plpro_abundance_00001` … `plpro_abundance_05335`) as a model input.

---

## 1. Label noise and the measurement floor

Per-variant DiMSum error bars (`Errors_abundance`) are preserved in worksheet `"Fig. 3"` of `plpro_flynn2024_MOESM9_ESM.xlsx`, allowing exact variance decomposition of the 2-replicate FACS abundance screen.

### 1.1 Experimental quality controls and replicate structure

| Property | Value | Evidence |
|---|---|---|
| **Biological replicates** | **$n = 2$ independent lentiviral transductions** from 1 viral stock in parental HEK293T cells ($\text{MOI} \approx 0.2$, $\sim 10\times$ barcode representation) | *"We expressed this library in parental 293T cells (two independent transductions from one viral stock) and sorted cells from high and low mClover3 gates, recovered mRNA to count variants, and calculated ratiometric scores to determine the abundance of each variant."* `[Source: Main, Page 5]` |
| **Internal barcode replication** | **Mean $16.8$ unique 16-bp barcodes per single-site variant** (105,942 single-site barcodes out of 146,972 total barcodes mapped by Nanopore R10.4.1 + Illumina whitelist) | *"We identified 146,972 unique barcodes. Among them, 105,942 barcodes aligned to a single site substituted PLpro variant, leading to an average of 16.8 barcodes per variant."* `[Source: Main, Page 12]` |
| **Upstream read & error filter** | Excluded variants with $\le 10$ reads at $t_0$ (`mClover3-low`) or scaled DiMSum error $\sigma \ge 0.45$ | *"After normalization, data was filtered to remove variants with no more than 10 reads at time-point 0 and sigma no less than 0.45 (Supplementary Fig. 4)."* `[Source: Main, Page 13]` |
| **Per-variant error distribution (`Errors_abundance`)** | Median $\sigma = 0.1195$, mean $\sigma = 0.1363$, IQR $0.0972\text{–}0.1597$, p95 $= 0.2671$, max $= 0.4494$ (`0.3245` standardized SD units) | Computed from column `Errors_abundance` across the 5,335 missense rows in `"Fig. 3"` of `plpro_flynn2024_MOESM9_ESM.xlsx` |
| **Analytical $R^2_{\max}$ ceiling** | **$R^2_{\max} = 1 - \overline{\sigma_i^2} / \text{Var}(y) = 1 - 0.022559 / 0.136019 = \mathbf{0.8341}$** | Direct variance decomposition on `"Fig. 3"` (`fitness_abundance` variance $= 0.1360$) |

### 1.2 Factors governing the error structure

1. **Two transduction replicates ($n = 2$) vs. seven ($n = 7$) in `plpro_activity`:** Because `plpro_abundance` was screened in 2 transduction replicates rather than 7, its mean squared measurement error ($\overline{\sigma^2} = 0.0226$) is $5.1\times$ larger than that of `plpro_activity` ($\overline{\sigma^2} = 0.0044$), lowering the analytical $R^2_{\max}$ ceiling from `0.9713` to `0.8341`.
2. **Binary two-gate FACS sorting (`mClover3-high` vs. `mClover3-low`):** Rather than sorting into 4–8 fluorescence bins, cells were sorted into two gates (Supplementary Fig. 4a), which increases binomial partitioning variance for partially destabilized variants sitting near the gate boundary.
3. **Subcloning boundary at `L290`:** Positions 1–290 were transferred directly from the barcoded `pDONR221-PLpro` library via `PpuMI`/`EcoRV` digestion, while C-terminal positions `L290–K315` were generated via an additional round of MITE mutagenesis on `pDONR221-PLpro-linker-mClover3` (Main p. 12; Nanopore coverage across both regions verified in Supplementary Fig. 19).

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **5,335 single-site missense variants** of the 315-aa **SARS-CoV-2 Nsp3 PLpro domain** (residues 746–1060, numbered 1–315; UniProt `P0DTD1`, **EC 3.4.22.69**).
- **Biophysical determinants of `plpro_abundance` across subdomains:**
  - **Zinc-coordinating fingers motif (`Cys189`, `Cys192`, `Cys224`, `Cys226`):** Tetrahedral $\text{Zn}^{2+}$ coordination is essential for folding; mutations at these four cysteines (except `C224H`, which forms a functional $\text{C}_2\text{CH}$ zinc finger) ablate abundance (`fitness_abundance` $\approx 0$).
  - **Hydrophobic core of the thumb and palm domains:** Intolerant of polar/charged or proline substitutions.
  - **Catalytic triad (`Cys111`, `His272`, `Asp286`) & substrate-binding pocket (`Trp106`, `Tyr268`, `Gln269`):** Structurally tolerant—variants at 20 functional positions maintain high abundance ($\text{abundance} - \sigma_{\text{abundance}} > 0.75$) while losing proteolytic activity ($\text{activity} < 0.32$; Fig. 3c–d, Table 1).
- **Construct representation (`315` aa):** Experimentally expressed as `PLpro(1–315)–PVGGSGGGGSGGGG–mClover3`; represented in Enzymado as the 315-aa untagged PLpro sequence (100% canonical amino acids, 1 MMseqs2 cluster at 50% identity).
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** Assayed in parental HEK293T cells without FRET biosensor or small-molecule inhibitors.

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Cellular protein-abundance DMS (FACS sort-seq) of `PLpro–PVGGSGGGGSGGGG–mClover3` in parental HEK293T cells, sorting `mClover3-high` (folded) vs. `mClover3-low` (misfolded) gates followed by Illumina NextSeq 2000 barcode sequencing | No | `reported` | *"To identify abundant but functionally impaired variants, we reformatted the library to install a C-terminal mClover3 fusion (Supplementary Fig. 4), allowing PLpro abundance to be monitored by flow cytometry."* `[Source: Main, Page 5]` |
| **Analytical readout** | Direct `mClover3` FACS fluorescence ($488\text{ nm}$ excitation, $530/30\text{ nm}$ emission); DiMSum ratiometric fitness calibrated to nonsense (1–304) $= 0$ and synonymous WT $= 1$ | No | `reported` | *"Samples from the mClover3 low gate (misfolded PLpro) were treated as time-point 0, while those from the mClover3 high gate (folded PLpro) were treated as timepoint 1."* `[Source: Main, Page 13]` |
| **Expression system** | Intracellular expression in parental human HEK293T cells from lentiviral `FU-tetO-Gateway-rtTA-2A-Puro` ($\text{MOI} \approx 0.2$, $2\text{ ng/}\mu\text{L}$ puromycin selection), induced with $300\text{ ng/mL}$ doxycycline | No | `reported` | *"Virus containing the PLpro library was then transduced into HEK293T biosensor cells, or in the case of the mClover3-containing abundance library, parental HEK293T cells, ensuring a 10-fold replication of barcodes at an MOI of approximately 0.2."* `[Source: Main, Page 12]` |
| **Expression decoupled from label** | **false** (cellular steady-state abundance is the measured target observable) | — | `reported` | *"Positive fitness scores indicate expression and negative fitness scores indicate impaired expression."* `[Source: Main, Page 13]` |
| **Temperature ($T$)** | **$37\,^\circ\text{C}$** (humidified $5\%\text{ CO}_2$ mammalian incubator) | No (constant) | `reported` | *"All cells were cultured at 37 °C in a humidified incubator with 5% CO2."* `[Source: Main, Page 11]` |
| **pH & buffer** | Physiological HEK293T cytosolic pH ($\sim 7.2\text{–}7.4$) in DMEM + $10\%$ FBS; sorted in PBS + $2\%$ FBS | No (constant) | `reported` | *"DMEM supplemented with 10% Fetal Bovine Serum (FBS; Sigma-Aldrich, #F9423) was used to culture the 293T cells and derivatives."* `[Source: Main, Page 11]` |
| **Cofactors & metals** | Endogenous cellular $\text{Zn}^{2+}$ coordinated by `C189`, `C192`, `C224`, `C226` in the fingers domain | No (constant) | `reported` | Main p. 3 |
| **Readout window** | **24 h** doxycycline induction ($300\text{ ng/mL}$) prior to FACS sorting | No (constant) | `reported` | *"For the abundance screen we similarly treated with dox but sorted on mClover3 high and mClover3 low cells (Supplementary Fig. 4)."* `[Source: Main, Page 13]` |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Upstream library construction and DiMSum scoring in Wu et al. (2024)

1. **`PLpro–mClover3` library assembly:** *"We designed a PLpro-mClover3 fusion separated by the linker PVGGSGGGGSGGGG and ordered this as an IDT G-block and cloned it into pDONR221 with BP clonase (Invitrogen, #11789020) to be the basis of our new abundance library."* `[Source: Main, Page 12]`. Positions 1–290 were transferred from the barcoded PLpro library (`PpuMI`/`EcoRV`), and positions `L290–K315` were recreated via MITE mutagenesis.
2. **Oxford Nanopore R10.4.1 + Illumina whitelist barcode mapping:** ~26 million Nanopore reads mapped 146,972 unique barcodes (105,942 single-site variant barcodes, mean $16.8$ barcodes/variant) across 6,130 of 6,301 potential single-site variants ($\sim 97\%$ coverage; Main pp. 5, 12).
3. **DiMSum normalization and quality filtering:** Treating `mClover3-low` as $t_0$ and `mClover3-high` as $t_1$, raw DiMSum scores across 2 independent transductions were linearly scaled so synonymous WT $= 1.0$ and nonsense (positions 1–304, which truncate upstream of `mClover3`) $= 0.0$ (Eqs. 4–6, Main p. 13), and variants with $\le 10$ reads at $t_0$ or $\sigma_{\text{scaled}} \ge 0.45$ were excluded.

### 4.2 Source-to-benchmark reconciliation in [`plpro_dms.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/plpro_dms.py)

As documented in `plpro_activity`, worksheet `"Fig. 3"` of `plpro_flynn2024_MOESM9_ESM.xlsx` (Columns A–G, `#this is for all data`) stores the **5,342 non-synonymous variants** passing quality filters in both the activity and abundance screens (`5,335` missense + `7` C-terminal `*` variants at positions `306–314`). Filtering `mut_aa in CANONICAL_AA` in [`plpro_dms.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/plpro_dms.py#L72-L108) drops the 7 `*` rows and writes **5,335 unique missense rows** with target column `fitness_abundance` (range $[-0.9023, +1.6296]$, Identity transform, standardized on the 4,270 training rows with $\mu_{\text{train}} = 0.6143$, $\sigma_{\text{train}} = 0.3682$).

---

## 5. Split semantics and leakage caveats

One split file is published: `plpro_abundance_random_80_10_10_standardized` ($4,270\text{ train} / 533\text{ val} / 532\text{ test} = 5,335\text{ total}$; see [`dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/plpro_abundance/dataset_card.json) for exact split moments).

1. **Position-level sharing across splits (`random_split_unit="point"`):** All 315 residue positions are mutated $\sim 13.6$ times in `train` and $\sim 1.7$ times in `val` and `test`; the split evaluates within-position substitution interpolation rather than unseen-position generalization.
2. **Independent decile stratification vs. `plpro_activity`:** Although `plpro_abundance` and `plpro_activity` contain the exact same 5,335 variants, `generate_real_case_splits.py` stratifies each task independently on its own target column (`fitness_abundance` vs. `fitness_activity`). Consequently, their train/val/test row partitions are not identical—never transfer features or out-of-fold predictions between the two tasks without re-aligning split masks.
3. **Identifier leakage risk:** `dataset_id` (`plpro_abundance_00001` … `plpro_abundance_05335`) is ordered monotonically by sequence position (1 to 315) and must never be passed as a model feature.
