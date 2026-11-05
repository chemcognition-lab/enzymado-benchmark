---
card_type: dataset
task_id: plpro_activity
dataset_id: plpro
property_name: plpro_activity
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
label_noise_floor: 0.0588 normalized fitness units (median scaled DiMSum sigma across 5,335 variants; analytical R^2_max = 0.9713)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/plpro_activity/dataset_card.json#/statistics
---

# `plpro_activity` — SARS-CoV-2 papain-like protease (PLpro) cellular proteolytic activity DMS

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/plpro_activity/dataset_card.json`](../../data/processed/real_case/plpro_activity/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is a dense single-site saturation landscape of the 315-aa SARS-CoV-2 PLpro domain assayed in human HEK293T cells.** Every one of the **5,335 rows** is a single canonical amino acid substitution (Hamming distance 1, $99.68\%$ sequence identity) of the SARS-CoV-2 Nsp3 papain-like protease domain (Nsp3 residues 746–1060, numbered 1–315), covering $89.1\%$ of all $315 \times 19 = 5,985$ theoretical missense mutations across the N-terminal ubiquitin-like (`Ubl2`) domain and the catalytic thumb, palm, and zinc-finger subdomains.
2. **Cellular biosensor cleavage conflates catalytic efficiency with intracellular protein folding/abundance—unless paired with `plpro_abundance`.** Because the assay measures 24-hour doxycycline-induced cleavage of a stably integrated `mClover3–TLKGGAPTKV–mRuby3` FRET biosensor inside live HEK293T cells, any mutation that destabilizes PLpro or triggers cellular degradation lowers the activity score just like a catalytic-triad knockout (`C111`, `H272`, `D286`). Crucially, [`scripts/data_prep/real_cases/plpro_dms.py`](../../scripts/data_prep/real_cases/plpro_dms.py) extracts both `plpro_activity` and `plpro_abundance` from the **exact same 5,335 variants** in worksheet `"Fig. 3"` of `plpro_flynn2024_MOESM9_ESM.xlsx`, enabling direct mechanistic deconvolution of folding defects vs. catalytic impairment.
3. **Internal 13-barcode replication across 7 independent transductions yields an analytical $R^2_{\max}$ ceiling of `0.9713`.** Each variant was linked to an average of 13 distinct 16-bp 3' UTR barcodes (mapped by PacBio Revio long-read sequencing) and assayed across **7 independent lentiviral transductions** (from 4 viral productions), with per-variant DiMSum error bars (`Errors_activity`, median $\sigma = 0.0588$) preserved in the raw Source Data table.
4. **Under the published point-stratified 80/10/10 split, every sequence position is seen $\sim 13.6$ times in `train`.** Because splitting is performed at the variant level (`random_split_unit="point"`), all 315 positions appear in the training set; the split therefore tests **within-position amino acid substitution interpolation** rather than unseen-site or unseen-scaffold extrapolation.

**Practical consequences for use:**
- Pair `plpro_activity` with `plpro_abundance` to separate active-site/allosteric functional mutants ($\text{abundance} > 0.75$ after subtracting $\sigma_{\text{abundance}}$ and $\text{activity} < 0.32$) from folding/stability mutants.
- Account for the bimodal Distribution of Fitness Effects (DFE) peaking near `0.0` (nonsense-like dead/destabilized variants) and `1.0` (synonymous WT-like variants that completely cleave the biosensor within 24 h).
- Note on PDF naming: the files `plpro_flynn2024_natcommun.pdf` and `plpro_flynn2024_suppinfo.pdf` in `docs/pdfs/` are the Wu, Go, Nguyen, ..., Call & Call (*Nature Communications* 15, 6219, 2024) SARS-CoV-2 PLpro paper (named by analogy to the Flynn et al. 2022 Mpro DMS paper).
- Never pass `dataset_id` (`plpro_activity_00001` … `plpro_activity_05335`, which is sorted monotonically by residue position 1 to 315) as a model feature.

---

## 1. Label noise and the measurement floor

Unlike multi-lab database aggregates or single-replicate screens, `plpro_activity` has an **explicitly quantified per-variant empirical-Bayes error model (DiMSum)** derived from 7 independent biological transduction replicates and ~13 internal barcodes per amino acid variant.

### 1.1 Experimental quality controls and replicate structure

| Property | Value | Evidence |
|---|---|---|
| **Biological replicates** | **$n = 7$ independent lentiviral transductions** from 4 viral productions (~2 million transductants per replicate, $\sim 10\times$ library coverage) | *"In all, we did seven independent transductions from 4 independent viral productions from a single library plasmid stock."* `[Source: Main, Page 3]` |
| **Internal barcode replication** | **Mean $13$ unique 16-bp barcodes per variant** (102,363 total barcodes mapped by PacBio Revio HiFi; WT aligned to 6,657 barcodes) | *"We found 6263 of the 6300 protein variants were present in the library representing 99.4% coverage and each variant was replicated with an average of 13 barcodes, except wildtype which aligned to 6657 barcodes."* `[Source: Main, Page 3]` |
| **Technical PCR indexing** | Triplicate indexed PCR per sorted population; >30 average NextSeq 2000 reads per barcode | *"Technical replicates were used to combine samples with triplicate indexes after checking for concordance."* `[Source: Main, Page 13]` |
| **Upstream read & error filter** | Excluded variants with $\le 10$ reads at $t_0$ or scaled DiMSum error $\sigma \ge 0.20$ | *"After normalization, data was filtered to remove variants with no more than 10 reads at timepoint 0 and sigma no less than 0.2 (Supplementary Fig. 3)."* `[Source: Main, Page 13]` |
| **Per-variant error distribution (`Errors_activity`)** | Median $\sigma = 0.0588$, mean $\sigma = 0.0628$, IQR $0.0471\text{–}0.0756$, p95 $= 0.1059$, max $= 0.1977$ (`0.1490` standardized SD units) | Computed from column `Errors_activity` across the 5,335 missense rows in `"Fig. 3"` of `plpro_flynn2024_MOESM9_ESM.xlsx` |
| **Analytical $R^2_{\max}$ ceiling** | **$R^2_{\max} = 1 - \overline{\sigma_i^2} / \text{Var}(y) = 1 - 0.004407 / 0.153602 = \mathbf{0.9713}$** | Direct variance decomposition on `"Fig. 3"` (`fitness_activity` variance $= 0.1536$) |

### 1.2 Factors governing the error structure

1. **24-hour cellular steady-state saturation at the top of the dynamic range:** PLpro is induced with $300\text{ ng/mL}$ doxycycline for 24 h prior to FACS sorting. Variants with near-WT or moderately reduced catalytic efficiency still cleave nearly all intracellular biosensor molecules over 24 h, compressing their scores toward `1.0` (verified against in vitro initial-rate $k_{\text{cat}}/K_M$ measurements on recombinant purified variants in Supplementary Fig. 15).
2. **Activity–abundance coupling in live cells:** *"Impaired variants may carry a mutation that impacts abundance, most likely via folding or stability defects, or function, giving mechanistic insight."* `[Source: Main, Page 5]`. In Fig. 3c, the majority of low-activity variants also show low abundance in `plpro_abundance`.
3. **Donor fluorophore instability upon cleavage:** Cleavage of the linker leaves a destabilizing `TLKGG` tail on `mClover3`: *"The reduction of mClover3 fluorescence indicates that free mClover3 fused to a fragment of the PLpro cleavage motif (TP5LP4KP3GP2GP1) is less stable than the intact biosensor. We therefore present data as FRET versus acceptor fluorescence."* `[Source: Main, Page 3]`.
4. **Anti-swapping lentiviral packaging control:** To prevent pseudodiploid lentiviral recombination from uncoupling 3' barcodes from coding mutations, the library plasmid was co-packaged 1:2 with an unrelated `pFGH1-UTG-mTagBFP2` vector and transduced at $\text{MOI} \approx 0.2$ (Main pp. 3, 12).

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **5,335 single-site missense variants** of a single 315-aa protein domain: the **SARS-CoV-2 papain-like protease (PLpro)** domain of Non-structural protein 3 (`Nsp3` residues 746–1060, numbered 1–315; UniProt `P0DTD1`, **EC 3.4.22.69**, peptidase family C16)—*"We chose to examine 315 residues of Nsp3 spanning residues 746 to 1060, which encompasses the PLpro core and the ubiquitin-like domain N-terminal to PLpro."* `[Source: Main, Page 2]`.
- **Structural subdomains & functional motifs covered:**
  - **N-terminal Ubiquitin-like (`Ubl2`) domain (residues 1–60):** Highly tolerant to mutation (`fitness_activity` near 1.0).
  - **Thumb, Palm, and Fingers catalytic core (residues 61–305):** Strongly constrained in the hydrophobic core, at the **catalytic triad (`Cys111`, `His272`, `Asp286`)**, at the **tetrahedral zinc-coordinating motif (`Cys189`, `Cys192`, `Cys224`, `Cys226`)** where only `C224H` is tolerated, and along the S1–S4 ubiquitin/ISG15 binding pocket and blocking loop (`BL2`, residues 265–271).
  - **C-terminal tail after `Tyr305` (residues 306–315):** Tolerant even to premature stop codons (`*`), which is why nonsense normalization in DiMSum was restricted to positions 1–304.
- **Reference crystal structures (not task inputs):** SARS-CoV-2 PLpro:ubiquitin (`6XAA`), unliganded PLpro (`8FWN`), and the $2.00\,\text{\AA}$ structure of drug-escape variant `M208W` solved in the study (`8VEC`).
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** Every row assays cleavage of the same genetically encoded polypeptide substrate (`mClover3–TLKGGAPTKV–mRuby3`, containing the native viral `Nsp2/3` cleavage sequence `TLKGG↓APTKV`) inside untreated (+dox, no inhibitor) HEK293T cells. No small-molecule SMILES or numeric condition columns are present.

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Cellular FACS sort-seq of an `mClover3–Nsp2/3–mRuby3` FRET biosensor in human HEK293T cells, sorting `FRET+` (inactive) vs. `FRET-` (active) gates followed by Illumina NextSeq 2000 barcode sequencing | No | `reported` | *"Our FRET-based biosensor is composed of N-terminal mClover3 donor and C-terminal mRuby3 acceptor fluorophores separated by a linker containing the Nsp2/3 PLpro cleavage motif TP5LP4KP3GP2GP1 ↓ AP-1PP-2TP-3KP-4VP-5 (Fig. 1b)."* `[Source: Main, Page 3]` |
| **Analytical readout** | FACS excitation at $488\text{ nm}$, FRET emission at $610\text{ nm}$ vs. `mRuby3` acceptor fluorescence; DiMSum ratiometric fitness calibrated to nonsense (1–304) $= 0$ and synonymous WT $= 1$ | No | `reported` | *"A FRET signal between mClover3 and mRuby3 was readily observable with excitation at 488 nm and emission at 610 nm (Supplementary Fig. 1a)."* `[Source: Main, Page 3]` |
| **Expression system** | Intracellular expression in human HEK293T cells from lentiviral `FU-tetO-Gateway-rtTA-2A-Puro` ($\text{MOI} \approx 0.2$, $2\text{ ng/}\mu\text{L}$ puromycin selection), induced with $300\text{ ng/mL}$ doxycycline | No | `reported` | *"Virus containing the PLpro library was then transduced into HEK293T biosensor cells, or in the case of the mClover3-containing abundance library, parental HEK293T cells, ensuring a 10-fold replication of barcodes at an MOI of approximately 0.2."* `[Source: Main, Page 12]` |
| **Expression decoupled from label** | **false** (cellular readout reflects abundance $\times$ specific activity; paired with `plpro_abundance` for deconvolution) | — | `reported` | *"To identify abundant but functionally impaired variants, we reformatted the library to install a C-terminal mClover3 fusion"* `[Source: Main, Page 5]` |
| **Temperature ($T$)** | **$37\,^\circ\text{C}$** (humidified $5\%\text{ CO}_2$ mammalian incubator) | No (constant) | `reported` | *"All cells were cultured at 37 °C in a humidified incubator with 5% CO2."* `[Source: Main, Page 11]` |
| **pH & buffer** | Physiological HEK293T cytosolic pH ($\sim 7.2\text{–}7.4$) in DMEM + $10\%$ FBS; sorted in PBS + $2\%$ FBS | No (constant) | `reported` | *"DMEM supplemented with 10% Fetal Bovine Serum (FBS; Sigma-Aldrich, #F9423) was used to culture the 293T cells and derivatives."* `[Source: Main, Page 11]` |
| **Cofactors & metals** | Endogenous cellular $\text{Zn}^{2+}$ coordinated by `C189`, `C192`, `C224`, `C226` in the fingers subdomain | No (constant) | `reported` | *"Additionally, the four cysteines (Cys189, Cys192, Cys224, Cys226) that tetrahedrally coordinate the zinc in the fingers domain3 were absolutely required except for Cys224, where another zinc coordinating residue (histidine) was tolerated."* `[Source: Main, Page 3]` |
| **Readout window** | **24 h** doxycycline induction ($300\text{ ng/mL}$) prior to FACS sorting | No (constant) | `reported` | *"For the activity screen, we treated cells with 300 ng/ml dox for 24 h and sorted FRET positive from FRET negative cells for mRNA extraction (Fig. 1g)."* `[Source: Main, Page 13]` |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Upstream library construction and DiMSum scoring in Wu et al. (2024)

1. **Hybrid Twist dsDNA + MITE library construction:** Active-site-proximal residues (`62, 69–70, 73–74, 77, 93, 104, 106–119, 151–174, 206–212, 243–253, 260–276, 285–286, 296–304`) were cloned from Twist dsDNA fragments and the remaining positions via 7 MITE `NNK` oligo cohorts, appending a 16-bp barcode 3' to the stop codon (Main p. 12).
2. **PacBio Revio barcode lookup table:** ~4 million HiFi reads mapped 102,363 unique barcodes across 6,263 of 6,300 theoretical single-site variants ($99.4\%$ coverage, mean 13 barcodes/variant; Main pp. 3, 12).
3. **DiMSum normalization and quality filtering:** Treating the sorted `FRET+` (inactive) gate as $t_0$ and the `FRET-` (active) gate as $t_1$, raw DiMSum fitness scores $F_{\text{raw}}$ and errors $\sigma_{\text{raw}}$ across 7 independent transductions were linearly scaled (Main p. 13, Eqs. 4–6):

$$F_{\text{scaled}} = \frac{F_{\text{raw}} - F_{\text{avg-nonsense(res 1–304)}}}{F_{\text{WT}} - F_{\text{avg-nonsense(res 1–304)}}}, \qquad \sigma_{\text{scaled}} = \frac{\sigma_{\text{raw}}}{F_{\text{WT}} - F_{\text{avg-nonsense(res 1–304)}}}$$

Variants with $\le 10$ reads at $t_0$ or $\sigma_{\text{scaled}} \ge 0.20$ were excluded.

### 4.2 Source-to-benchmark reconciliation in [`plpro_dms.py`](../../scripts/data_prep/real_cases/plpro_dms.py)

Worksheet `"Fig. 3"` of `plpro_flynn2024_MOESM9_ESM.xlsx` places three tables side-by-side:
- **Columns A–G (`#this is for all data`, 5,342 rows):** Contains all non-synonymous variants passing quality filters in **both** the activity screen ($\sigma < 0.20$) and the abundance screen ($\sigma < 0.45$). These 5,342 rows consist of **5,335 single-site missense variants** (`Mut` in the 20 canonical amino acids) plus **7 C-terminal nonsense (`*`) variants** after `Tyr305` (at positions `306, 307, 308, 309, 311, 312, 314`, which retain activity because the extreme C-terminal tail is dispensable).
- **Columns I–M (`#this is for synonymous data`, 340 rows)** and **Columns O–U (`#this is for nonsense variants`, 285 rows at positions 1–305)**.

[`scripts/data_prep/real_cases/plpro_dms.py`](../../scripts/data_prep/real_cases/plpro_dms.py#L72-L99) reads Columns A–E and filters `mut_aa in CANONICAL_AA`, dropping the 7 C-terminal `*` rows and retaining all **5,335 unique missense variants** with zero duplicates. Because `fitness_activity` is already a calibrated log-enrichment ratio ($[-0.2799, +1.2867]$), Enzymado applies an **Identity** transformation followed by train-split z-score standardization ($\mu_{\text{train}} = 0.5999$, $\sigma_{\text{train}} = 0.3947$).

---

## 5. Split semantics and leakage caveats

One split file is published: `plpro_activity_random_80_10_10_standardized` ($4,270\text{ train} / 533\text{ val} / 532\text{ test} = 5,335\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/plpro_activity/dataset_card.json) for exact split moments).

1. **Position-level sharing across splits (`random_split_unit="point"`):** Every one of the 315 sequence positions is mutated $\sim 13.6$ times in `train` and $\sim 1.7$ times in `val` and `test`. High pooled $R^2$ on this split reflects accurate interpolation across amino acid substitutions at known positions rather than de novo structural generalization to unseen positions.
2. **Identical variant set with `plpro_abundance`:** All 5,335 variants in `plpro_activity` also appear in `plpro_abundance`. Never leak `plpro_abundance` targets when benchmarking `plpro_activity`.
3. **Monotonic position ordering of `dataset_id`:** `plpro_activity_00001` through `plpro_activity_05335` are ordered monotonically by residue position (`Pos` 1 to 315); `dataset_id` is flagged in `categorical_high_variance_columns` and must never be used as a feature.
