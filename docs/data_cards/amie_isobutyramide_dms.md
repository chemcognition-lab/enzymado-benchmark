---
card_type: dataset
task_id: amie_isobutyramide_dms
dataset_id: amie_isobutyramide
property_name: amie_isobutyramide_dms
track: real_case
task_type: protein_global
label_kind: scalar_regression
layout: Layout A
doi: 10.1038/ncomms15695
license: CC BY 4.0
source_pdfs:
  - docs/pdfs/amie_wrenbeck2017_natcommun.pdf (main, pp. 1-10)
  - docs/pdfs/amie_wrenbeck2017_suppinfo.pdf (si, pp. 1-21)
  - docs/pdfs/amie_cavity_faber2019_biorxiv.pdf (review, pp. 1-27)
noise_regime: single_lab_replicated
label_noise_floor: replicate Pearson r = 0.943 above lower bound zeta = -0.6 (global Pearson r = 0.889, Spearman rho = 0.9484, median |rep1 - rep2| = 0.0499; synonymous WT SD = 0.07)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/amie_isobutyramide_dms/dataset_card.json#/statistics
---

# `amie_isobutyramide_dms` — *Pseudomonas aeruginosa* AmiE deep mutational scan on isobutyramide

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/amie_isobutyramide_dms/dataset_card.json`](../../data/processed/real_case/amie_isobutyramide_dms/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is the sterically bulky, non-native substrate arm ($10\text{ mM}$ isobutyramide, $\alpha$-branched C4 amide) of the matched 3-substrate DMS of *Pseudomonas aeruginosa* `AmiE` (346 aa).** Paired with `amie_acetamide_dms` ($10\text{ mM}$ acetamide, C2 linear) and `amie_propionamide_dms` ($15\text{ mM}$ propionamide, C3 linear), it profiles **6,229 unique sequences** (1 wild-type + 6,228 single-site missense variants across residues 1–341) under competitive *E. coli* nitrogen-source growth selection.
2. **Unlike the native acetamide landscape, `amie_isobutyramide_dms` is an adaptive gain-of-function landscape ($21.5\%$ of missense variants beat wild-type).** Because wild-type `AmiE` has poor catalytic efficiency on bulky isobutyramide ($K_M = 297.2 \pm 54.8\text{ mM}$, $k_{\text{cat}} = 13.3 \pm 1.1\text{ s}^{-1}$, $k_{\text{cat}}/K_M = 45\text{ M}^{-1}\text{s}^{-1}$—$280\times$ lower than on acetamide), the selection required a higher-expression plasmid (`pEDA6_amiE`: stronger promoter `proK14` + T7 bacteriophage ribosome binding site `t7RBS`) so that wild-type *E. coli* MG1655 $rph^+$ grew at $\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.54 \pm 0.11$ at $10\text{ mM}$ isobutyramide ($\sim 0.034\times K_M$). Under this regime, **1,394 missense mutations ($21.5\%$) exceed wild-type ($\zeta > 0$)** and **483 ($7.5\%$) confer $\ge 10\%$ faster growth ($\zeta > 0.15$)**—more than $4\times$ the beneficial mutation rate observed on acetamide ($4.7\%$) or propionamide ($5.1\%$).
3. **It has the cleanest, most symmetric target distribution and highest global replicate correlation (`r = 0.889`) of the three AmiE tasks.** Because 4,977 of the 6,229 variants sit above the sequencing-depth lower bound ($\zeta = -0.6$) and there are zero extreme Poisson boundary outliers (raw range $[-2.2788, +0.8438]$, skewness $= -0.056$), replicate Pearson correlation is **`r = 0.943`** above the lower bound and **`r = 0.889`** globally (Spearman $\rho = 0.9484$, median $|\text{rep}_1 - \text{rep}_2| = 0.0499\ \zeta\text{ units}$).
4. **Under the point-stratified 80/10/10 split, every scanned position (1–341) appears $\sim 14.6$ times in `train`.** Splitting at the observation level (`random_split_unit="point"`) tests **within-position amino acid substitution interpolation** rather than unseen-site or unseen-scaffold extrapolation.

**Practical consequences for use:**
- Use `amie_isobutyramide_dms` to test whether models can predict **non-native specificity-remodeling mutations** distributed $9\text{–}24\,\text{\AA}$ from the catalytic nucleophile `Cys166` (where zero-shot evolutionary conservation models underperform because enlarging the acyl pocket often mutates conserved second-shell residues such as `W138`, `M202`, `M203`, or `Q273`).
- Never leak labels or split masks across `amie_acetamide_dms`, `amie_propionamide_dms`, and `amie_isobutyramide_dms`.
- Never pass `dataset_id` (`amie_isobutyramide_00001` … `amie_isobutyramide_06229`) as a model feature.

---

## 1. Label noise and the measurement floor

### 1.1 Experimental quality controls and replicate structure

| Property | Value | Evidence |
|---|---|---|
| **Biological replicates** | **$n = 2$ replicate 8-generation selections** from a shared pre-selection library; published label combines reads across both replicates | *"Based on the high correlation between replicates and the fact that a major source of error in deep sequencing measurements are counting errors (Poisson noise)34, the fitness metrics used in subsequent analysis were computed by combining reads from the two replicates and repeating the analysis."* `[Source: Main, Page 8]` |
| **Replicate correlation above lower bound ($\zeta \ge -0.6$)** | **Pearson $r = 0.943$** ($n = 4,977$, $P < 2.2 \times 10^{-308}$; Supplementary Fig. 6b, SI p. 12) | *"When we excluded variants with fitness metrics below the lower bounds the correlation coefficients improved to 0.932, 0.949 and 0.943 for the ACT, PR and IB selections, respectively"* `[Source: Main, Page 3]` |
| **Global replicate correlation (all variants)** | **Pearson $r = 0.889$** ($n = 6,569$); **Spearman $\rho = 0.9484$**, median $|\text{rep}_1 - \text{rep}_2| = 0.0499\ \zeta\text{ units}$ across 6,024 finite missense pairs | *"we calculated the Pearson product moment correlation coefficients of fitness metrics between replicates and found them to be 0.661, 0.842 and 0.889 for the ACT, PR and IB selections, respectively"* `[Source: Main, Page 3]` |
| **Synonymous wild-type dispersion** | Weighted mean **$-0.01 \pm 0.07$** ($\pm 1\text{ SD}$ across synonymous WT codons; beneficial threshold $\zeta > 0.15$, $P < 0.01$) | *"Weighted means for synonymous codon fitness metrics, where the weights were read counts (depth of coverage) for each mutation, were calculated to be 0.03±0.09, −0.02±0.11 and −0.01±0.07 for the ACT, PR and IB data sets, respectively."* `[Source: Main, Page 9]` |
| **Quantitative lower bound ($\zeta_{\text{LB}}$)** | **$\zeta_{\text{LB}} = -0.6$** (derived from $\le 10$ post-selection reads vs. half-median pre-selection count of 31 and post/pre read ratio of 2.04) | *"The lower bounds were found to be −1.3, −0.8 and −0.6 for the ACT, PR and IB selections, respectively."* `[Source: Main, Page 3]` |
| **Pre-selection count vs. fitness check** | Pearson $r = -0.064$ (Supplementary Fig. 2c), confirming no pre-selection abundance bias | *"found them to be to 0.047, 0.033 and −0.064 for the ACT, PR and IB selections, respectively (Supplementary Fig. 2)."* `[Source: Main, Page 2]` |

### 1.2 Factors governing the error structure

1. **Compressed deleterious floor below $\zeta_{\text{LB}} = -0.6$ (`-0.7742` standardized):** Because `pEDA6_amiE` in $10\text{ mM}$ isobutyramide had a lower pre-selection half-median read count (31 reads) and lower wild-type enrichment $e_{\text{WT}}$, the 10-read post-selection detection threshold sits at $\zeta = -0.6$. Thus, while $\sim 80\%$ of the dataset ($4,977 / 6,229$ rows) lies above $-0.6$ with very tight precision (median $|\text{rep}_1 - \text{rep}_2| = 0.0499$), strongly deleterious variants are compressed into $[-2.2788, -0.6000]$.
2. **Higher intracellular expression on `pEDA6_amiE` buffers marginally destabilizing mutations:** Expressing `AmiE` from the stronger `proK14 + t7RBS` cassette (`pEDA6_amiE`) rather than `proK17 + kRBS3` (`pEDA2_amiE`) raises cytoplasmic protein concentration, reducing the fraction of deleterious missense variants to $45.4\%$ (vs. $75.8\%$ on ACT and $74.2\%$ on PR) and allowing destabilized active-site pocket-opening mutants like `M203W` ($\zeta = +0.43$, which precipitates upon in vitro purification; Main p. 9) to function in vivo.

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **6,229 unique sequences** (1 wild-type + 6,228 single-site missense variants across residues 1–341) of the 346-aa ***Pseudomonas aeruginosa* PAO1 aliphatic amidase `AmiE`** (UniProt `P11436`, **EC 3.5.1.4**, homohexamer, PDB `2UXY`).
- **Global encoding of substrate specificity (`9–24 Å` shell):** A central finding of Wrenbeck et al. (2017) is that mutations selectively improving isobutyramide fitness ($\zeta_{\text{IB}} > 0.15$) are not restricted to the first contact shell of the acyl-binding pocket (`Trp138`, `Ile165`, `Val201`, `Met202`, `Met203`, `Ala234`), but Peak $9\text{–}24\,\text{\AA}$ away from the active site across subunit interfaces of the homohexamer (Fig. 6, Main p. 7).
- **Unmutated C-terminal tail (`342–346`):** Residues `342–346` (`LEKEA`) were outside the four PFunkel tiles and are invariant across all 6,229 rows.
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** All 6,229 rows are selected on **$10\text{ mM}$ isobutyramide** ($(\text{CH}_3)_2\text{CHCONH}_2$, recrystallized from water).

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Pooled competitive exponential growth selection of *E. coli* MG1655 $rph^+$ in $\text{M9 N}^-$ minimal medium with $10\text{ mM}$ isobutyramide as sole nitrogen source over $\sim 8$ generations, read out by 300 bp PE Illumina MiSeq sequencing | No | `reported` | *"Three millilitre of selection media (M9 N− + 10 mM ACT, M9 N− + 15 mM PR and M9 N− + 10 mM IB for ACT, PR and IB selections, respectively) was inoculated to an initial OD600 of 0.02 at a volume of 3 ml (46 × 106 cells)."* `[Source: Main, Page 8]` |
| **Analytical readout** | Enrich log2 enrichment ratio over $g_p = 8$ doublings; validated by clonal $\text{OD}_{600}$ growth rates and Berthelot ammonia assays at $625\text{ nm}$ | No | `reported` | *"Pre- and post-selection samples were pooled and sequenced with 300 bp PE reads on an Illumina MiSeq available at the Michigan State University sequencing core."* `[Source: Main, Page 8]` |
| **Expression system** | Constitutive cytoplasmic expression in *E. coli* MG1655 $rph^+$ from low-copy plasmid `pEDA6_amiE` (`proK14` promoter + T7 gene 10 RBS `t7RBS`, $\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.54 \pm 0.11$) | No | `reported` | *"Promoter proK14 with the high translational efficiency RBS from gene 10 of T7 bacteriophage (t7RBS) had a suitable mS,wt/mM9,wt at 0.54±0.11 for IB selection media (plasmid pEDA6_amiE, Fig. 2 and Supplementary Table 1)."* `[Source: Main, Page 2]` |
| **Expression decoupled from label** | **false** (in vivo ammonium flux depends on both cytoplasmic homohexamer folding/solubility and specific catalytic activity) | — | `reported` | *"In this work, we address this open question by finding the local fitness landscapes for two different variants of the amidase AmiE in which catalytic efficiencies are statistically indistinguishable but the enzyme variants have lower probabilities of folding in vivo."* `[Source: Review, Page 2]` |
| **Temperature ($T$)** | **$37\,^\circ\text{C}$** ($250\text{ rpm}$ shaking) | No (constant) | `reported` | *"Cultures were grown at 37 °C with shaking at 250 r.p.m."* `[Source: Main, Page 9]` |
| **pH & buffer** | 1X $\text{M9 N}^-$ minimal salts ($\text{pH } \sim 7.0$ extracellularly; *E. coli* cytoplasm $\text{pH } \sim 7.5$) + $50\,\mu\text{g/mL}$ carbenicillin | No (constant) | `reported` | Main p. 8 |
| **Cofactors & metals** | None (`Cys166–Glu59–Lys134` triad requires no metal or organic cofactor) | No | `not_reported` | — |
| **Substrate loading** | **$10\text{ mM}$ isobutyramide** (recrystallized from water; $\sim 0.034\times$ wild-type $K_M = 297.2 \pm 54.8\text{ mM}$) | No (constant) | `reported` | *"PR and IB solids were recrystallized from ethyl acetate and water, respectively."* `[Source: Main, Page 8]` |
| **Reaction time** | **$\sim 8$ generations** of exponential growth (two serial 4-generation passages from $\text{OD}_{600} = 0.02$) | No (constant) | `reported` | Main p. 8 |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation from `amiESelectionFitnessData_Isobutyramide.txt` to 6,229 rows

The raw Figshare file `data/raw/dms/amie/amiESelectionFitnessData_Isobutyramide.txt` contains $341\text{ positions} \times 21\text{ symbols (20 AA + `*`)} = \mathbf{7,161\text{ rows}}$:
- **6,228 finite missense rows** (`Residue != WT_AA` and `Residue != '*'`).
- **341 finite wild-type rows** (`Residue == WT_AA` at positions 1–341, all with `normalized_fitness == 0.0`).
- **341 finite nonsense (`*`) rows** ($6,228\text{ missense} + 341\text{ nonsense} = \mathbf{6,569\text{ non-synonymous + nonsense variants}}$, matching the exact $n = 6,569$ reported for the global replicate correlation on Main p. 3!).
- **251 `NA` rows** (all 251 are missense rows with $\le 5$ pre-selection reads or complete post-selection dropout).

In [`scripts/data_prep/real_cases/amie_dms.py`](../../scripts/data_prep/real_cases/amie_dms.py#L64-L117), `build_amie_rows` drops the 341 stop-codon (`*`) rows and 251 missense `NA` rows, constructs the 346-aa sequences, and collapses the 341 synonymous wild-type rows into **1 wild-type row** (`amie_isobutyramide_00001`), yielding **$1\text{ WT} + 6,228\text{ missense} = \mathbf{6,229\text{ unique sequences}}$**.

### 4.2 Target transformation

Enzymado applies an **Identity** transform to `normalized_fitness` ($\zeta_i = \log_2(\mu_{S,i}/\mu_{S,\text{wt}})$), followed by train-split z-score standardization ($\mu_{\text{train}} = -0.3015$, $\sigma_{\text{train}} = 0.3856$ across the 4,989 training rows; raw range $[-2.2788, +0.8438]$, skewness $= -0.056$).

---

## 5. Split semantics and leakage caveats

One split file is published: `amie_isobutyramide_random_80_10_10_standardized` ($4,989\text{ train} / 620\text{ val} / 620\text{ test} = 6,229\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/amie_isobutyramide_dms/dataset_card.json) for exact split moments).

1. **Position-level sharing across splits (`random_split_unit="point"`):** Every scanned residue position (1–341) is mutated $\sim 14.6$ times in `train` and $\sim 1.8$ times in `val` and `test`; the split evaluates within-position substitution interpolation.
2. **Cross-task sequence overlap with `amie_acetamide_dms` and `amie_propionamide_dms`:** Over 6,100 protein sequences are shared across the three AmiE tasks; never leak labels or split information across tasks.
3. **Left-censoring below $\zeta = -0.6$ (`-0.7742` SD):** $\sim 20\%$ of variants fall below the quantitative detection limit ($\zeta_{\text{LB}} = -0.6$).
4. **Identifier leakage risk:** `dataset_id` (`amie_isobutyramide_00001` … `amie_isobutyramide_06229`) is ordered monotonically by sequence position and must never be used as a feature.
