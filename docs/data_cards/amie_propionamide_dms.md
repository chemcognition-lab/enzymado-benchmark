---
card_type: dataset
task_id: amie_propionamide_dms
dataset_id: amie_propionamide
property_name: amie_propionamide_dms
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
label_noise_floor: replicate Pearson r = 0.949 above lower bound zeta = -0.8 (global Pearson r = 0.842, Spearman rho = 0.9412, median |rep1 - rep2| = 0.0708; synonymous WT SD = 0.11)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/amie_propionamide_dms/dataset_card.json#/statistics
---

# `amie_propionamide_dms` — *Pseudomonas aeruginosa* AmiE deep mutational scan on propionamide

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/amie_propionamide_dms/dataset_card.json`](../../data/processed/real_case/amie_propionamide_dms/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is the intermediate-substrate arm ($15\text{ mM}$ propionamide, linear C3 amide) of the matched 3-substrate single-mutation DMS of *Pseudomonas aeruginosa* `AmiE` (346 aa).** Paired with `amie_acetamide_dms` ($10\text{ mM}$ acetamide, C2 linear) and `amie_isobutyramide_dms` ($10\text{ mM}$ isobutyramide, C4 branched), it profiles **6,292 unique sequences** (1 wild-type + 6,291 single-site missense variants across residues 1–341) under competitive *E. coli* nitrogen-source growth selection.
2. **It probes a sub-saturating $[S] < K_M$ kinetic regime despite high $k_{\text{cat}}$.** On propionamide, wild-type `AmiE` has a $2.45\times$ higher $k_{\text{cat}}$ than on acetamide ($144.7 \pm 9.9\text{ s}^{-1}$ vs. $59.0 \pm 2.0\text{ s}^{-1}$), but an $11.2\times$ higher $K_M$ ($52.7 \pm 8.3\text{ mM}$ vs. $4.7 \pm 0.5\text{ mM}$; $k_{\text{cat}}/K_M = 2,750\text{ M}^{-1}\text{s}^{-1}$, Table 2, Main p. 8). Because the selection uses **$15\text{ mM}$ propionamide** ($\sim 0.28\times K_M$) with the same low-expression plasmid `pEDA2_amiE` (`proK17` + `kRBS3`) used for acetamide, wild-type relative growth rate is $\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.37 \pm 0.08$, and mutations that lower $K_M$ (improving C3 acyl chain accommodation) or boost $k_{\text{cat}}$ directly increase cellular growth rate.
3. **High replicate agreement above the $\zeta = -0.8$ lower bound (`r = 0.949`), plus a single extreme Poisson boundary outlier (`E320W = -10.0`) in `train`.** Above the sequencing-depth lower bound ($\zeta \ge -0.8$, $n = 2,954$), biological replicate correlation is **`r = 0.949`** (and **`r = 0.842`** globally across all $n = 6,630$ variants, with Spearman $\rho = 0.9412$). Below $\zeta = -0.8$, dead variants pile up near the average stop-codon floor of **$-1.9$**—except for a single variant in the training set (`E320W`, `amie_propionamide_05906`), where post-selection depletion $e_i / 8$ landed at $-1 + 2^{-10}$ and produced **`normalized_fitness = -10.0000`** (`-12.8388` standardized SD units, while the second-lowest value across the entire dataset is `-3.1704`).
4. **Under the point-stratified 80/10/10 split, every scanned position (1–341) appears $\sim 14.8$ times in `train`.** Splitting at the observation level (`random_split_unit="point"`) tests **within-position amino acid substitution interpolation** rather than unseen-site or unseen-scaffold extrapolation.

**Practical consequences for use:**
- Guard against the single `-12.84 SD` training outlier (`E320W`, `normalized_fitness = -10.0000`) when training with unclipped MSE loss, and report Spearman $\rho$ and above-bound $R^2$ ($\zeta \ge -0.8$, standardized $\ge +0.1555$) alongside global $R^2$.
- Never leak labels or split masks across `amie_acetamide_dms`, `amie_propionamide_dms`, and `amie_isobutyramide_dms`.
- Never pass `dataset_id` (`amie_propionamide_00001` … `amie_propionamide_06292`) as a model feature.

---

## 1. Label noise and the measurement floor

### 1.1 Experimental quality controls and replicate structure

| Property | Value | Evidence |
|---|---|---|
| **Biological replicates** | **$n = 2$ replicate 8-generation selections** from a shared pre-selection library; published label combines reads across both replicates | *"Based on the high correlation between replicates and the fact that a major source of error in deep sequencing measurements are counting errors (Poisson noise)34, the fitness metrics used in subsequent analysis were computed by combining reads from the two replicates and repeating the analysis."* `[Source: Main, Page 8]` |
| **Replicate correlation above lower bound ($\zeta \ge -0.8$)** | **Pearson $r = 0.949$** ($n = 2,954$, $P < 2.2 \times 10^{-308}$; plotted in Fig. 3a, Main p. 4) | *"When we excluded variants with fitness metrics below the lower bounds the correlation coefficients improved to 0.932, 0.949 and 0.943 for the ACT, PR and IB selections, respectively"* `[Source: Main, Page 3]` |
| **Global replicate correlation (all variants)** | **Pearson $r = 0.842$** ($n = 6,630$); **Spearman $\rho = 0.9412$**, median $|\text{rep}_1 - \text{rep}_2| = 0.0708\ \zeta\text{ units}$ across 6,090 finite missense pairs | *"we calculated the Pearson product moment correlation coefficients of fitness metrics between replicates and found them to be 0.661, 0.842 and 0.889 for the ACT, PR and IB selections, respectively"* `[Source: Main, Page 3]` |
| **Synonymous wild-type dispersion** | Weighted mean **$-0.02 \pm 0.11$** ($\pm 1\text{ SD}$ across synonymous WT codons; beneficial threshold $\zeta > 0.15$, $P < 0.01$) | *"Weighted means for synonymous codon fitness metrics, where the weights were read counts (depth of coverage) for each mutation, were calculated to be 0.03±0.09, −0.02±0.11 and −0.01±0.07 for the ACT, PR and IB data sets, respectively."* `[Source: Main, Page 9]` |
| **Quantitative lower bound ($\zeta_{\text{LB}}$)** | **$\zeta_{\text{LB}} = -0.8$** (derived from $\le 10$ post-selection reads vs. half-median pre-selection count of 49 and post/pre read ratio of 1.86; stop-codon average floor $= -1.9$) | *"The lower bounds were found to be −1.3, −0.8 and −0.6 for the ACT, PR and IB selections, respectively."* `[Source: Main, Page 3]` |
| **Pre-selection count vs. fitness check** | Pearson $r = 0.033$ (Supplementary Fig. 2b), confirming no pre-selection abundance bias | *"found them to be to 0.047, 0.033 and −0.064 for the ACT, PR and IB selections, respectively (Supplementary Fig. 2)."* `[Source: Main, Page 2]` |

### 1.2 Error structure and the `E320W` (`-10.0000`) boundary point

Because wild-type relative growth rate on `pEDA2_amiE` in $15\text{ mM}$ propionamide was $\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.37 \pm 0.08$ (lower than the $0.56 \pm 0.06$ on acetamide), the sequencing-depth quantitative lower bound sits at $\zeta_{\text{LB}} = -0.8$ (`+0.1555` standardized), and the average stop-codon floor sits at $-1.9$. Across 6,291 of the 6,292 rows, `normalized_fitness` spans $[-3.1704, +0.6352]$. A single row in the training split—`amie_propionamide_05906` (`E320W`)—has `normalized_fitness = -10.0000` (`-12.8388` standardized) because its pooled enrichment ratio $e_i$ over $g_p = 8$ generations landed right at the $-8.0$ singularity of $\log_2(e_i/8 + 1)$ (whereas 188 other depleted missense variants landed at $e_i \le -8.0$ and were excluded as `NA`).

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **6,292 unique sequences** (1 wild-type + 6,291 single-site missense variants across residues 1–341) of the 346-aa ***Pseudomonas aeruginosa* PAO1 aliphatic amidase `AmiE`** (UniProt `P11436`, **EC 3.5.1.4**, homohexamer, PDB `2UXY`).
- **DFE and cross-substrate comparison:** Like acetamide, propionamide is dominated by purifying selection: $74.2\%$ of missense mutations reduce growth rate by $\ge 20\%$, while **$5.1\%$ ($n = 328$) exceed wild-type ($\zeta > 0$)** (Main p. 3, Fig. 4a). Above the lower bound, propionamide and acetamide fitness scores correlate at $r = 0.87$ ($n = 3,054$, Fig. 5a, Main p. 6), with a small set of binding-pocket and second-shell mutations (e.g., `R89E`, `V201T`, `M202H`, `Q273A` in Table 2) selectively enhancing propionamide turnover.
- **Unmutated C-terminal tail (`342–346`):** Residues `342–346` (`LEKEA`) were outside the four PFunkel tiles and are invariant across all 6,292 rows.
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** All 6,292 rows are selected on **$15\text{ mM}$ propionamide** ($\text{CH}_3\text{CH}_2\text{CONH}_2$, recrystallized from ethyl acetate).

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Pooled competitive exponential growth selection of *E. coli* MG1655 $rph^+$ in $\text{M9 N}^-$ minimal medium with $15\text{ mM}$ propionamide as sole nitrogen source over $\sim 8$ generations, read out by 300 bp PE Illumina MiSeq sequencing | No | `reported` | *"Three millilitre of selection media (M9 N− + 10 mM ACT, M9 N− + 15 mM PR and M9 N− + 10 mM IB for ACT, PR and IB selections, respectively) was inoculated to an initial OD600 of 0.02 at a volume of 3 ml (46 × 106 cells)."* `[Source: Main, Page 8]` |
| **Analytical readout** | Enrich log2 enrichment ratio over $g_p = 8$ doublings; validated by clonal $\text{OD}_{600}$ growth rates and Berthelot ammonia assays at $625\text{ nm}$ | No | `reported` | *"Pre- and post-selection samples were pooled and sequenced with 300 bp PE reads on an Illumina MiSeq available at the Michigan State University sequencing core."* `[Source: Main, Page 8]` |
| **Expression system** | Constitutive cytoplasmic expression in *E. coli* MG1655 $rph^+$ from low-copy plasmid `pEDA2_amiE` (`proK17` promoter + knockdown `kRBS3`, $\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.37 \pm 0.08$) | No | `reported` | *"One construct containing promoter proK17 and a knockdown RBS 3 (kRBS3) sequence had a mS,wt/mM9,wt of 0.56±0.06 for ACT and 0.37±0.08 for PR (plasmid pEDA2_amiE, Fig. 2 and Supplementary Table 1)."* `[Source: Main, Page 2]` |
| **Expression decoupled from label** | **false** (in vivo ammonium flux depends on both cytoplasmic homohexamer folding/solubility and specific catalytic activity) | — | `reported` | *"In this work, we address this open question by finding the local fitness landscapes for two different variants of the amidase AmiE in which catalytic efficiencies are statistically indistinguishable but the enzyme variants have lower probabilities of folding in vivo."* `[Source: Review, Page 2]` |
| **Temperature ($T$)** | **$37\,^\circ\text{C}$** ($250\text{ rpm}$ shaking) | No (constant) | `reported` | *"Cultures were grown at 37 °C with shaking at 250 r.p.m."* `[Source: Main, Page 9]` |
| **pH & buffer** | 1X $\text{M9 N}^-$ minimal salts ($\text{pH } \sim 7.0$ extracellularly; *E. coli* cytoplasm $\text{pH } \sim 7.5$) + $50\,\mu\text{g/mL}$ carbenicillin | No (constant) | `reported` | Main p. 8 |
| **Cofactors & metals** | None (`Cys166–Glu59–Lys134` triad requires no metal or organic cofactor) | No | `not_reported` | — |
| **Substrate loading** | **$15\text{ mM}$ propionamide** (recrystallized from ethyl acetate; $\sim 0.28\times$ wild-type $K_M = 52.7 \pm 8.3\text{ mM}$) | No (constant) | `reported` | *"PR and IB solids were recrystallized from ethyl acetate and water, respectively."* `[Source: Main, Page 8]` |
| **Reaction time** | **$\sim 8$ generations** of exponential growth (two serial 4-generation passages from $\text{OD}_{600} = 0.02$) | No (constant) | `reported` | Main p. 8 |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation from `amiESelectionFitnessData_Propionamide.txt` to 6,292 rows

The raw Figshare file `data/raw/dms/amie/amiESelectionFitnessData_Propionamide.txt` contains $341\text{ positions} \times 21\text{ symbols (20 AA + `*`)} = \mathbf{7,161\text{ rows}}$:
- **6,291 finite missense rows** (`Residue != WT_AA` and `Residue != '*'`).
- **341 finite wild-type rows** (`Residue == WT_AA` at positions 1–341, all with `normalized_fitness == 0.0`).
- **339 finite nonsense (`*`) rows** ($6,291\text{ missense} + 339\text{ nonsense} = \mathbf{6,630\text{ non-synonymous + nonsense variants}}$, matching the exact $n = 6,630$ reported for the global replicate correlation on Main p. 3!).
- **190 `NA` rows** (188 missense + 2 `*`).

In [`scripts/data_prep/real_cases/amie_dms.py`](../../scripts/data_prep/real_cases/amie_dms.py#L64-L108), `build_amie_rows` drops the 341 stop-codon (`*`) rows and 188 missense `NA` rows, constructs the 346-aa sequences, and collapses the 341 synonymous wild-type rows into **1 wild-type row** (`amie_propionamide_00001`), yielding **$1\text{ WT} + 6,291\text{ missense} = \mathbf{6,292\text{ unique sequences}}$**.

### 4.2 Target transformation

Enzymado applies an **Identity** transform to `normalized_fitness` ($\zeta_i = \log_2(\mu_{S,i}/\mu_{S,\text{wt}})$), followed by train-split z-score standardization ($\mu_{\text{train}} = -0.9101$, $\sigma_{\text{train}} = 0.7080$ across the 5,032 training rows; raw range $[-10.0000, +0.6352]$).

---

## 5. Split semantics and leakage caveats

One split file is published: `amie_propionamide_random_80_10_10_standardized` ($5,032\text{ train} / 630\text{ val} / 630\text{ test} = 6,292\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/amie_propionamide_dms/dataset_card.json) for exact split moments).

1. **Position-level sharing across splits (`random_split_unit="point"`):** Every scanned residue position (1–341) is mutated $\sim 14.8$ times in `train` and $\sim 1.8$ times in `val` and `test`; the split evaluates within-position substitution interpolation.
2. **Cross-task sequence overlap with `amie_acetamide_dms` and `amie_isobutyramide_dms`:** Over 6,100 protein sequences are shared across the three AmiE tasks; never leak labels or split information across tasks.
3. **Single extreme outlier in `train` (`E320W = -12.8388` SD) and left-censoring below $\zeta = -0.8$ (`+0.1555` SD):** Over half the dataset sits below the quantitative detection limit ($\zeta_{\text{LB}} = -0.8$).
4. **Identifier leakage risk:** `dataset_id` (`amie_propionamide_00001` … `amie_propionamide_06292`) is ordered monotonically by sequence position and must never be used as a feature.
