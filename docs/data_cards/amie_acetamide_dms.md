---
card_type: dataset
task_id: amie_acetamide_dms
dataset_id: amie_acetamide
property_name: amie_acetamide_dms
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
label_noise_floor: replicate Pearson r = 0.932 above lower bound zeta = -1.3 (global Pearson r = 0.661, Spearman rho = 0.9502, median |rep1 - rep2| = 0.0887; synonymous WT SD = 0.09)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/amie_acetamide_dms/dataset_card.json#/statistics
---

# `amie_acetamide_dms` — *Pseudomonas aeruginosa* AmiE deep mutational scan on acetamide

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/amie_acetamide_dms/dataset_card.json`](../../data/processed/real_case/amie_acetamide_dms/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is the native-substrate arm of a matched 3-substrate single-mutation DMS of *Pseudomonas aeruginosa* aliphatic amidase (`AmiE`, 346 aa).** Wrenbeck, Azouz & Whitehead (*Nature Communications* 2017) scanned all single-site substitutions across positions 1–341 of homohexameric `AmiE` (UniProt `P11436`, EC 3.5.1.4) under competitive *E. coli* growth selection where an aliphatic amide is the **sole nitrogen source**:
   - **`amie_acetamide_dms` ($10\text{ mM}$ acetamide, native optimal substrate):** $k_{\text{cat}} = 59.0\text{ s}^{-1}$, $K_M = 4.7\text{ mM}$, $k_{\text{cat}}/K_M = 1.26 \times 10^4\text{ M}^{-1}\text{s}^{-1}$ (6,290 rows).
   - **`amie_propionamide_dms` ($15\text{ mM}$ propionamide, intermediate substrate):** $k_{\text{cat}} = 144.7\text{ s}^{-1}$, $K_M = 52.7\text{ mM}$, $k_{\text{cat}}/K_M = 2.75 \times 10^3\text{ M}^{-1}\text{s}^{-1}$ (6,292 rows).
   - **`amie_isobutyramide_dms` ($10\text{ mM}$ isobutyramide, bulky branched substrate):** $k_{\text{cat}} = 13.3\text{ s}^{-1}$, $K_M = 297.2\text{ mM}$, $k_{\text{cat}}/K_M = 44.8\text{ M}^{-1}\text{s}^{-1}$ (6,229 rows).
2. **Constitutive promoter/RBS tuning couples cellular growth rate linearly to intracellular amidase flux.** Because wild-type `AmiE` is fast on acetamide, the authors engineered a weak synthetic promoter (`proK17`) and knockdown ribosome binding site (`kRBS3`) on low-copy plasmid `pEDA2_amiE` so that wild-type *E. coli* MG1655 $rph^+$ grows in $\text{M9 N}^- + 10\text{ mM}$ acetamide at $56 \pm 6\%$ of its ammonium-replete M9 growth rate ($\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.56 \pm 0.06$). Consequently, relative fitness $\zeta_i = \log_2(\mu_{S,i}/\mu_{S,\text{wt}})$ directly reports changes in intracellular ammonia liberation (which is the product of in vivo homohexamer folding probability at $37\,^\circ\text{C}$ and catalytic turnover at $10\text{ mM}$ acetamide).
3. **Below the sequencing-depth lower bound ($\zeta = -1.3$), scores are dominated by logarithmic transformation of Poisson counting noise.** Across variants above the sequencing-depth lower bound ($\zeta \ge -1.3$, $n = 3,834$), biological replicate correlation is **`r = 0.932`**. However, for strongly deleterious variants whose post-selection read counts drop below 10 reads after $g_p = 8$ generations, the growth-rate formula $\zeta_i = \log_2((e_i/g_p + 1)/(e_{\text{WT}}/g_p + 1))$ diverges toward $-\infty$ as $e_i \to -8^+$, stretching raw values down to **`-10.5192`** (and turning 190 missense variants with $e_i \le -8$ into `NA`). Enzymado retains all 6,289 finite missense rows plus wild-type (6,290 rows total), so $\sim 39\%$ of the dataset sits in this left-censored/Poisson-noisy tail below $\zeta = -1.3$.
4. **Under the point-stratified 80/10/10 split, every scanned position (1–341) appears $\sim 14.8$ times in `train`.** Because splitting is performed at the variant level (`random_split_unit="point"`), the split tests **within-position amino acid substitution interpolation** rather than unseen-site or unseen-scaffold extrapolation.

**Practical consequences for use:**
- Report **Spearman $\rho$** and **above-bound $R^2$ ($\zeta \ge -1.3$, standardized $\ge -0.1036$)** alongside global pooled $R^2$: global MSE/$R^2$ is heavily weighted by whether a model predicts $-2.0$ vs. $-8.0$ on categorically dead variants whose true growth rate is zero.
- Never leak predictions or split masks across `amie_acetamide_dms`, `amie_propionamide_dms`, and `amie_isobutyramide_dms`.
- Never pass `dataset_id` (`amie_acetamide_00001` … `amie_acetamide_06290`, which is ordered monotonically by sequence position) as a model feature.

---

## 1. Label noise and the measurement floor

Each selection was performed in two biological replicates from a shared pre-selection library, and the raw Figshare file (`amiESelectionFitnessData_Acetamide.txt`) preserves both the individual replicate scores (`normalized_fitness_rep1`, `normalized_fitness_rep2`) and the combined-reads score (`normalized_fitness`).

### 1.1 Experimental quality controls and replicate structure

| Property | Value | Evidence |
|---|---|---|
| **Biological replicates** | **$n = 2$ replicate 8-generation selections** from a shared pre-selection library; published label combines reads across both replicates | *"Based on the high correlation between replicates and the fact that a major source of error in deep sequencing measurements are counting errors (Poisson noise)34, the fitness metrics used in subsequent analysis were computed by combining reads from the two replicates and repeating the analysis."* `[Source: Main, Page 8]` |
| **Replicate correlation above lower bound ($\zeta \ge -1.3$)** | **Pearson $r = 0.932$** ($n = 3,834$, $P < 2.2 \times 10^{-308}$) | *"When we excluded variants with fitness metrics below the lower bounds the correlation coefficients improved to 0.932, 0.949 and 0.943 for the ACT, PR and IB selections, respectively"* `[Source: Main, Page 3]` |
| **Global replicate correlation (all variants)** | **Pearson $r = 0.661$** ($n = 6,627$); **Spearman $\rho = 0.9502$**, median $|\text{rep}_1 - \text{rep}_2| = 0.0887\ \zeta\text{ units}$ across 6,073 finite missense pairs | *"we calculated the Pearson product moment correlation coefficients of fitness metrics between replicates and found them to be 0.661, 0.842 and 0.889 for the ACT, PR and IB selections, respectively"* `[Source: Main, Page 3]` |
| **Synonymous wild-type dispersion** | Weighted mean **$0.03 \pm 0.09$** ($\pm 1\text{ SD}$ across synonymous WT codons; beneficial threshold $\zeta > 0.15$, $P < 0.01$) | *"Weighted means for synonymous codon fitness metrics, where the weights were read counts (depth of coverage) for each mutation, were calculated to be 0.03±0.09, −0.02±0.11 and −0.01±0.07 for the ACT, PR and IB data sets, respectively."* `[Source: Main, Page 9]` |
| **Quantitative lower bound ($\zeta_{\text{LB}}$)** | **$\zeta_{\text{LB}} = -1.3$** (derived from $\le 10$ post-selection reads vs. half-median pre-selection count of 63 and post/pre read ratio of 2.97; stop-codon average floor $= -2.6$) | *"The lower bounds were found to be −1.3, −0.8 and −0.6 for the ACT, PR and IB selections, respectively."* `[Source: Main, Page 3]` |
| **Extracellular cross-feeding ("cheating") check** | 1:1 competition of WT vs. dead `C166S` gave $\zeta_{\text{C166S}} = -2.40 \pm 0.9$ vs. $-2.46$ theoretical zero-growth | *"Using the fitness equations laid out in Kowalsky et al.34, the average fitness metric for the C166S mutant was −2.40±0.9 (n = 2), close to the fitness metric expected if no cheating occurred (−2.46)."* `[Source: Main, Page 2]` |

### 1.2 Why the negative tail extends to `-10.52` (and why 190 missense variants are `NA`)

In the Kowalsky et al. growth-rate normalization (Main p. 9, Eqs. 2–3), each variant's log-enrichment ratio is $e_i = \log_2(f_{\text{final},i} / f_{\text{initial},i})$ and its relative growth rate over $g_p = 8$ population doublings is:

$$\zeta_i = \log_2\!\left(\frac{e_i / g_p + 1}{e_{\text{WT}} / g_p + 1}\right)$$

When a misfolded or catalytically dead mutant fails to divide over $g_p = 8$ generations of wild-type growth, its frequency drops by $\sim 2^{-8}$, so $e_i \approx -8$ and $e_i / g_p + 1 \approx 0$.
- If Poisson read sampling yields $e_i \le -8.0$ (so $e_i / 8 + 1 \le 0$), the logarithm is undefined and Enrich outputs `NA` (190 missense rows and 3 stop-codon rows in `amiESelectionFitnessData_Acetamide.txt`).
- If Poisson read sampling yields $e_i$ just above $-8.0$ (e.g., $e_i = -7.999$), $\log_2(e_i / 8 + 1)$ plunges to extreme negative numbers down to **$-10.5192$**. Thus, all values below $\zeta_{\text{LB}} = -1.3$ represent categorically deleterious variants where exact numerical magnitude reflects Poisson read noise rather than true growth rate.

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **6,290 unique sequences** (1 wild-type + 6,289 single-site missense variants across residues 1–341) of the 346-aa ***Pseudomonas aeruginosa* PAO1 aliphatic amidase `AmiE`** (UniProt `P11436`, **EC 3.5.1.4**, nitrilase superfamily, homohexamer with `Cys166–Glu59–Lys134` catalytic triad; crystal structure PDB `2UXY`).
- **Distribution of Fitness Effects (DFE) on the native substrate acetamide:** Because acetamide is the small, native optimal substrate of `AmiE`, $75.8\%$ of missense mutations are deleterious ($\ge 20\%$ growth rate reduction), while only **$4.7\%$ ($n = 306$) exceed wild-type ($\zeta > 0$)** (Main p. 3, Fig. 4a). As shown in the companion stability study (Faber et al. 2019, `review` pp. 2–15), most deleterious mutations outside the active site destabilize in vivo folding of the homohexamer.
- **Unmutated C-terminal tail (`342–346`):** The four PFunkel mutagenesis tiles spanned residues `1–85`, `86–170`, `171–255`, and `256–341`; the final 5 C-terminal residues (`342–346`, `LEKEA`) were not mutagenized.
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** All 6,290 rows are assayed on **$10\text{ mM}$ acetamide** ($\text{CH}_3\text{CONH}_2$). No SMILES or condition columns are included in Layout A.

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Pooled competitive exponential growth selection of *E. coli* MG1655 $rph^+$ in $\text{M9 N}^-$ minimal medium with $10\text{ mM}$ acetamide as sole nitrogen source over $\sim 8$ generations, read out by 300 bp PE Illumina MiSeq sequencing | No | `reported` | *"Three millilitre of selection media (M9 N− + 10 mM ACT, M9 N− + 15 mM PR and M9 N− + 10 mM IB for ACT, PR and IB selections, respectively) was inoculated to an initial OD600 of 0.02 at a volume of 3 ml (46 × 106 cells)."* `[Source: Main, Page 8]` |
| **Analytical readout** | Enrich log2 enrichment ratio over $g_p = 8$ doublings; validated by clonal $\text{OD}_{600}$ growth rates and cell-lysate/purified-enzyme phenol–hypochlorite (Berthelot) ammonia assays at $625\text{ nm}$ | No | `reported` | *"Deep-sequencing derived fitness corresponded to increased growth rates for 16/17 beneficial mutations, near wild-type growth rates for 2/2 neutral mutations, and no growth for 1/1 deleterious mutation tested."* `[Source: Main, Page 3]` |
| **Expression system** | Constitutive cytoplasmic expression in *E. coli* MG1655 $rph^+$ from low-copy plasmid `pEDA2_amiE` (`proK17` promoter + knockdown `kRBS3`, $\mu_{S,\text{wt}}/\mu_{\text{M9,wt}} = 0.56 \pm 0.06$) | No | `reported` | *"One construct containing promoter proK17 and a knockdown RBS 3 (kRBS3) sequence had a mS,wt/mM9,wt of 0.56±0.06 for ACT and 0.37±0.08 for PR (plasmid pEDA2_amiE, Fig. 2 and Supplementary Table 1)."* `[Source: Main, Page 2]` |
| **Expression decoupled from label** | **false** (in vivo ammonium flux depends on both cytoplasmic homohexamer folding/solubility and specific catalytic activity) | — | `reported` | *"In this work, we address this open question by finding the local fitness landscapes for two different variants of the amidase AmiE in which catalytic efficiencies are statistically indistinguishable but the enzyme variants have lower probabilities of folding in vivo."* `[Source: Review, Page 2]` |
| **Temperature ($T$)** | **$37\,^\circ\text{C}$** ($250\text{ rpm}$ shaking) | No (constant) | `reported` | *"Cultures were grown at 37 °C with shaking at 250 r.p.m."* `[Source: Main, Page 9]` |
| **pH & buffer** | 1X $\text{M9 N}^-$ minimal salts ($\text{pH } \sim 7.0$ extracellularly; *E. coli* cytoplasm $\text{pH } \sim 7.5$) + $50\,\mu\text{g/mL}$ carbenicillin | No (constant) | `reported` | *"except 1X M9 minimal media lacking ammonium chloride (M9 N−) was used to wash cell pellets before inoculation of selection media."* `[Source: Main, Page 8]` |
| **Cofactors & metals** | None (`Cys166–Glu59–Lys134` triad requires no metal or organic cofactor) | No | `not_reported` | — |
| **Substrate loading** | **$10\text{ mM}$ acetamide** ($\sim 2.1\times$ wild-type $K_M = 4.7 \pm 0.5\text{ mM}$) | No (constant) | `reported` | Main p. 8 |
| **Reaction time** | **$\sim 8$ generations** of exponential growth (two serial 4-generation passages from $\text{OD}_{600} = 0.02$) | No (constant) | `reported` | *"after approximately four generations the cells were harvested, washed with M9 N−, and a fresh 3 ml culture with selection media was inoculated to the same initial OD600 of 0.02"* `[Source: Main, Page 8]` |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation from `amiESelectionFitnessData_Acetamide.txt` to 6,290 rows

The raw Figshare file `data/raw/dms/amie/amiESelectionFitnessData_Acetamide.txt` contains a complete $341\text{ positions} \times 21\text{ symbols (20 AA + `*`)} = \mathbf{7,161\text{ rows}}$ grid:
- **6,289 finite missense rows** (`Residue != WT_AA` and `Residue != '*'`).
- **341 finite wild-type rows** (`Residue == WT_AA` at each of positions 1–341, all with `normalized_fitness == 0.0`).
- **338 finite nonsense (`*`) rows** (note that $6,289\text{ missense} + 338\text{ nonsense} = \mathbf{6,627\text{ non-synonymous + nonsense variants}}$, matching the exact $n = 6,627$ reported for the global replicate correlation on Main p. 3!).
- **193 `NA` rows** (190 missense + 3 `*`, corresponding to variants with $\le 5$ pre-selection reads or post-selection depletion $e_i \le -8$ where Eq. 3 has a non-positive argument inside $\log_2$).

In [`scripts/data_prep/real_cases/amie_dms.py`](../../scripts/data_prep/real_cases/amie_dms.py#L64-L99), `build_amie_rows` drops the 341 stop-codon (`*`) rows and the 190 missense `NA` rows, constructs the full 346-aa sequence for each row, and calls `collapse_duplicate_rows`, which collapses the 341 identical wild-type rows (`normalized_fitness = 0.0`) into **1 wild-type row** (`amie_acetamide_00001`), yielding **$1\text{ WT} + 6,289\text{ missense} = \mathbf{6,290\text{ unique sequences}}$**.

### 4.2 Target transformation

Because `normalized_fitness` ($\zeta_i = \log_2(\mu_{S,i}/\mu_{S,\text{wt}})$) is already a log-ratio growth rate centered at $\text{WT} = 0.0$, Enzymado applies an **Identity** transform followed by train-split z-score standardization ($\mu_{\text{train}} = -1.1734$, $\sigma_{\text{train}} = 1.2221$ across the 5,030 training rows; raw range $[-10.5192, +0.4070]$).

---

## 5. Split semantics and leakage caveats

One split file is published: `amie_acetamide_random_80_10_10_standardized` ($5,030\text{ train} / 630\text{ val} / 630\text{ test} = 6,290\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/amie_acetamide_dms/dataset_card.json) for exact split moments).

1. **Position-level sharing across splits (`random_split_unit="point"`):** Every scanned residue position (1–341) is mutated $\sim 14.8$ times in `train` and $\sim 1.8$ times in `val` and `test`; the split evaluates within-position substitution interpolation.
2. **Cross-task sequence overlap with `amie_propionamide_dms` and `amie_isobutyramide_dms`:** Over 6,100 of the 6,290 protein sequences are shared verbatim across all three AmiE tasks, and their fitness scores correlate strongly across folding-sensitive core residues ($r = 0.87$ between ACT and PR above the lower bound; Main p. 6). Never leak labels or split information across the three AmiE datasets.
3. **Left-censored Poisson noise tail ($\zeta < -1.3$, standardized $< -0.1036$):** $\sim 39\%$ of rows fall below the sequencing-depth lower bound $\zeta_{\text{LB}} = -1.3$, where numerical values down to $-10.52$ reflect single-read Poisson fluctuations.
4. **Identifier leakage risk:** `dataset_id` (`amie_acetamide_00001` … `amie_acetamide_06290`) is ordered monotonically by sequence position (WT first, then positions 1 to 341) and must never be used as a feature.
