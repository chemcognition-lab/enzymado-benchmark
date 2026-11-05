---
card_type: dataset
task_id: trpb3_epistasis
dataset_id: trpb3
property_name: trpb3_epistasis
track: real_case
task_type: protein_global
label_kind: scalar_regression
layout: Layout A
doi: 10.1073/pnas.2400439121
license: CC BY-NC-ND 4.0
source_pdfs:
  - docs/pdfs/trpb_johnston2024_pnas.pdf (main, pp. 1-11)
  - docs/pdfs/trpb_johnston2024_suppinfo.pdf (si, pp. 1-74)
  - docs/pdfs/trpb_long2025_arxiv.pdf (review, pp. 1-32)
noise_regime: single_lab_replicated
label_noise_floor: 0.0230 fitness units (inactive background robust SD across ~7,900 dead variants vs. train SD = 0.0515; single-replicate Flask A1 averaged over t = 18 h and 44 h; estimated R^2_max = 0.8006)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/trpb3_epistasis/dataset_card.json#/statistics
---

# `trpb3_epistasis` — *Thermotoga maritima* TrpB (`Tm9D8*`) 3-site combinatorial epistasis landscape (Library 3A: positions 104, 105, 106)

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/trpb3_epistasis/dataset_card.json`](../../data/processed/real_case/trpb3_epistasis/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is a combinatorially complete 3-site saturation library (`Library 3A`: $20^3 = 8,000$ theoretical variants, 7,970 rows in Enzymado) at active-site residues `Ala104`, `Glu105`, and `Thr106` of `Tm9D8*` (397 aa).** While the flagship landscape in Johnston et al. (*PNAS* 2024) is the 160,000-variant 4-site library at positions `183, 184, 227, 228`, the authors first constructed nine 3-site combinatorial libraries (`Libraries A–I`, $8,000$ variants each; Main p. 3, SI Fig. S6–S8, SI Table S6–S8). `trpb3_epistasis` ingests `Johnston2024_TrpB3A.csv` (**Library A**), which simultaneously mutates contiguous 1-based positions **`104`, `105`, and `106`** (parent tripeptide `AET`).
2. **Because `Glu105` is an essential catalytic residue, $\sim 99\%$ of the landscape is catalytically dead ("needle-in-a-haystack" distribution, skewness $= +10.22$).** In tryptophan synthase $\beta$-subunit (`TrpB`), `Glu105` coordinates the indole ring $\text{N1}$ hydrogen and stabilizes the aminoacrylate intermediate. Consequently, as noted in SI Fig. S8 (`si` p. 18) and SI Table S7 (`si` p. 61), almost no substitutions at positions 104–106 are tolerated: **$\sim 7,927$ of the 7,970 variants ($\sim 99.46\%$) are inactive** and cluster tightly in a narrow Gaussian around the stop-codon zero floor (median $= -0.0446$, IQR $= [-0.0604, -0.0294]$, robust $\text{SD} = 0.0230$), while only **$\sim 43$ variants ($\sim 0.54\%$)** exhibit substantial catalytic activity ($\text{fitness} > 0.15$, standardized $> +3.67\text{ SD}$), reaching the landscape maximum of **`1.0000` at the non-canonical epistatic triple mutant `ACC` (`Ala104–Cys105–Cys106`)** (parent `AET` $= 0.7278$, `ACT` $= 0.8583$, `GCT` $= 0.8286$).
3. **Library 3A was sequenced from a single biological replicate flask (`Flask A1`) across $t = 0, 18, 44\text{ h}$.** Whereas the 4-site library and 3-site libraries `D–I` were sequenced in two replicate flasks, SI Table S7 (`si` p. 61) and Main p. 9 explicitly document that for 3-site libraries `A, B, and C`, only **one replicate flask (`A1`)** was sequenced (at $t = 0\text{ h}$ [$\text{OD}_{600} = 0.10$], $18\text{ h}$ [$\text{OD}_{600} = 0.72$], and $44\text{ h}$ [$\text{OD}_{600} = 2.55$]) because preliminary results showed that nearly all variants were inactive.
4. **Under the 80/10/10 split, the task tests 3-body epistatic interaction learning at fixed positions (`104, 105, 106`).** Every single amino acid at each of the three positions appears $\sim 318$ times in `train`, and every pairwise combination at two of the three positions appears $\sim 16$ times in `train`. The challenge is predicting when a specific 3-residue combination (such as `Cys105 + Cys106` paired with `Ala104` or `Gly104`) unlocks high catalytic turnover via reciprocal sign epistasis.

**Practical consequences for use:**
- **Do not rely solely on pooled $R^2$ or MSE:** In the 800-row `test` split, $\sim 795$ rows are dead variants sitting near $0.0\text{ SD}$ and only $\sim 4\text{–}5$ rows are active variants sitting up to **`+16.05 SD`**. Pooled $R^2$ is overwhelmingly governed by those $4\text{–}5$ extreme right-tail points. Always report **active-variant AUPRC / ROC-AUC** (e.g., at $\text{fitness} > 0.10$ or $> 0.15$) and **Top-K recall** alongside $R^2$.
- Note the **1-row `"NAN"` parser drop**: out of 7,971 rows in `Johnston2024_TrpB3A.csv`, row 4390 is the valid tripeptide **`NAN` (`Asn104–Ala105–Asn106`, inactive with `fitness = -0.063185`)**, which [`utils.clean_text`](../../scripts/data_prep/real_cases/utils.py#L148-L155) treats as a null string (`text.lower() in {"nan", "none", "null"}`), yielding 7,970 rows in `trpb3_epistasis.csv`.
- Never pass `dataset_id` (`trpb3_00001` … `trpb3_07970`, ordered lexicographically from `AAA` to `YYY`) as a model input.

---

## 1. Label noise and the measurement floor

### 1.1 Experimental quality controls and noise anchors

| Property | Value | Evidence |
|---|---|---|
| **Replicate design for Library A** | **1 sequenced flask (`Flask A1`)**, sampled at $t = 0\text{ h}$ ($\text{OD}_{600} = 0.1$), $18\text{ h}$ ($\text{OD}_{600} = 0.72$), and $44\text{ h}$ ($\text{OD}_{600} = 2.55$) and averaged across timepoints; synonymous `NNK` codons aggregated per amino acid sequence | *"In libraries where only one replicate was sequences (3-site landscapes A, B, and C), the 99.5% percentile was used for the single replicate (2.576 SD above the mean)."* `[Source: Main, Page 9]`; *"*These samples were sequenced. Based on preliminary results where most variants were inactive, the remaining timepoints and replicates were not sequenced."* `[Source: SI, Page 61]` |
| **Upstream read-depth gate** | Excluded variants with $< 10$ counts at $t = 0$ or $0$ counts at $t > 0$ (retaining $7,971 / 8,000 = 99.64\%$ of Library A) | *"Sequences with fewer than an average of ten sequencing counts at t = 0 or zero counts in either replicate at any other timepoint were omitted from the fitness calculation pipeline."* `[Source: Main, Page 9]` |
| **Stop-codon background subtraction** | Subtracted mean stop-codon specific growth rate $\mu_{\text{stop}}$ at each timepoint and normalized by $(\mu_{\text{max}} - \mu_{\text{stop}})$ | *"We observed that sequences containing stop codons, which can be presumed nonfunctional, had slightly nonzero μi values. Therefore, we subtracted the average μstop from each μi . Finally, we divided this value by the background-subtracted μmax for that timepoint to normalize the maximum fitness of each landscape to 1."* `[Source: Main, Page 9]` |
| **Inactive background dispersion ($\sigma_{\text{bg}}$)** | Median $= -0.0446$, IQR $= 0.0310$, **robust $\text{SD} = 0.0230$** fitness units ($0.4466$ standardized SD units across the $\sim 99\%$ inactive bulk) | Direct distribution analysis of `Johnston2024_TrpB3A.csv` |
| **Estimated analytical $R^2_{\max}$ ceiling** | **$R^2_{\max} \approx 1 - \sigma_{\text{bg}}^2 / \text{Var}(y) = 1 - 0.0230^2 / 0.0515^2 = \mathbf{0.8006}$** | Because $99\%$ of variants have true activity $= 0$ and fluctuate by $\sigma_{\text{bg}} = 0.0230$, background noise accounts for $\sim 20\%$ of total variance ($\sigma_{\text{train}}^2 = 0.0515^2$) |

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **7,970 unique 397-aa sequences** (99.625% of all $20^3 = 8,000$ combinations at 1-based positions **`104`, `105`, and `106`**) of **`Tm9D8*`**, an engineered stand-alone $\beta$-subunit of tryptophan synthase from the hyperthermophile *Thermotoga maritima* (UniProt `Q56328` derivative carrying 10 mutations: `P19G, E30G, I69V, K96L, P140L, N167D, I184F, L213P, G228S, T292S`; **EC 4.2.1.20**, PLP-dependent lyase; crystal structure PDB `8VHH`).
- **Biochemical role of positions 104–106 and epistatic discovery of `ACC`:** In the parent `Tm9D8*` (`AET`, $\text{fitness} = 0.7278$), `Glu105` H-bonds with the indole substrate. Mutating `Glu105` alone in the `A_T` context nearly always destroys activity—except when substituted by `Cys105` (`ACT`, $\text{fitness} = 0.8583$), and combining `Cys105` with `Cys106` (`ACC`, $\text{fitness} = 1.0000$) produces the global optimum of the landscape, illustrating strong positive epistasis at the catalytic site.
- **Construct representation (`397` aa):** Full-length `Tm9D8*` sequence including the C-terminal `LEHHHHHH` tag (`SI Table S3`, p. 52; 100% canonical amino acids, 1 MMseqs2 cluster at 50% identity).
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** All variants catalyze the condensation of **$200\,\mu\text{M}$ indole** + L-serine $\to$ L-tryptophan with endogenous PLP cofactor in *E. coli* Trp-dropout minimal media.

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Pooled-culture auxotrophic complementation enrichment in *E. coli* BW25113 $\Delta trpAB::\text{Kan}^R$ grown in M9 Trp-dropout media with $200\,\mu\text{M}$ indole and $0.05\%$ arabinose, read out by Illumina HiSeq 2500 sequencing and $\text{OD}_{600}$ at $t = 0, 18, 44\text{ h}$ | No | `reported` | *"We decided to monitor fitness in a similar fashion through a pooled-culture enrichment scheme that connects TrpB activity to growth rate. More-active TrpB variants in an Escherichia coli Trp auxotroph increase in frequency within the population over the course of selection."* `[Source: Main, Page 3]` |
| **Analytical readout** | Specific growth rate $\mu_i = \frac{1}{t}\ln(x_{f,i}/x_{0,i})$ from $\text{OD}_{600} \times$ read frequency; stop-codon subtracted and max-normalized to $1.0$ | No | `reported` | *"where x0,i and xf ,i represent the concentration of E. coli harboring the given amino acid sequence i in the initial population and the population at time t respectively. These values were calculated based on the OD600 of the culture and the frequency of sequence in each population."* `[Source: Main, Page 9]` |
| **Expression system** | Cytoplasmic expression in *E. coli* BW25113 $\Delta trpAB::\text{Kan}^R$ from `pBAD24-Tm9D8*` induced with $0.05\%$ L-arabinose | No | `reported` | *"For the pooled-culture enrichment assays, Tm9D8* was cloned into the arabinose-inducible pBAD24 backbone while for in vitro assays and protein expression for purification, the pET22b(+) backbone was used."* `[Source: Main, Page 8]` |
| **Expression decoupled from label** | **true** (hyperthermostable *T. maritima* `Tm9D8*` scaffold decouples active-site catalytic effects from thermal unfolding at $37\,^\circ\text{C}$) | — | `interpreted` | *"Tm9D8* is also highly thermostable and exhibits high activity at E. coli growth temperatures (34), a useful feature for decoupling a substitution’s contribution to loss of catalytic activity from a loss of stability (4)."* `[Source: Main, Page 3]` |
| **Temperature ($T$)** | **$37\,^\circ\text{C}$** ($250\text{ rpm}$ shaking in $25\text{ mL}$ flasks) | No (constant) | `reported` | *"Cells were incubated in Trp-DO media at 37 °C and 250 rpm in total volumes of 25 (1-, 2-, or 3-site libraries) or 50 mL (4-site library)"* `[Source: Main, Page 9]` |
| **pH & buffer** | 1X M9 minimal salts + $0.74\text{ g/L}$ `-Trp` dropout supplement + $2\text{ mM MgSO}_4$ + $100\,\mu\text{M CaCl}_2$ + $0.4\%$ glycerol (washed in 1X PBS pH 7.4 prior to inoculation) | No (constant) | `reported` | *"Assay media was composed of 1X M9 salts (Sigma Aldrich, Catalog # M6030), 0.74 g/L dropout supplement -Trp (Takara Bio Inc., Catalog # 630413), 2 mM MgSO4, 100 µM CaCl 2, and 0.4% glycerol."* `[Source: SI, Page 5]` |
| **Cofactors & metals** | Endogenous **pyridoxal 5'-phosphate (PLP)** bound to `Lys83` | No (constant) | `reported` | Main p. 2 |
| **Substrate loading** | **$200\,\mu\text{M}$ indole** (from $500\text{ mM}$ DMSO stock) + L-serine + $0.05\%$ arabinose; initial $\text{OD}_{600} = 0.10$ | No (constant) | `reported` | *"choosing final concentrations of 0.05% arabinose (stock concentration 20% in M9) and 200 µM indole (stock concentration 500 mM in DMSO)."* `[Source: SI, Page 5]` |
| **Reaction time** | **44 h** growth time course in Flask A1, sequenced at **$t = 0, 18, 44\text{ h}$** (SI Table S7) | No (constant) | `reported` | SI Table S7 (`si` p. 61) |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation ($8,000 \to 7,971 \to 7,970$)

1. **Theoretical 3-site combinatorial space:** $20^3 = \mathbf{8,000}$ canonical tripeptides at positions `104, 105, 106`.
2. **Upstream sequencing depth filter (`Johnston2024_TrpB3A.csv`):** 29 combinations had $< 10$ reads at $t = 0$ or $0$ reads at $t = 18\text{ h}/44\text{ h}$, leaving **7,971 rows** in `data/raw/dms/trpb3/Johnston2024_TrpB3A.csv`.
3. **The 1-row `"NAN"` string-cleaner drop in [`trpb3_epistasis.py`](../../scripts/data_prep/real_cases/trpb3_epistasis.py#L46-L78):** In `build_trpb3_rows`, `mutation_code = utils.clean_text(row.get("sequences"))` passes each 3-letter amino acid code through `clean_text`, which returns `""` whenever `text.lower() in {"nan", "none", "null"}`. For row 4390 of `Johnston2024_TrpB3A.csv`, the valid tripeptide **`"NAN"` (`Asn104–Ala105–Asn106`, `fitness = -0.063185`)** matches `"nan"` case-insensitively and is skipped, yielding **7,970 unique 397-aa sequences**.

### 4.2 Target transformation

Enzymado applies an **Identity** transform to `fitness` ($(\mu_i - \mu_{\text{stop}})/(\mu_{\text{max}} - \mu_{\text{stop}})$), followed by train-split z-score standardization ($\mu_{\text{train}} = -0.0391$, $\sigma_{\text{train}} = 0.0515$ across the 6,370 training rows; raw range $[-0.5795, +1.0000]$, standardized range $[-10.4910, +20.1665]$, skewness $= +10.22$).

---

## 5. Split semantics and leakage caveats

One split file is published: `trpb3_random_80_10_10_standardized` ($6,370\text{ train} / 800\text{ val} / 800\text{ test} = 7,970\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/trpb3_epistasis/dataset_card.json) for exact split moments).

1. **Higher-order epistasis at seen sites (`random_split_unit="point"`):** All 7,970 variants mutate the same three positions (`104, 105, 106`), so every single substitution is seen $\sim 318$ times in `train` and every double substitution is seen $\sim 16$ times in `train`. The split evaluates **3-body epistatic combination prediction** at positions 104–106.
2. **Extreme right-tail sensitivity ($\sim 0.54\%$ active variants):** Because 99.46% of rows are inactive noise around $-0.045$ (`-0.11` standardized) and the $\sim 43$ active rows reach $+1.000$ (`+20.17` standardized), always supplement $R^2$ with active-hit classification metrics (AUPRC, Top-K recall).
3. **Lexicographical `dataset_id` ordering:** `trpb3_00001` … `trpb3_07970` follow alphabetical tripeptide order (`AAA` to `YYY`) and must never be passed as a feature.
