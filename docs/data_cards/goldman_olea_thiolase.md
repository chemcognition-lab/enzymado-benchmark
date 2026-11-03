---
card_type: dataset
task_id: goldman_olea_thiolase
dataset_id: goldman_olea_thiolase
property_name: goldman_olea_thiolase
track: real_case
task_type: protein_ligand_global
label_kind: scalar_regression
layout: Layout B
doi: 10.1093/synbio/ysaa004
license: CC BY 4.0
source_pdfs:
  - docs/pdfs/olea_robinson2020_synbio.pdf (main, pp. 1-13)
  - docs/pdfs/goldman2022_ploscompbiol.pdf (main / review, pp. 1-20)
  - docs/pdfs/olea_robinson2020_suppinfo.pdf (si, pp. 1-5)
  - docs/pdfs/goldman2022_supp_s1_text.pdf (si, pp. 1-18)
noise_regime: single_lab_replicated
label_noise_floor: 0.1990 log10 units median within-pair replicate SD (mean SD = 0.2208, 1.58-fold error across 262 multi-replicate pairs in enzyme_substrate_activity_data.csv; analytical R^2_max = 0.7282 for replicate-averaged target, or 0.5685 for single replicate)
scope_kind: family_wide_screen
computed_statistics_ref: data/processed/real_case/goldman_olea_thiolase/dataset_card.json#/statistics
---

# `goldman_olea_thiolase` — Bacterial `OleA` thiolase family-wide substrate specificity screen (73 homologs $\times$ 14 active *p*-nitrophenyl ester substrates)

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/goldman_olea_thiolase/dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/goldman_olea_thiolase/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is a family-wide enzyme homolog $\times$ substrate matrix restricted to the 550 verified active pairs (`Layout B`: 550 unique `(protein_sequence, smiles)` rows across 73 bacterial `OleA` homologs and 14 active *p*-nitrophenyl ester substrates).** Smith et al. (*mBio* 2020) and Robinson et al. (*Synthetic Biology* 2020) screened **73 phylogenetically diverse bacterial `OleA` thiolase homologs** (spanning 7 bacterial phyla and **49 MMseqs2 clusters at 50% identity**; lengths $312\text{–}371\text{ aa}$) against **15 *p*-nitrophenyl (`pNP`) alkanoate/arylalkanoate substrates** ($73 \times 15 = 1,095$ total pairs; repackaged by Goldman et al., *PLOS Comput. Biol.* 2022). However, the raw quantitative rate table (`data/raw/olea/enzyme_substrate_activity_data.csv`, 938 replicate rows) records **only the positive-activity replicate wells** whose maximum 15-minute rolling-window slope achieved **$R^2 \ge 0.9$**. Averaging those 938 valid replicate measurements by `(protein_sequence, smiles)` in [`goldman_olea.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/goldman_olea.py#L75-L120) produces the **550 unique active enzyme–substrate pairs** in `goldman_olea_thiolase` (and omits the 15th tested substrate, `clorophene`, which had 0 active wells across all 73 enzymes).
2. **Because `enzyme_substrate_activity_data.csv` preserves all 938 uncollapsed replicate wells, the experimental noise ceiling is known exactly ($R^2_{\max} = \mathbf{0.7282}$).** Across the 550 active pairs, 288 were measured in $n = 1$ valid replicate (passing $R^2 \ge 0.9$), 137 in $n = 2$ replicates, 124 in $n = 3$ replicates, and 1 in $n = 4$ replicates ($288 + 274 + 372 + 4 = 938$). Across the 262 multi-replicate pairs ($n \ge 2$), the within-pair biological replicate standard deviation on `log_slope` ($\log_{10}(\text{nmol } p\text{NP} / \text{OD}_{1.0} / \text{h})$) has **median $\text{SD} = 0.1990\text{ log}_{10}\text{ units}$** ($\text{mean SD} = 0.2208$, $1.58\text{-fold}$ error; mean within-pair variance $\overline{\sigma_{\text{rep}}^2} = 0.0709$). Accounting for the replicate count $n_i \in \{1, 2, 3, 4\}$ per pair gives a mean squared standard error of $\overline{\text{SEM}^2} = 0.0447$ against the training variance $\sigma_{\text{train}}^2 = 0.4053^2 = 0.1643$, yielding an analytical ceiling of **$R^2_{\max} = 1 - 0.0447 / 0.1643 = \mathbf{0.7282}$** (or $R^2_{\max} = 0.5685$ for single-replicate predictions).
3. **It measures a whole-cell chromogenic surrogate esterase reaction (`EC 3.1.1.-`) of a condensing thiolase (`EC 2.3.1.-`).** Physiologically, `OleA` catalyzes the non-decarboxylative Claisen condensation of two long-chain acyl-CoA thioesters to initiate microbial olefin and $\beta$-lactone biosynthesis. Smith et al. (2020) discovered that `OleA` also rapidly hydrolyzes *p*-nitrophenyl esters via acylation of its catalytic **`Cys143`** followed by water-mediated deacylation, allowing activity to be read out in intact *E. coli* BL21(DE3) cells at $410\text{ nm}$. Because the assay is performed in whole cells normalized by culture $\text{OD}_{600}$, heterologous soluble expression and cell-wall permeation of hydrophobic substrates are coupled into `log_activity`.
4. **Three critical sequence and chemical bugs in the upstream deposit are repaired in [`goldman_olea.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/goldman_olea.py#L43-L72):**
   - Strips the 20-aa N-terminal expression tag (`MGSSHHHHHHSSGLVPRGSH`) from the *Xanthomonas campestris* `4KU5` sequence (`358 aa` $\to$ `338 aa`);
   - Restores the wild-type catalytic nucleophile **`Cys143`** in `4KU5` (`FDVANASLAF` $\to$ `FDVANACLAF`), which had been mutated to `Ala143` in the crystal structure PDB `4KU5`;
   - Corrects the substrate name typo `"oxidazole"` $\to$ `"oxadiazole"` (*p*-nitrophenyl 3-(4-phenyl-1,2,5-oxadiazol-3-yl)propanoate), rescuing 26 replicate rows (14 unique pairs).

**Practical consequences for use:**
- **Do not confuse this 550-row continuous rate dataset with the 1,095-row binary classification matrix:** Models trained on `goldman_olea_thiolase` predict quantitative turnover rate *conditional on the pair being active* ($R^2 \ge 0.9$).
- **Evaluate models relative to $R^2_{\max} = 0.728$:** A model reaching $R^2 = 0.60\text{–}0.68$ is already capturing $82\%\text{–}93\%$ of the explainable variance.
- Never pass `dataset_id` (`goldman_olea_00001` … `goldman_olea_00550`) as a model feature.

---

## 1. Label noise and the measurement floor

### 1.1 Exact replicate noise analysis from the 938 raw measurements

| Property | Value | Evidence |
|---|---|---|
| **Biological replicate design** | Assayed in **biological triplicate** in 96-well plates; replicates passing $R^2 \ge 0.9$ in a 15-min rolling window were retained (`938` rows across `550` pairs: $288\times 1$, $137\times 2$, $124\times 3$, $1\times 4$; mean $1.71$ replicates/pair) | *"Slopes were calculated using a rolling linear regression window method by calculating slopes for all overlapping 15-min intervals over the course of the first 45 min of each reaction. The greatest slope for each enzyme with R2 ≥ 0.9 was selected as maximum enzyme activity for a given OleA (nanomoles of p-NP/OD of 1.0/hour)."* `[Source: Main, Page 11]` |
| **Within-pair replicate SD ($n = 262$ multi-replicate pairs)** | **Median $\text{SD} = 0.1990\text{ log}_{10}\text{ units}$** ($\text{IQR} = [0.0975, 0.3082]$), **mean $\text{SD} = 0.2208\text{ log}_{10}\text{ units}$** ($1.58\text{-fold}$ error), pooled RMS $\text{SD} = 0.2662\text{ log}_{10}\text{ units}$ | Computed directly across the 262 multi-replicate `(protein_sequence, smiles)` pairs in `data/raw/olea/enzyme_substrate_activity_data.csv` |
| **Analytical $R^2_{\max}$ ceiling (replicate-averaged target)** | **$R^2_{\max} = 1 - \overline{\text{SEM}^2} / \sigma_{\text{train}}^2 = 1 - 0.0447 / 0.1643 = \mathbf{0.7282}$** (single-replicate ceiling: $1 - 0.0709 / 0.1643 = \mathbf{0.5685}$) | Exact variance decomposition from the 938 raw replicate rows vs. train split variance ($\sigma_{\text{train}} = 0.4053$) |
| **Published 10-fold CV regression RMSE (Robinson et al. 2020)** | Random Forest $\text{RMSE} = 0.453 \pm 0.056$, FFNN $\text{RMSE} = 0.506 \pm 0.056$, Elastic Net $\text{RMSE} = 0.525 \pm 0.064$ (on full $1,095$-pair imputed dataset) | *"For regression models, the random forest RMSE was 0.453 ± 0.056, the feedforward neural network RMSE was 0.506 ± 0.056 and the elastic net RMSE was 0.525 ± 0.064."* `[Source: SI, Page 3]` |

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`family_wide_screen`):** **550 unique active `(protein_sequence, smiles)` pairs** spanning **73 bacterial `OleA` thiolase homologs** ($312\text{–}371\text{ aa}$, median $341\text{ aa}$; **EC 2.3.1.-** condensing thiolase / **EC 3.1.1.-** promiscuous *p*-nitrophenyl ester hydrolase) and **14 active *p*-nitrophenyl ester substrates**.
- **Phylogenetic breadth (7 bacterial phyla, 49 MMseqs2 clusters at 50% identity):**
  - Mined from 251 genomic `oleA` sequences flanked by `oleBCD` hydrocarbon/$\beta$-lactone biosynthetic genes, filtered to $\le 500\text{ aa}$ (`234` sequences), and clustered via CD-HIT at 50% identity into 41 cluster representatives plus 32 additional diverse homologs (`Main p. 11`).
  - Whole-dataset MMseqs2 clustering at 50% sequence identity yields **49 clusters across the 73 proteins** (`ratio_unique_variants = 49 / 73 = 0.6712`), providing genuine family-wide evolutionary diversity.
- **Chemical scope (`has_ligand_smiles: true`, Layout B):** **14 active *p*-nitrophenyl (`pNP`) carboxylic esters** (out of 15 in `pNP_substrates.csv`): linear $\text{C}_6\text{–C}_{14}$ alkanoates (`hexanoate`, `heptanoate`, `octanoate`, `nonanoate`, `decanoate`, `dodecanoate`, `myristate`), branched `trimethylhexanoate`, cyclic `cyclohexylpropanoate`, $\omega$-bromo esters (`6-bromohexanoate`, `7-bromoheptanoate`), aryl esters (`phenylpropanoate`, `phenoxypropanoate`), and heterocyclic `biotin` and `oxadiazole` esters. The 15th tested substrate (`clorophene`, *p*-nitrophenyl 2-(4-chloro-2-methylphenoxy)propanoate) was inactive across all 73 enzymes (`Smith et al. 2020 Fig. 3A`, `Main p. 5`).

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Whole-cell 96-well microplate colorimetric *p*-nitrophenyl ester hydrolysis assay in *E. coli* BL21(DE3) read at $410\text{ nm}$ | No | `reported` | *"The high rates allowed activity to be determined in vitro and with heterologously expressed OleA in vivo via the release of the yellow p-nitrophenol product."* `[Source: Main, Page 1]` |
| **Analytical readout** | Absorbance at $410\text{ nm}$ read every $1\text{ min}$ for $60\text{ min}$, converted via $\varepsilon_{410} = 15,546\,\text{M}^{-1}\text{cm}^{-1}$ to $\text{nmol } p\text{NP} / \text{OD}_{1.0} / \text{h}$ from the steepest 15-min rolling window with $R^2 \ge 0.9$ | No | `reported` | *"Absorbance was converted to nanomoles of p-NP produced by an E. coli BL21 culture with an OD of 1.0 using the Beer-Lambert law (ε410 = 15,546 M−1 cm−1)."* `[Source: Main, Page 11]` |
| **Expression system** | Heterologous cytoplasmic expression in *E. coli* BL21(DE3) from `pET-28b+` induced with $100\,\mu\text{M}$ IPTG at $\text{OD}_{600} = 0.3$, grown overnight at $16\,^\circ\text{C}$, resuspended at $\text{OD}_{600} = 0.1$ and diluted 16-fold | No | `reported` | *"E. coli BL21(DE3) cells were transformed with either empty pET-28b+ vector or the vector containing an oleA gene. Cells were induced with 100 μM isopropyl-β-D-thiogalactopyranoside when the absorbance reached 0.3 (600 nm). Induced cells were grown overnight at 16°C followed by resuspension in 50 mM Tris-HCl (pH 8.0) buffer at 0.1 OD."* `[Source: Main, Page 11]` |
| **Expression decoupled from label** | **false** (whole-cell in vivo rate normalized by culture $\text{OD}_{600}$, not purified protein concentration) | — | `reported` | Main p. 1, p. 11 |
| **Temperature ($T$)** | **$37.0\,^\circ\text{C}$** | No (constant) | `reported` | *"Absorbance was read every minute for 60 min at 410 nm at 37°C."* `[Source: Main, Page 11]` |
| **pH & buffer** | **pH 8.0**, **$50\text{ mM}$ Tris-HCl buffer** with **$5\%\text{ v/v}$ ethanol** ($200\,\mu\text{L}$ total volume) | No (constant) | `reported` | *"Resuspended cells were then diluted 16-fold and added to a 96-well microtiter plate containing 5% ethanol and 200 μM p-nitrophenyl alkanoate for a total volume of 200 μl."* `[Source: Main, Page 11]` |
| **Cofactors & metals** | None required (covalent nucleophilic catalysis via `Cys143–His285–Asn315` triad) | No (constant) | `reported` | Main p. 5 |
| **Substrate loading** | **$200\,\mu\text{M}$ *p*-nitrophenyl ester substrate** (in $5\%$ ethanol) + whole cells at final $\text{OD}_{600} = 0.00625$ | **Yes (`smiles`)** | `reported` | Main p. 11 |
| **Reaction time** | **60 min** total readout (steepest 15-min rolling window within the first 45 min with $R^2 \ge 0.9$) | No (constant) | `reported` | Main p. 11 |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation ($1,095 \to 938 \to 550$) and three bug fixes

1. **Full experimental grid ($73 \times 15 = \mathbf{1,095}$ pairs):** 73 `OleA` homologs screened in triplicate against 15 *p*-nitrophenyl ester substrates (`Goldman et al. 2022 Table 1`).
2. **Active replicate gate ($R^2 \ge 0.9 \to \mathbf{938}$ replicate rows):** `data/raw/olea/enzyme_substrate_activity_data.csv` retains only individual replicate wells where the steepest 15-min window achieved $R^2 \ge 0.9$ (`938` rows across 73 enzymes and 14 substrates; `clorophene` had 0 active rows).
3. **Three curation fixes in [`goldman_olea.py`](file:///home/beangoben/github/enzymado-benchmark/scripts/data_prep/real_cases/goldman_olea.py#L43-L72):**
   - **`4KU5` N-terminal His-tag removal:** Strips `MGSSHHHHHHSSGLVPRGSH` (`358 aa` $\to$ `338 aa`);
   - **`4KU5` catalytic `Cys143` restoration:** Replaces the crystallographic active-site knockout motif `FDVANASLAF` (`Ala143`) with wild-type `FDVANACLAF` (`Cys143`);
   - **Substrate spelling fix (`"oxidazole"` $\to$ `"oxadiazole"`):** Matches `enzyme_substrate_activity_data.csv` (`oxidazole`, 26 replicate rows / 14 unique pairs) to `pNP_substrates.csv` (`oxadiazole`).
4. **Replicate averaging (`938` $\to$ `550` unique active pairs):** `collapse_duplicate_rows` averages `log_slope` across the 1 to 4 valid replicates per `(protein_sequence, smiles)` pair, yielding **550 rows**.

### 4.2 Target transformation

$$\text{log\_activity} = \frac{1}{n_i}\sum_{r=1}^{n_i} \log_{10}\!\left(\text{Activity}_{i,r}\;[\text{nmol } p\text{NP} / \text{OD}_{1.0} / \text{h}]\right)$$

followed by train-split z-score standardization ($\mu_{\text{train}} = 1.9863$, $\sigma_{\text{train}} = 0.4053$ across the 401 training rows; raw range $[1.1476, 3.3160]$, standardized range $[-2.0691, +3.2804]$).

---

## 5. Split semantics and leakage caveats

One split file is published: `goldman_olea_thiolase_random_80_10_10_standardized` ($401\text{ train} / 74\text{ val} / 75\text{ test} = 550\text{ total}$; see [`dataset_card.json`](file:///home/beangoben/github/enzymado-benchmark/data/processed/real_case/goldman_olea_thiolase/dataset_card.json) for exact split moments).

1. **Matrix-completion split (`random_split_unit="point"`):** 72 of the 73 enzymes and all 14 active substrates appear in `train`. Evaluating out-of-distribution generalization to unseen homologs or unseen substrates requires cold-protein or cold-ligand splits.
2. **Active-only conditioning:** Inactive pairs (545 of the 1,095 tested pairs) are absent from this regression dataset.
3. **Sequential `dataset_id` ordering:** `goldman_olea_00001` … `goldman_olea_00550` follow organism and substrate order and must never be passed as a feature.
