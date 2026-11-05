---
card_type: dataset
task_id: pyks_solubility
dataset_id: pyks
property_name: pyks_solubility
track: real_case
task_type: protein_global
label_kind: scalar_regression
layout: Layout A
doi: 10.1021/acssynbio.8b00486
license: ACS AuthorChoice / PMC Open Access Manuscript
source_pdfs:
  - docs/pdfs/pyks_wrenbeck2019_acssynbio.pdf (main, pp. 1-27)
  - docs/pdfs/pyks_wrenbeck2019_suppinfo.pdf (si, pp. 1-33)
noise_regime: single_lab_replicated
label_noise_floor: 0.22 Normalized_ER units (synonymous wild-type codon SD from SI Table S4 vs. train SD = 0.3786; analytical R^2_max = 0.6623)
scope_kind: single_protein_mutational_scan
computed_statistics_ref: data/processed/real_case/pyks_solubility/dataset_card.json#/statistics
---

# `pyks_solubility` — Wrenbeck et al. (2019) mGFPmut3-fusion FACS solubility DMS (Supporting Data S1: *Lipomyces starkeyi* LGK)

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/pyks_solubility/dataset_card.json`](../../data/processed/real_case/pyks_solubility/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **Crucial provenance reconciliation (`pyks_solubility` ingests Supporting Data S1 = 439-aa *Lipomyces starkeyi* levoglucosan kinase `LGK`, NOT 389-aa *Atropa belladonna* `PyKS`).** In Wrenbeck et al. (*ACS Synth. Biol.* 2019), the authors generated two C-terminal `mGFPmut3`-fusion FACS solubility DMS datasets: *"Supporting Data S1 and S2 contain full LGK and PyKS GFP-fusion datasets, respectively."* `[Source: Main, Page 20]`. The first half of the paper validates the `mGFPmut3`-fusion solubility screen on the 439-aa yeast enzyme **levoglucosan kinase (`LGK`, UniProt `B3IUN8`, PDB `4ZFV`, EC 2.7.1.214; Supporting Data S1, Tables S1–S4)**, while the second half applies it to the 389-aa plant enzyme pyrrolidine ketide synthase (`PyKS`, Supporting Data S2, Tables S5–S7). Because [`scripts/data_prep/real_cases/pyks_solubility.py`](../../scripts/data_prep/real_cases/pyks_solubility.py) ingests `pyks_wrenbeck2019_NIHMS1058253-supplement-DataS1.csv` (**Supporting Data S1**), all **6,486 sequences in `pyks_solubility` are 439-aa variants of *Lipomyces starkeyi* `LGK`**. Any structure-based model or MSA pipeline must map this task to **`LGK` (`PDB 4ZFV` / `UniProt B3IUN8`)**, never `PyKS`.
2. **The observable is in vivo heterologous soluble expression via C-terminal `mGFPmut3` folding reporter in *E. coli* BL21\*(DE3).** When a variant of the upstream 439-aa enzyme folds solubly in the *E. coli* cytoplasm at $30\,^\circ\text{C}$, the C-terminally fused monomeric GFP (`mGFPmut3`) folds and becomes fluorescent; when the upstream protein misfolds or aggregates, GFP fluorescence is quenched. FACS sorting of the **top 5% brightest $\text{GFP}^+$ cells** relative to the total FSC/SSC reference population produces the wild-type-centered $\log_2$ enrichment score (`normalized_er`, $\text{WT} = 0.0$).
3. **Single-sort 2-gate FACS noise sets an analytical $R^2_{\max}$ ceiling of `0.6623`.** Each of the four mutagenesis tiles (residues `G8–T435`) was sorted once (700,000 cells per gate; SI Table S2). Supporting Table S4 reports that the synonymous wild-type codon distribution has a standard deviation of **`SD = 0.22`** `Normalized_ER` units (SI p. 17)—which is $58.1\%$ of the training target standard deviation ($\sigma_{\text{train}} = 0.3786$), bounding maximum achievable $R^2$ at $1 - 0.22^2 / 0.3786^2 = \mathbf{0.6623}$. Despite this single-variant noise floor, the screen successfully recovers **9 of 12 known thermostabilizing mutations in LGK** ($\Delta T_m \ge 1\,^\circ\text{C}$ from Klesmith et al. 2015) above the $+0.15$ threshold ($p = 2.0 \times 10^{-9}$, Main p. 10, SI Table S3), with the global dataset maximum occurring at the known stabilizing mutant **`H113G` (`+1.7143`, $\Delta T_m = +4.9\,^\circ\text{C}$)**.
4. **Under the point-stratified 80/10/10 split, all 428 mutagenized positions (`G8–T435`) appear $\sim 12.1$ times in `train`.** Splitting at the variant level (`random_split_unit="point"`) evaluates **within-position amino acid substitution interpolation** for soluble expression rather than unseen-site or unseen-scaffold generalization.

**Practical consequences for use:**
- Always map structure/MSA lookups for `pyks_solubility` to ***Lipomyces starkeyi* levoglucosan kinase (`LGK`, 439 aa, `UniProt B3IUN8`, `PDB 4ZFV`)**.
- Normalize regression $R^2$ against the synonymous-codon noise ceiling **$R^2_{\max} \approx 0.662$**: an $R^2$ of $0.45\text{–}0.55$ represents $68\text{–}83\%$ of explainable variance.
- Remember the **stability–activity trade-off** central to Wrenbeck et al. (2019): mutations that remove conserved catalytic residues or helix-breaking prolines frequently boost `mGFPmut3` solubility (`normalized_er > 0.15`) while destroying catalytic activity.
- Never pass `dataset_id` (`pyks_00001` … `pyks_06486`, ordered monotonically by sequence position) as a model input.

---

## 1. Label noise and the measurement floor

### 1.1 Experimental quality controls and noise anchors

| Property | Value | Evidence |
|---|---|---|
| **FACS sorting design** | **700,000 cells collected per gate** across 4 tiles (FSC/SSC reference gate vs. top 5% $\text{GFP}^+$ gate; actual sort fractions $4.7\%, 4.7\%, 4.8\%, 4.3\%$ in SI Table S2) | *"700,000 cells each from two populations were collected for each sample: a reference population - forward scatter (FSC) vs. side scatter (SSC) gate - and a selected population (intersection of FSC vs. SSC gate and either the top 5% (LGK) or top 8-10% (PyKS) based on GFP fluorescence intensity with a 530/40 nm filter [488 nm])."* `[Source: SI, Page 3]` |
| **Deep sequencing depth** | **$1.03\text{M–}1.99\text{M}$ post-filter reads per gate** on Illumina MiSeq 300 bp PE (SI Table S1, p. 14) | Supporting Table S1 (`si` p. 14) |
| **Synonymous wild-type codon dispersion** | **$\text{SD}_{\text{syn}} = 0.22$** `Normalized_ER` units ($0.5811$ standardized SD units) | *"The distribution of synonymous wild-type codons for the GFP-fusion solubility screen has a standard deviation of 0.22."* `[Source: SI, Page 17]` |
| **Analytical $R^2_{\max}$ ceiling** | **$R^2_{\max} = 1 - \text{SD}_{\text{syn}}^2 / \text{Var}(y) = 1 - 0.0484 / 0.1433 = \mathbf{0.6623}$** | Computed from $\text{SD}_{\text{syn}} = 0.22$ (SI Table S4) and $\sigma_{\text{train}} = 0.3786$ |
| **Biophysical $\Delta T_m$ validation (SI Table S3)** | **9 of 12** known stabilizing LGK mutations ($\Delta T_m \ge 1\,^\circ\text{C}$) score $> 0.15$ ($p = 2.0 \times 10^{-9}$), led by `H113G` ($\Delta T_m = +4.9\,^\circ\text{C}$, score $= +1.7143$) and `Q369L` ($\Delta T_m = +3.4\,^\circ\text{C}$, score $= +1.4882$) | *"We identified a mutation as stabilizing if its solubility score was above 0.15, which corresponds to a mean fluorescence intensity of 10% above the wild-type sequence. The GFP fusion screens with this threshold identified 9/12 of these mutations (p-value 2.0x10-9)."* `[Source: Main, Page 10]` |

### 1.2 Factors governing the error structure

1. **Two-gate top-5% FACS enrichment:** Collecting only the top 5% brightest $\text{GFP}^+$ cells vs. the unsorted reference gate is optimized for discovering gain-of-solubility mutations ($\text{Normalized\_ER} > 0.15$), but provides coarse resolution between neutral and moderately destabilized variants, yielding $\text{SD}_{\text{syn}} = 0.22$ vs. total $\sigma = 0.3786$.
2. **Four independent tiles (`G8–T435`):** Tile 1 (`8–110`), Tile 2 (`111–220`), Tile 3 (`221–330`), and Tile 4 (`331–435`) were sorted and sequenced separately, with each tile centered at its own internal wild-type enrichment score of `0.0` (Main p. 10, SI Tables S1–S2).
3. **Cell-to-cell plasmid/induction variance in *E. coli* BL21\*(DE3):** T7lac induction with $250\,\mu\text{M}$ IPTG in Terrific Broth at $30\,^\circ\text{C}$ exhibits single-cell expression noise, partially averaged over the 700,000 sorted events per gate.

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`single_protein_mutational_scan`):** **6,486 unique sequences** (1 wild-type + 6,485 single-site missense variants across residues `G8–T435`) of the 439-aa ***Lipomyces starkeyi* levoglucosan kinase (`LGK`)** (UniProt `B3IUN8`, **EC 2.7.1.214**, crystal structure PDB `4ZFV`; Supporting Data S1 of Wrenbeck et al. 2019).
- **Mutagenized window (`G8–T435`, 428 of 439 positions):** Nicking mutagenesis used degenerate `NNN` primers across residues `G8` to `T435` (`si` p. 2); N-terminal residues `1–7` (`MPSVDTI`) and C-terminal residues `436–439` (`KKLA`) were left unmutated.
- **Construct representation (`439` aa):** Experimentally expressed in `pET29NK_LGK/mGFPmut3` with a C-terminal `mGFPmut3` fusion tag; stored in Enzymado as the 439-aa untagged LGK sequence (100% canonical amino acids, 1 MMseqs2 cluster at 50% identity).
- **Chemical & condition scope (`has_ligand_smiles: false`, Layout A):** Assayed in *E. coli* BL21\*(DE3) without exogenous catalytic substrates (levoglucosan or ATP).

---

## 3. Experimental measurement and STRENDA protocol

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | Heterologous protein-solubility DMS via C-terminal `mGFPmut3` folding reporter and BD Influx FACS sorting in *E. coli* BL21\*(DE3), collecting 700,000 cells in the FSC/SSC reference gate and 700,000 cells in the top-5% $\text{GFP}^+$ gate followed by Illumina MiSeq sequencing | No | `reported` | *"Individual cells were sorted using FACS and two populations were collected: a gated reference population and the top 5% of cells based on GFP fluorescence intensity. Libraries were harvested, prepared, and deep sequenced in a standardized pipeline26."* `[Source: Main, Page 9]` |
| **Analytical readout** | BD Influx FACS ($488\text{ nm}$ laser, $530/40\text{ nm}$ filter) + 300 bp PE Illumina MiSeq sequencing ($1.03\text{M–}1.99\text{M}$ reads/gate) + Enrich $\log_2$ enrichment score centered at $\text{WT} = 0.0$ | No | `reported` | *"The resulting deep sequencing datasets were converted to a solubility score centered about a wild-type score of zero."* `[Source: Main, Page 10]` |
| **Expression system** | Intracellular cytoplasmic expression of `pET29NK_LGK/mGFPmut3` in *E. coli* BL21\*(DE3) in Terrific Broth (TB) + $50\,\mu\text{g/mL}$ kanamycin, induced with $250\,\mu\text{M}$ IPTG | No | `reported` | *"Nicking mutagenesis was used to generate a comprehensive single-site mutagenesis libraries on the pET29NK_PyKS-RD/mGFPmut3 and pET29NK_LGK/mGFPmut3 plasmids4. Mutagenic primers encoding degenerate bases (NNN) were used for residues G8 to T435 for LGK ."* `[Source: SI, Page 2]` |
| **Expression decoupled from label** | **false** (soluble heterologous expression of the GFP fusion is the target observable) | — | `reported` | *"Upon expression, folded proteins will permit the folding and subsequent chromophore formation of GFP, while unfolded proteins will be nonfluorescent23."* `[Source: Main, Page 9]` |
| **Temperature ($T$)** | **$30\,^\circ\text{C}$** during growth and IPTG induction ($250\text{ rpm}$); washed in cold sterile PBS prior to FACS sorting | No (constant) | `reported` | *"For each library, 3 mL TB cultures were inoculated to an initial OD 600=0.05 in Hungate tubes and grown at 30°C with 250 rpm shaking."* `[Source: SI, Page 3]` |
| **pH & buffer** | Terrific Broth during induction (*E. coli* cytoplasm $\text{pH } \sim 7.5$); washed twice in cold sterile PBS ($\text{pH } 7.4$) for FACS sorting | No (constant) | `reported` | *"Once the cultures reached an OD 600=0.25-0.3, 1 mL was pelleted and washed with cold sterile PBS twice."* `[Source: SI, Page 3]` |
| **Cofactors & metals** | None added in the solubility screen | No | `not_reported` | — |
| **Readout window** | Induced with $250\,\mu\text{M}$ IPTG at $\text{OD}_{600} = 0.05\text{–}0.08$ (after 4–5 pre-induction generations from $\text{OD}_{600} = 0.0025$) and harvested at $\text{OD}_{600} = 0.25\text{–}0.3$ | No (constant) | `reported` | *"After approximately 4-5 generations (OD 600=0.05-0.08) IPTG was added to a final concentration of 250 µM. Once the cultures reached an OD 600=0.25-0.3, 1 mL was pelleted and washed with cold sterile PBS twice."* `[Source: SI, Page 3]` |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Exact row reconciliation from `DataS1.csv` to 6,486 rows

The raw file `data/raw/dms/pyks/pyks_wrenbeck2019_NIHMS1058253-supplement-DataS1.csv` (**Supporting Data S1**) contains a complete $439\text{ positions (0-indexed `Location` 0..438)} \times 21\text{ symbols (20 AA + `*`)} = \mathbf{9,219\text{ rows}}$ grid:
- **439 synonymous wild-type rows** (`Normalized_ER == 0.0`, one at each `Location` 0..438, which [`pyks_solubility.py`](../../scripts/data_prep/real_cases/pyks_solubility.py#L22-L47) uses to reconstruct the exact 439-aa wild-type *Lipomyces starkeyi* LGK sequence).
- **6,485 finite missense rows** (`Amino Acid != WT_AA` and `Amino Acid != '*'`).
- **280 finite nonsense (`*`) rows** (note that $6,485\text{ missense} + 280\text{ nonsense} = \mathbf{6,765\text{ finite non-synonymous + nonsense variants}}$, matching the sum of observed mutations in Supporting Table S1!).
- **2,015 `NA` rows** (1,856 missense + 159 `*`, including all non-WT rows at the unmutated N-terminal positions `0..6` and C-terminal positions `435..438`).

In [`scripts/data_prep/real_cases/pyks_solubility.py`](../../scripts/data_prep/real_cases/pyks_solubility.py#L50-L92), `build_pyks_rows` drops the 439 stop-codon (`*`) rows and 1,856 missense `NA` rows, constructs the 439-aa sequences, and collapses the 439 identical wild-type rows (`Normalized_ER == 0.0`) into **1 wild-type row** (`pyks_00001`), yielding **$1\text{ WT} + 6,485\text{ missense} = \mathbf{6,486\text{ unique sequences}}$**.

### 4.2 Target transformation

Enzymado applies an **Identity** transform to `normalized_er` (`Normalized_ER`), followed by train-split z-score standardization ($\mu_{\text{train}} = -0.3688$, $\sigma_{\text{train}} = 0.3786$ across the 5,186 training rows; raw range $[-1.6840, +1.7143]$, skewness $= +0.072$).

---

## 5. Split semantics and leakage caveats

One split file is published: `pyks_random_80_10_10_standardized` ($5,186\text{ train} / 650\text{ val} / 650\text{ test} = 6,486\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/pyks_solubility/dataset_card.json) for exact split moments).

1. **Protein identity caveat (`LGK` vs. `PyKS`):** `pyks_solubility` contains the 439-aa *Lipomyces starkeyi* levoglucosan kinase (`LGK`, UniProt `B3IUN8`, PDB `4ZFV`) dataset from Supporting Data S1, not the 389-aa *Atropa belladonna* `PyKS` dataset from Supporting Data S2.
2. **Position-level sharing across splits (`random_split_unit="point"`):** Every mutagenized residue position (`8–435`) is sampled $\sim 12.1$ times in `train` and $\sim 1.5$ times in `val` and `test`; the split evaluates within-position substitution interpolation.
3. **Synonymous noise ceiling ($R^2_{\max} \approx 0.662$):** Because $\text{SD}_{\text{syn}} = 0.22$ (`0.581` standardized SD units), models cannot exceed $R^2 \approx 0.66\text{–}0.70$ without fitting sorting noise.
4. **Identifier leakage risk:** `dataset_id` (`pyks_00001` … `pyks_06486`) is ordered monotonically by sequence position and must never be used as a feature.
