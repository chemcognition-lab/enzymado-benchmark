---
card_type: dataset
task_id: erickson_petase
dataset_id: erickson
property_name: erickson_petase
track: real_case
task_type: protein_global
label_kind: scalar_regression
layout: Layout A + numeric conditions
doi: 10.1038/s41467-022-35237-x
license: CC BY 4.0
source_pdfs:
  - docs/pdfs/erickson_petase_2023_natcommun.pdf (main, pp. 1-15)
  - docs/pdfs/erickson_petase_2023_suppinfo.pdf (si, pp. 1-47; combined pp. 16-62)
noise_regime: single_lab_replicated
label_noise_floor: unquantified_in_grid (n=3 triplicate means published without per-cell SD; 1-200 mg/L calibration R^2 >= 0.995)
scope_kind: family_wide_homolog_panel
computed_statistics_ref: data/processed/real_case/erickson_petase/dataset_card.json#/statistics
---

# `erickson_petase` — PET hydrolase activity across temperature, buffer, and pH

> **Computed statistics pointer:** Tabular counts, sequence-length quantiles, and per-split label distributions are maintained programmatically in [`data/processed/real_case/erickson_petase/dataset_card.json`](../../data/processed/real_case/erickson_petase/dataset_card.json). This card documents the **non-computable experimental, curation, and methodological provenance** required to use and interpret the dataset correctly.

## 0. How to think about this dataset

Four structural properties govern how this dataset behaves in machine learning benchmarks:

1. **It is a $19 \times 30$ factorial grid, not 570 independent proteins.** Nineteen natural PET hydrolase scaffolds are each measured across all thirty combinations of five temperatures ($30, 40, 50, 60, 70\,^\circ\text{C}$) and six buffer/pH conditions. Every row belonging to a given scaffold carries the exact same amino acid sequence; only the reaction condition varies across its 30 rows. For any purely sequence-level question, the sample size is **$N = 19$ scaffolds**, not 570.
2. **Under the published point-stratified split, the task tests condition-surface interpolation across seen scaffolds.** Because [`scripts/data_prep/generate_real_case_splits.py`](../../scripts/data_prep/generate_real_case_splits.py) splits real-case datasets at the observation level (`random_split_unit="point"`, stratified across 10 target deciles into $456\text{ train} / 57\text{ val} / 57\text{ test}$ rows), each of the 19 scaffolds has on average **24 of its 30 reaction conditions in `train`, 3 in `val`, and 3 in `test`**. A model that identifies which of the 19 scaffolds it is looking at and learns the smooth temperature $\times$ pH response surface will achieve high pooled $R^2$ without learning how to generalise to a 20th unseen scaffold.
3. **The label is a 96-hour solid-polymer endpoint yield, not a steady-state rate constant.** Cumulative aromatic product concentration released from a solid amorphous PET film over four days conflates intrinsic interfacial catalysis, polymer surface adsorption, and four-day thermal denaturation (which dominates the $60\text{–}70\,^\circ\text{C}$ rows). This observable should never be equated with or jointly modelled alongside solution $k_{\text{cat}}$.
4. **The 19-protein panel is survivorship-filtered three times over.** Starting from 74 bioinformatically mined thermophilic candidates $\to$ 51 that expressed solubly in *E. coli* $\to$ the **19 best-performing scaffolds** across seven phylogenetic groups were chosen for the 30-condition grid. Dead and poorly expressing scaffolds are systematically absent.

**What this dataset is uniquely suited for:** Every measurement comes from a **single coordinated study** using one substrate lot, one purification workflow, and one UHPLC-DAD method in triplicate, with no-enzyme background below detection across all 30 conditions. The multi-lab protocol heterogeneity that dominates database-aggregated kinetics tasks is completely absent here.

**Practical consequences for use:**
- Treat scores on `erickson_random_80_10_10_standardized` as **condition-response interpolation** on a 19-enzyme panel; if your claim is about unseen-scaffold generalisation, you must re-split grouped by `protein_sequence` (keeping in mind that a 10% test split is only $\sim 2$ proteins).
- Report metrics split by **detectable ($> 0\text{ mg/L}$) vs. non-detectable ($0\text{ mg/L}$)** cells: $\sim 20\%$ of the grid sits at the exact zero floor ($-1.328$ standardized).
- Never pass `dataset_id` as a feature (it encodes `<enzyme_id>_<temperature>_<condition>` and is flagged in `categorical_high_variance_columns`).
- Do not transfer learned condition optima to other PET morphologies: the paper demonstrates that optimal temperature/pH shifts on powders and that enzymes `503`, `602`, and `711` invert the expected amorphous-film vs. crystalline-powder preference.

---

## 1. Label noise and the measurement floor

Unlike multi-lab database aggregates, experimental noise here is **procedural and bounded within a single lab**, but its per-cell replicate spread is **not retained in the published screening matrix**.

### 1.1 Experimental quality controls and replicate structure

| Property | Value | Evidence |
|---|---|---|
| **Replicate design** | **$n = 3$ reactions per cell**; published label is the triplicate mean | *"All PET hydrolysis screening reactions were performed in triplicate."* `[Source: Main, Page 11]` |
| **Dispersion in published grid** | **None in `values.csv`** — means only | *"is reported as the average of reactions performed in triplicate (n = 3)"* `[Source: Main, Page 4]` |
| **No-enzyme background** | Below detection limit ($< 1\text{ mg/L}$) at all 30 conditions | *"Background aromatic product release for both amorphous PET film and crystalline PET powder was below the detection limit for all pH and temperature combinations tested."* `[Source: Main, Page 11]` |
| **UHPLC calibration range** | **$1\text{–}200\text{ mg/L}$** per analyte, 10 standards, $R^2 \ge 0.995$ | *"The calibration curve for each analyte was evaluated between concentrations of 1–200 mg/L with DAD detection at a wavelength of 240 nm."* `[Source: Main, Page 11]` |
| **Calibration verification** | Standard injected every 12–24 samples; autosampler at $15\,^\circ\text{C}$ | Main p. 11 |

### 1.2 Three factors that govern error structure

1. **Replicate dispersion is discarded in the 30-condition matrix.** Each of the 570 labels is the mean of three reactions whose individual spread is not carried into `values.csv` (standard deviations *are* plotted for the separate 168 h time-course experiments in Supplementary Figs. 5–8, which are not part of this dataset). Consequently, no per-row standard deviation is available to compute an exact analytical $R^2_{\max}$ ceiling or weight the regression loss.
2. **Dynamic range exceeds the UHPLC calibration curve by $>13\times$.** Direct DAD calibration covers $1\text{–}200\text{ mg/L}$, whereas `sum_aromatic_products` reaches $\sim 2,700\text{ mg/L}$. High-activity wells therefore rely on volumetric sample dilution prior to injection: *"Samples were diluted with ultrapure water for analysis and maintained at 15 °C during the analysis."* `[Source: Main, Page 11]`. Dilution error is multiplicative on the raw concentration scale, making raw variance strongly heteroscedastic at the top of the range—a physical justification for the $\log_{10}$ transformation independent of distributional skewness.
3. **Zeros are left-censored at the $1\text{ mg/L}$ LLOQ, not measured zeros.** A reported `0.0` means product release fell below the $1\text{ mg/L}$ calibration floor, not that zero ester bonds were cleaved. Roughly $20\%$ of cells are zeros, all mapping to the exact standardized minimum **$-1.3280$** across `train`, `val`, and `test`. Because the $+1.0\text{ mg/L}$ offset in $\log_{10}(x + 1.0)$ equals the $1\text{ mg/L}$ LLOQ, it compresses the bottom of the scale ($0\text{ mg/L}$ and $1\text{ mg/L}$ differ by $0.301$ log units).

### 1.3 Confounders specific to heterogeneous interfacial polymer catalysis

- **Film geometry and liquid contact:** Each tube contains a $0.25\text{ mm}$ thick amorphous PET film coupon at $2.9\%$ mass loading; accessible surface area depends on how the coupon sits in the liquid phase during agitation.
- **96-hour activity–stability coupling:** Over a four-day incubation at $60\text{–}70\,^\circ\text{C}$, an enzyme with high initial turnover that unfolds after a few hours yields the same cumulative product as a slow enzyme that stays folded for 96 h.
- **Mass summation across interconverting products ($\text{BHET} + \text{MHET} + \text{TPA}$):** Because MHET and TPA are secondary hydrolysis products of BHET, summing mass concentrations ($\text{mg/L}$) rather than molar equivalents weights unhydrolysed BHET ($254.24\text{ g/mol}$) $1.53\times$ more heavily per mole than TPA ($166.13\text{ g/mol}$) and masks differences in MHETase efficiency.
- **Low-expression loading exceptions:** The nominal $10\,\mu\text{g}$ enzyme loading ($0.7\text{ mg/g PET}$) holds *"unless noted otherwise in Supplementary Table 11 due to low expression levels"* `[Source: Main, Page 11]`, so poorly expressing scaffolds assayed at lower loading are not strictly normalized on a per-milligram basis.

---

## 2. Specificity vs. breadth — biological and chemical scope

- **Scope classification (`family_wide_homolog_panel`):** **19 natural bacterial scaffolds** across **570 enzyme–condition rows**, selected as the top performers from 51 expressible enzymes out of **74 thermophilic candidates spanning seven phylogenetic groups** (Main pp. 2–3).
- **Enzyme families & EC breadth:** Canonical PET hydrolases belong to **EC 3.1.1.-** (carboxylesterases, lipases, cutinases) and **EC 3.1.1.74** (cutinase)—*"known PET hydrolases are cutinases, lipases, and carboxylesterases (Enzyme Commission 3.1.1.-)"* `[Source: Main, Page 2]`—spanning Phylogenetic Groups 5–7, while Groups 1–4 probe more divergent scaffolds extending toward EC 3.4.-.- peptidases.
- **Taxonomic & environmental provenance:** Bacterial sequences from Actinobacteria, Proteobacteria, Firmicutes, and Bacteroidetes (Fig. 1a, Main p. 3), plus uncultured thermal metagenome assemblies from JGI IMG. All candidates passed an organism/environment thermotolerance gate: *"Sequences with OGT lower than 50 °C were discarded."* `[Source: Main, Page 2]`.
- **Construct modifications (`186–290` aa, median `270` aa):** Sequences in Enzymado are the **mature, signal-peptide-truncated recombinant constructs** with C-terminal $\text{His}_6$ tags: *"From 74 putative thermophilic PET hydrolase sequences, 36 signal peptides were removed for construct synthesis."* `[Source: Main, Page 10]`. No sequence exceeds 1,024 residues, no non-standard amino acids are present, and all 570 rows survive the shared PLM intersection (`used_in_runs`).
- **Structural coverage (not task inputs):** Eight X-ray crystal structures at $1.43\text{–}2.19\,\text{\AA}$ resolution (`7QJM` for 202, `7QJN` for 306, `7QJO` for 606, `7QJP` for 611, `7QJQ` for 702, `7QJR` for 703, `7QJS` for 705, `7QJT` for 711) plus AlphaFold2 models for all 74 candidates.
- **Chemical & condition scope (`has_ligand_smiles: false`):** Every row uses the same solid polymer substrate—amorphous PET film (Goodfellow `ES30-FM-000145`, $0.25\text{ mm}$ thick)—so no SMILES feature is defined. Chemical variation enters exclusively through the **7 numeric reaction-condition features**:
  - `temperature`: $30, 40, 50, 60, 70\,^\circ\text{C}$
  - `ph`: $6.0, 7.0, 7.5, 8.0, 9.0$ (five distinct pH values across six buffer/pH combinations)
  - One-hot buffer indicators: `buffer_c` (citrate, pH 6.0), `buffer_np` ($\text{NaH}_2\text{PO}_4$, pH 7.0 & 7.5), `buffer_h` (HEPES, pH 7.5), `buffer_b` (bicine, pH 8.0), `buffer_g` (glycine, pH 9.0).

---

## 3. Experimental measurement and STRENDA protocol

In contrast to literature-aggregated databases, nearly every STRENDA state variable is explicitly reported and controlled, and three state variables (`temperature`, `ph`, and buffer identity) serve directly as model features.

| STRENDA variable | Value | Model feature? | Status | Evidence & notes |
|---|---|---|---|---|
| **Assay technique** | 96 h endpoint hydrolysis of amorphous PET film; 1:1 methanol quench; $0.2\,\mu\text{m}$ nylon filtration; UHPLC-DAD quantitation of BHET + MHET + TPA at $240\text{ nm}$ | No | `reported` | *"Analyte analysis of BHET, MHET, and TPA was performed on an Infinity II 1290 ultra-high-performance liquid chromatography (UHPLC) system (Agilent Technologies) equipped with a G7117A diode array detector (DAD)."* `[Source: Main, Page 10]` |
| **Expression system** | *E. coli* BL21(DE3), OverExpress C41(DE3), or Lemo21(DE3); codon-optimised in `pET21b(+)` with C-terminal $\text{His}_6$ tag across 4 induction strategies (A–D) | No | `reported` | *"Each construct includes a C-terminal hexa-histidine epitope tag."* `[Source: Main, Page 10]`; *"In strategy A, the starter culture was inoculated at a 100-fold dilution into a 2xYT medium"* `[Source: SI, Page 2 (Combined p. 17)]` |
| **Purification & QC** | HisTrap HP $\to$ dialysis ($300\text{ mM NaCl}$, $20\text{ mM Tris pH 8.0}$) $\to$ HiLoad Superdex 75 SEC; SDS-PAGE, anti-His Western blot, BCA protein assay | No | `reported` | Main p. 10 |
| **Expression decoupled from label** | **true**, with low-yield exceptions | — | `interpreted` | *"In each screening reaction, 2.9% loading by mass of an amorphous PET film (Goodfellow) was incubated with 10 µg enzyme of interest (0.7 mg enzyme/g PET)"* `[Source: Main, Page 11]`. Exceptions noted in Suppl. Table 11 for low expressers |
| **Temperature ($T$)** | **$30, 40, 50, 60, 70\,^\circ\text{C}$** (5 levels) | **Yes (`temperature`)** | `reported` | *"and incubated at 30, 40, 50, 60, or 70 °C"* `[Source: Main, Page 11]` |
| **pH** | **$6.0, 7.0, 7.5, 8.0, 9.0$** across 6 buffer/pH combinations ($\text{pH } 7.5$ tested in both $\text{NaH}_2\text{PO}_4$ and HEPES) | **Yes (`ph`)** | `reported` | *"citrate at pH 6.0, NaH2PO4 at pH 7.0, NaH2PO4 at pH 7.5, HEPES at pH 7.5, bicine at pH 8.0, and glycine at pH 9.0"* `[Source: Main, Page 11]` |
| **Buffer composition** | $50\text{ mM}$ buffering agent + $100\text{ mM NaCl}$ in polypropylene tubes | **Yes (`buffer_c/np/h/b/g`)** | `reported` | *"Reactions were performed in polypropylene tubes containing 100 mM NaCl and 50 mM buffering agent"* `[Source: Main, Page 11]` |
| **Ionic strength** | Fixed $100\text{ mM NaCl} + 50\text{ mM}$ buffer background | No (constant) | `interpreted` | Formal ionic strength not separately computed |
| **Cofactors & metals** | None added (canonical Ser–His–Asp triad requires no cofactor) | No | `not_reported` | No metal or surfactant additives listed in Methods |
| **Substrate loading** | $2.9\%$ mass loading amorphous PET film ($0.25\text{ mm}$); $10\,\mu\text{g}$ enzyme ($0.7\text{ mg/g PET}$) | No (constant) | `reported` | Main p. 11 |
| **Reaction time** | Single **96 h** endpoint, quenched with equal volume $100\%$ methanol | No (constant) | `reported` | *"All reactions were terminated after 96 h by the addition of an equal volume of 100% methanol and PET was removed from the reaction solution."* `[Source: Main, Page 11]` |

---

## 4. How the dataset was built — curation and target transformation

### 4.1 Upstream selection and curation in the source paper

1. **HMM & phylogenetic mining:** *"Consequently, 74 sequences were selected."* `[Source: Main, Page 2]` (across 7 phylogenetic groups from NCBI nr and JGI IMG thermal metagenomes).
2. **Thermotolerance filter:** *"Sequences with OGT lower than 50 °C were discarded."* `[Source: Main, Page 2]`.
3. **Signal-peptide truncation:** *"From 74 putative thermophilic PET hydrolase sequences, 36 signal peptides were removed for construct synthesis."* `[Source: Main, Page 10]`.
4. **Soluble-expression gate ($74 \to 51$):** *"Insoluble enzymes and enzymes with yields below 0.1 mg/L were not screened."* `[Source: SI, Page 4 (Combined p. 19)]`.
5. **Top-performer selection ($51 \to 19$):** *"release across 30 reaction conditions using amorphous PET film as substrate for the 19 best-performing enzymes from each of the seven phylogenetic groups"* `[Source: Main, Page 3]`.
6. **Blank referencing:** *"Aromatic product release data are reported throughout relative to background aromatic product release detected in no-enzyme control reactions at each pH and temperature."* `[Source: Main, Page 11]`.

### 4.2 Source-table bug fix in Enzymado (`ERICKSON_ROW_FIXES`)

In the raw source table (`values.csv`), the seventh data row (0-based row index **6**) is labelled `"30°C, C6"` a second time immediately before the $40\,^\circ\text{C}$ block, which otherwise lacks a `C6` row, and its product yields fall smoothly between the $30\,^\circ\text{C}$ and $50\,^\circ\text{C}$ series. [`scripts/data_prep/real_cases/erickson_petase.py`](../../scripts/data_prep/real_cases/erickson_petase.py#L33) deterministically relabels row index 6 from `("30°C", "C6")` to `("40°C", "C6")`, restoring the complete $5\text{ temperatures} \times 6\text{ buffers} = 30\text{ conditions} \times 19\text{ enzymes} = 570\text{ rows}$ factorial grid.

### 4.3 Target transformation and zero-floor rationale

From `sum_aromatic_products` ($\text{BHET} + \text{MHET} + \text{TPA}$ in $\text{mg/L}$), Enzymado computes:

$$y = \log_{10}(\text{sum\_aromatic\_products} + 1.0), \qquad z = \frac{y - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

where $(\mu_{\text{train}}, \sigma_{\text{train}})$ are fitted on the 456 training rows only. As detailed in Section 1.2, the $+1.0\text{ mg/L}$ offset matches the $1\text{ mg/L}$ UHPLC lower limit of quantitation, mapping the $\sim 20\%$ left-censored zeros to $0.0$ log units ($-1.3280$ standardized) without introducing an arbitrary $10^{-6}$ spike while compressing the $0\text{–}2,700\text{ mg/L}$ dynamic range into $0\text{–}3.43$ log units.

---

## 5. Split semantics and leakage caveats

One split file is published: `erickson_random_80_10_10_standardized` ($456\text{ train} / 57\text{ val} / 57\text{ test} = 570\text{ total}$; see [`dataset_card.json`](../../data/processed/real_case/erickson_petase/dataset_card.json) for exact split moments).

- **How the split was constructed:** [`scripts/data_prep/generate_real_case_splits.py`](../../scripts/data_prep/generate_real_case_splits.py#L69-L75) generates splits for `real_case` tasks using **`random_split_unit="point"`** (observation-level splitting keyed on `dataset_id`, stratified across 10 deciles of `log10_sum_aromatic_products_plus_one` with `seed=42`).
- **Why there is no `similarity_mmseqs2` split:** With only 19 distinct scaffolds in total, an 80/10/10 cluster- or protein-grouped split would place at most $1\text{–}2$ scaffolds in `val` and $1\text{–}2$ scaffolds in `test`, yielding extreme variance across random seeds.

### Summary of leakage risks and structural caveats

1. **Same-scaffold sharing across splits (`random_split_unit="point"`):** Because splitting is stratified by observation (`point`) rather than grouped by `protein_sequence`, every test enzyme sequence is already present in the training set under $\sim 24$ other reaction conditions. **Do not claim zero-shot sequence generalisation from this split; it evaluates condition-surface prediction across a known 19-enzyme panel.**
2. **Collinearity between `ph` and buffer one-hots:** Four of the five buffers occur at only a single pH (`buffer_c` = 6.0, `buffer_b` = 8.0, `buffer_g` = 9.0, `buffer_h` = 7.5), and only $\text{pH } 7.5$ is shared between two buffers ($\text{NaH}_2\text{PO}_4$ and HEPES). Linear and tree models cannot separate buffer-ion effects from pH outside $\text{pH } 7.0\text{–}7.5$.
3. **Zero-inflation at $-1.3280$:** Roughly $20\%$ of every split consists of left-censored zeros ($< 1\text{ mg/L}$ LLOQ). Evaluate models both overall and restricted to active ($> 0\text{ mg/L}$) reaction cells.
4. **High-yield dilution heteroscedasticity:** Yields $> 200\text{ mg/L}$ required volumetric dilution prior to UHPLC injection.
5. **Triple survivorship bias:** The 19 scaffolds represent the active, expressible winners of a 74-candidate thermophilic screen ($74 \to 51 \to 19$).
6. **Substrate morphology specificity:** Condition optima learned on amorphous PET film do not transfer to crystalline PET powder.
