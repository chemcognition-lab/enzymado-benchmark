# Dataset Card: `koudelakova_hld_panel` — Haloalkane Dehalogenase Family Substrate Specificity Panel (`13` HLDs $\times$ `30` Halogenated Substrates)

> **json card**: [`koudelakova_hld_panel_dataset_card_enriched.json`](./koudelakova_hld_panel_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/koudelakova_hld_panel/dataset_card.json`](../../data/processed/real_case/koudelakova_hld_panel/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `koudelakova_hld_panel` |
| **Property Name** | `log10_activity` |
| **Biochemical Observable** | $\log_{10}(w_{e,s} + 10^{-3})$, where $w_{e,s} = \frac{\log(a_{e,s} + 1)}{\sum_{s'} \log(a_{e,s'} + 1)}$ is the within-enzyme normalized log-specific-activity share from Supplementary Table S5 ($a_{e,s}$ in $\text{nmol s}^{-1}\text{ mg}^{-1}$ of IMAC-purified enzyme, measured via the Iwasaki $460\text{ nm}$ halide-release time-course assay across $\ge 3$ independent experiments) across 13 haloalkane dehalogenases and 30 statistically selected halogenated substrates |
| **Primary Experimental Source** | Koudelakova, T., Chovancova, E., Brezovsky, J., Monincova, M., Fortova, A., Jarkovsky, J. & Damborsky, J. *Substrate specificity of haloalkane dehalogenases*. *Biochemical Journal* **435**(2), 345–354 (2011), DOI: [`10.1042/BJ20101405`](https://doi.org/10.1042/BJ20101405) |
| **Raw Data Source** | Supplementary Online Data Table S5 (`Transformed data matrix`, p. 17 of `docs/pdfs/hld_koudelakova2011_biochemj.pdf`; raw rates in Table S3, pp. 15–16) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/koudelakova_hld.py`](../../scripts/data_prep/real_cases/koudelakova_hld.py) |
| **Published Split** | `koudelakova_hld_panel_random_80_10_10_standardized` (`312` train / `39` val / `39` test; `390` total rows) |

---

## 2. How to Think About This Dataset

1. **Complete `13`-enzyme $\times$ `30`-substrate haloalkane dehalogenase (`EC 3.8.1.5`) matrix (`390` rows)**: Profiles 9 wild-type bacterial HLDs spanning all three phylogenetic subfamilies (`HLD-I`, `HLD-II`, `HLD-III`) and four functional substrate-specificity groups (`SSG-I` to `SSG-IV`: `DatA`, `DbeA`, `DbjA`, `DhaA`, `DhlA`, `DmbA`, `DmbC`, `DrbA`, `LinB`) plus 4 engineered variants (`DbeA1`, `DbeA2`, `DbjAΔ`, `DhaA31`) against 30 chlorinated, brominated, and iodinated substrates selected by PCA + $2^4$ factorial experimental design from nearly 200 halogenated compounds.
2. **CRITICAL — Double-transformed within-enzyme relative specificity target (`Supplementary Table S5`)**: [`koudelakova_hld.py`](../../scripts/data_prep/real_cases/koudelakova_hld.py) ingests **Supplementary Table S5 (`Transformed data matrix`, p. 17)** rather than raw Table S3 ($\text{nmol s}^{-1}\text{ mg}^{-1}$). In Table S5, Koudelakova et al. deliberately removed the $>100$-fold differences in overall enzyme catalytic rate ($\text{LinB}\sim\text{DbjA} > \text{DhlA}\sim\text{DhaA}\sim\text{DbeA}\sim\text{DmbA} > \text{DatA}\sim\text{DmbC}\sim\text{DrbA}$) so PCA would cluster enzymes purely by *relative* substrate profile:
   $$w_{e,s} = \frac{\log(a_{e,s} + 1)}{\sum_{s'=1}^{30} \log(a_{e,s'} + 1)}, \qquad y_{e,s} = \log_{10}(w_{e,s} + 10^{-3})$$
3. **Practical consequence of $\sum_{s=1}^{30} w_{e,s} = 1.0$ per enzyme**:
   - **Sequence-only models have near-zero explanatory power** because every enzyme's 30 weights sum to $1.0$.
   - **Specialist vs. generalist scaling**: An enzyme active on only 9–11 substrates (`DrbA`, `DmbC`) concentrates its unit weight across fewer substrates (producing the dataset's highest single-cell values, up to $w_{e,s} = 0.1989$, `+1.7016` standardized) than a broad-spectrum generalist active on 28–29 substrates (`LinB`, `DbjA`).

---

## 3. Label Noise & Performance Ceiling

- **Multi-Timepoint Replicate Design (`n >= 3`) with GC-FID Substrate Verification**: Every cell in the underlying screen was measured across **at least three independent 40-min time-course experiments** in sealed $25\text{ cm}^3$ Mininert microflasks, with actual initial dissolved haloalkane concentrations verified by GC-FID (main text p. 2; SI Table S3 p. 16).
- **Decomposed Noise & Distortion Sources**:
  1. **Within-enzyme sum-to-one normalization coupling (`Table S5`)**: Couples each cell $w_{e,s}$ to the enzyme's sum across all 30 substrates.
  2. **Zero-activity floor (`79` of `390` rows, `20.26%`)**: Recalcitrant chlorinated alkanes/cycloalkanes with $w_{e,s} = 0.0000$ map to `-3.0` (`-1.4820` standardized).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`13` HLDs, lengths `291–327` aa, median `302.0` aa)**: 9 wild-type enzymes (`DatA`, `DbeA`, `DbjA`, `DhaA`, `DhlA`, `DmbA`, `DmbC`, `DrbA`, `LinB`) and 4 engineered tunnel/loop variants (`DbeA1`, `DbeA2`, `DbjAΔ`, `DhaA31`). All `390` rows (`100%`, all 13 proteins) are in `used_in_runs`.
- **Chemical Scope (`30` halogenated substrates)**: Chlorinated, brominated, and iodinated C1–C7 alkanes, alkenes, cycloalkanes, ethers (`bis(2-chloroethyl)ether`), arylalkyl halides (`(2-bromoethyl)benzene`), and nitriles (`4-bromobutyronitrile`).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Expression & Purification** | His-tagged expression in *E. coli* or *M. smegmatis*, purified to homogeneity by IMAC |
| **Assay Vessel & Sampling** | Sealed $25\text{ cm}^3$ microflasks with Mininert valves; 5 periodic $1\text{ cm}^3$ syringe aliquots over $40\text{ min}$, quenched in $35\%$ nitric acid (`n >= 3` independent experiments) |
| **Readout & Calibration** | Iwasaki mercuric thiocyanate + ferric ammonium sulfate colorimetric halide assay at $460\text{ nm}$ (Tecan Sunrise); initial dissolved substrate concentration measured by GC-FID (Thermo Fisher Trace 2000) |
| **Temperature** | $37\,^\circ\text{C}$ |
| **pH & Buffer** | pH $8.6$, $100\text{ mM}$ halide-free glycine buffer (cofactor-independent hydrolysis) |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min | p25 | Median | p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 312 | 13 | 30 | `0.0000` | `1.0000` | `-1.4820` | `-0.6844` | `0.2145` | `0.7330` | `1.7016` |
| **Val** | 39 | 13 | 23 | `0.0086` | `0.9963` | `-1.4820` | `-0.6001` | `0.2096` | `0.7131` | `1.3917` |
| **Test** | 39 | 13 | 19 | `-0.0156` | `1.0127` | `-1.4820` | `-0.8315` | `0.2339` | `0.7375` | `1.6305` |
