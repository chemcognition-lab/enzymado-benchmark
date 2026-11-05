# Dataset Card: `goldman_aminotransferase` — Aminotransferase Family Substrate-Specificity Screen

> **json card**: [`goldman_aminotransferase_dataset_card_enriched.json`](./goldman_aminotransferase_dataset_card_enriched.json)
> **computed statistics**: [`data/processed/real_case/goldman_aminotransferase/dataset_card.json`](../../data/processed/real_case/goldman_aminotransferase/dataset_card.json)
> **track**: `real_case` | **task type**: `protein_ligand_global` (scalar regression) | **layout**: Layout B

---

## 1. Identity & Provenance

| Field | Value |
|---|---|
| **Task / Dataset ID** | `goldman_aminotransferase` |
| **Property Name** | `log10_specific_activity` |
| **Biochemical Observable** | $\log_{10}(\text{specific activity in U mg}^{-1})$ for transamination of 18 canonical L-amino acids to 2-ketoacids using 2-ketoglutarate as amino acceptor |
| **Primary Experimental Source** | Li, T. et al. *Exploration of transaminase diversity for the oxidative conversion of natural amino acids into 2-ketoacids and high-value chemicals*. *ACS Catalysis* **10**, 7950–7957 (2020), DOI: [`10.1021/acscatal.0c01895`](https://doi.org/10.1021/acscatal.0c01895) |
| **Benchmark Curation Source** | Goldman, S., Das, R., Yang, K. K. & Coley, C. W. *Machine learning modeling of family wide enzyme-substrate specificity screens*. *PLoS Comput. Biol.* **18**(2), e1009853 (2022), DOI: [`10.1371/journal.pcbi.1009853`](https://doi.org/10.1371/journal.pcbi.1009853) |
| **Preprocessing Script** | [`scripts/data_prep/real_cases/goldman_aminotransferase.py`](../../scripts/data_prep/real_cases/goldman_aminotransferase.py) |
| **Published Split** | `goldman_aminotransferase_random_80_10_10_standardized` (`336` train / `48` val / `48` test; `432` total rows) |

---

## 2. How to Think About This Dataset

1. **Complete $24 \times 18$ family-wide specificity matrix (`432` pairs)**: Every one of the **24 wild-type prokaryotic aminotransferases** (12 Class I/II AspAT/ARAT/AlaAT from Pfam `PF00155` and 12 Class IV branched-chain aminotransferases BCAT from Pfam `PF01063`) is profiled against the exact same **18 proteinogenic L-amino acids** under uniform in vitro conditions.
2. **Heavily left-censored activity distribution (`57.2%` at the detection limit)**: In Table S7 of Li et al. (2020), any enzyme–substrate pair with specific activity below $10^{-2}\text{ U mg}^{-1}$ after extended 2–4 h incubation is reported as `<10-2` and encoded in Goldman et al.'s `LogSpActivity` column as `-2.0`. Exactly **247 of 432 rows (`57.2%`)** sit at this `-2.0` floor (`-0.7719` after training-set z-score standardization), making the minimum, 25th percentile, and median identical (`-0.7719`) in every split.
3. **Observation-level matrix-completion split**: Because `generate_real_case_splits.py` partitions individual `(enzyme, substrate)` observations (`random_split_unit="point"`), all 24 enzymes and all 18 L-amino acids appear across train, val, and test. The task evaluates **matrix completion / bilinear specificity interpolation** across seen enzymes and seen amino acids rather than zero-shot enzyme discovery.

---

## 3. Label Noise & Performance Ceiling

- **Assay Precision & Replicate Design**: Specific activities in Table S7 were measured on Ni-NTA affinity-purified, desalted $\text{His}_6$-tagged enzymes using calibrated reversed-phase HPLC with an internal standard (10 mM L-norvaline for DNFB derivatization of L-glutamate at 363 nm, or direct UV detection of 2-ketoacids for Phe, Tyr, Trp, and His). However, Table S7 (SI p. S32) explicitly notes: *"The values were calculated from one independent experiment"* (`n = 1`, sampled at two withdrawal timepoints: 10 min and 20 min, or 2 h and 4 h for slow reactions).
- **Decomposed Noise & Distortion Sources**:
  1. **Left-censoring at $<10^{-2}\text{ U mg}^{-1}$ (`57.2%` of pairs)**: Collapses all non-cognate and trace-activity pairs onto a point mass at `-2.0` ($\text{standardized } -0.7719$).
  2. **Single-replicate (`n = 1`) two-timepoint initial rate estimation**: Introduces unaveraged aliquot withdrawal and HPLC integration variance across the 185 active pairs.
  3. **Dual analytical readout protocols**: Aromatic/heteroaromatic substrates (Phe, Tyr, Trp, His) were quantified by direct 2-ketoacid UV absorption (210 or 294 nm), whereas the other 14 amino acids were quantified via DNFB derivatization of L-glutamate at 363 nm (Table S2, SI p. S26).
  4. **Acceptor swap on `ATI1`**: While 23 of 24 enzymes used 10 mM 2-ketoglutarate as amino acceptor, archaeal `ATI1` used 20 mM pyruvate instead (Table S7 footnote *a*, SI p. S32).

---

## 4. Specificity, Breadth & Construct Scope

- **Enzyme Panel (`24` proteins, lengths `273–450` aa)**: Chosen from Sequence Similarity Networks (SSNs) of Pfam `PF00155` (Class I/II AspAT, ARAT, AlaAT; Fold Type I) and `PF01063` (Class IV BCAT; Fold Type IV) spanning diverse bacterial and archaeal species (*E. coli*, *K. pneumoniae*, *P. putida*, *A. baylyi*, *T. uzoniensis*, *P. horikoshii*, etc.). Out of 28 cloned candidates (Tables S3–S4), 3 (`AlaAT7`, `ATI2`, `BCAT3`) failed soluble expression in *E. coli* BL21(DE3) (Table S6, SI p. S31), leaving 25 purified enzymes (24 unique UniProt sequences in Goldman et al.'s CSV).
- **Construct & Tagging**: Cloned into `pET22b(+)` with six histidine codons inserted immediately after the start codon (`N-terminal His6 tag`).
- **Chemical Scope (`18` L-amino acids)**: Gly, L-Ala, L-Val, L-Leu, L-Ile, L-Phe, L-Tyr, L-Trp, L-Ser, L-Thr, L-Cys, L-Met, L-Asn, L-Gln, L-Asp, L-Lys, L-Arg, and L-His (with L-stereochemistry preserved in `ligand_smiles`). L-Pro (secondary amine) and L-Glu (transamination product of 2-ketoglutarate) were not part of the donor panel.

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition |
|---|---|
| **Assay Format** | Discontinuous purified-enzyme reaction in $500\,\mu\text{L}$; $100\,\mu\text{L}$ aliquots quenched at 10 & 20 min (or 2 & 4 h) |
| **Readout** | Shimadzu LC2030 C18 RP-HPLC: DNFB derivatization of L-Glu at 363 nm (10 mM L-norvaline IS) or direct UV of 2-ketoacids (210/294 nm) |
| **Temperature** | $30\,^\circ\text{C}$ |
| **pH & Buffer** | pH $7.5$, $20\text{ mM}$ potassium phosphate buffer |
| **Cofactors & Acceptor** | $20\,\mu\text{M}$ PLP; $10\text{ mM}$ 2-ketoglutarate ($20\text{ mM}$ pyruvate for `ATI1`) |
| **Substrate & Enzyme Loading** | $20\text{ mM}$ L-amino acid donor; $10\,\mu\text{g}$ purified enzyme per $500\,\mu\text{L}$ ($0.02\text{ mg mL}^{-1}$) |

---

## 6. Split Semantics, Statistics & Caveats

| Split | Rows | Proteins | Ligands | Mean | Std | Min / p25 / Median | p75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train** | 336 | 24 | 18 | `0.0000` | `1.0000` | `-0.7719` | `0.7321` | `2.2636` |
| **Val** | 48 | 24 | 16 | `0.0079` | `1.0102` | `-0.7719` | `0.8192` | `2.2752` |
| **Test** | 48 | 24 | 17 | `-0.0268` | `1.0108` | `-0.7719` | `0.7619` | `1.9781` |

- **Caveat on `used_in_runs`**: Aminotransferases are obligate homodimers/oligomers whose catalytic pockets sit at the subunit interface; filtering by `used_in_runs == True` (single-chain monomer filter) retains only `90` rows (`5` proteins).
