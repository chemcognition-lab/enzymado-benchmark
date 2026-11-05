# SARS-CoV-2 Main Protease ($\text{M}^{\text{pro}}$ / $\text{3CL}^{\text{pro}}$) Deep Mutational Scan — Intracellular FRET Substrate Cleavage (`mpro_fret_cleavage`)

> **Track:** `real_case` | **Task Type:** `regression` (`scalar`, `protein_level`) | **N:** `5,745` unique variants (`4,595` train / `572` val / `578` test)
> **Primary Citation:** Flynn, J. M. *et al.* Comprehensive fitness landscape of SARS-CoV-2 $\text{M}^{\text{pro}}$ reveals insights into viral resistance mechanisms. *eLife* **11**, e77433 (2022). [DOI: 10.7554/eLife.77433](https://doi.org/10.7554/eLife.77433)
> **Machine-Readable Contract:** [`mpro_fret_cleavage_dataset_card_enriched.json`](../../docs/data_cards/mpro_fret_cleavage_dataset_card_enriched.json) | **Computed Stats:** [`data/processed/real_case/mpro_fret_cleavage/dataset_card.json`](../../data/processed/real_case/mpro_fret_cleavage/dataset_card.json)

---

## 1. How to Think About This Dataset

- **Comprehensive Antiviral Drug-Target DMS:** `mpro_fret_cleavage` measures **5,744 single missense variants plus 1 wild-type sequence (`5,745` total; $99.45\%$ of the $5,776$ possible missense variants across 304 synthesized positions)** of the **SARS-CoV-2 main protease ($\text{M}^{\text{pro}}$ / $\text{3CL}^{\text{pro}}$ / Nsp5, 306 aa)**, an obligate homodimeric Cys145–His41 cysteine protease and the clinical target of nirmatrelvir (Paxlovid).
- **Authentic N-Terminus via Cleavable Ubiquitin Fusion:** Because the free N-terminal Ser1 ("N-finger") of each $\text{M}^{\text{pro}}$ protomer inserts into the partner subunit's S1 pocket to stabilize the catalytic site, $\text{M}^{\text{pro}}$ was expressed in *Saccharomyces cerevisiae* as an N-terminal **Ubiquitin fusion (`Ub-Mpro`)**. Endogenous yeast deubiquitinases immediately cleave Ub to expose the native, un-tagged N-terminal Ser1.
- **Ceiling-Saturated Bimodal Cleavage Readout:** Cells co-expressing chromosomally integrated **`CyPet-MproCS-YPet`** (carrying the canonical Nsp4/5 autoprocessing site `TSAVLQ|SGFRK`) are induced with $125\text{ nM}$ $\beta$-estradiol for $1.5\text{ h}$—the exact duration needed for wild-type $\text{M}^{\text{pro}}$ to completely cleave the reporter—and sorted into cleaved (low FRET) and uncleaved (high FRET) gates. Consequently, tolerated variants pile up near the `1.0` ceiling (median `0.9805`, standardized `+0.604`), while mutations at the Cys145–His41 dyad, S1–S4 pockets, or dimer interface (`S10`, `G11`, `E14`) drop to the `0.0` stop-codon floor.

---

## 2. Label Noise & Performance Ceiling

| Attribute | Value / Finding | Source |
| :--- | :--- | :--- |
| **Noise Regime** | Very low binary FACS-seq classification noise with upper-ceiling saturation at `1.0` | `main` p. 6 |
| **Biological Replicates** | **$n = 2$ independent biological replicates** (`10–20` PacBio-linked barcodes/variant; `5,735` of `5,745` sequences present in both replicates and averaged) | `main` p. 6, 20 |
| **Replicate Concordance & Ceiling** | Paper reports **$R^2 > 0.98$**; exact computed across 5,735 shared unique missense/WT sequences: **Pearson $r = 0.9841$** ($R^2 = 0.9685$, Spearman $\rho = 0.6914$, $\text{MAD} = 0.0404$), giving a 2-replicate average ceiling of **$R^2_{\max} = 0.992$** ($r_{\max} = 0.996$) | `main` p. 6; computed from `elife-77433-fig2-data1-v2.xlsx` |
| **Missing Positions** | Positions **27 (`Leu27`) and 28 (`Asn28`)** were absent from Twist oligo synthesis | `main` p. 19 |

---

## 3. Specificity, Breadth & Construct Architecture

- **Enzyme:** SARS-CoV-2 main protease $\text{M}^{\text{pro}}$ / $\text{3CL}^{\text{pro}}$ / Nsp5 (EC `3.4.22.69`, GenBank `MN908947.3` ORF1ab residues 3264–3569, UniProt `P0DTC1`, PDB `6LU7`).
- **Construct:** `p416LexA-UbMpro(lib)-B112` in *S. cerevisiae* W303; N-terminal Ubiquitin is removed endogenously to leave the native, untagged 306-aa $\text{M}^{\text{pro}}$ polypeptide.

---

## 4. How the Dataset Was Built & Target Transformation

- **Upstream Definition:** Fraction of variant barcode reads in the cleaved (low FRET) population relative to both populations, $F_{\text{cut}} / (F_{\text{cut}} + F_{\text{uncut}})$, linearly scaled per replicate so that the average wild-type barcode equals `1.0` and the average premature stop codon equals `0.0`.
- **Enzymado Ingestion ([`mpro_fret.py`](../../scripts/data_prep/real_cases/mpro_fret.py)):** Reads sheets `FRET scores rep 1` (`6,047` rows) and `FRET scores rep 2` (`6,040` rows) from `data/raw/real_cases/mpro_fret/elife-77433-fig2-data1-v2.xlsx`, filters out stop codons (`*`), collapses explicit `WT` and 304 single-codon synonymous control rows (`S1S`, `G2G`, etc.) into the single wild-type sequence (`functional_score = 1.000557`), averages `functional_score` across the two replicates for all `5,745` unique sequences, and z-score standardizes using the training split ($\mu_{\text{train}} = 0.719284$, $\sigma_{\text{train}} = 0.432227$).

---

## 5. Experimental Measurement & STRENDA Conditions

| Parameter | Condition | Source |
| :--- | :--- | :--- |
| **Host & Growth Media** | *S. cerevisiae* W303 (`Ade::TEF-CyPet-MproCS-YPet`) in log-phase `SD-A-U` medium at **$30\,^\circ\text{C}$** ($180\text{ rpm}$) | `main` p. 21 |
| **Induction Window** | **$125\text{ nM}$ $\beta$-estradiol for $1.5\text{ h}$ at $30\,^\circ\text{C}$** (low toxicity regime) | `main` p. 6, 21 |
| **Intracellular Substrate** | Chromosomally integrated **`CyPet-MproCS-YPet`** fusion containing the Nsp4/5 cleavage site **`TSAVLQ|SGFRK`** | `main` p. 5, 21 |
| **FACS Wash & Sorting** | Washed $3\times$ in `TBST-BSA` (Tris-buffered saline + $0.1\%$ Tween + $0.1\%$ BSA) and sorted ($\ge 1.5 \times 10^6\text{ cells/gate}$) into low-FRET (cut) vs. high-FRET (uncut) populations | `main` p. 21 |

---

## 6. Split Semantics & Caveats

- **Split Design:** Deterministic 80/10/10 random split (`4,595` train / `572` val / `578` test).
- **Pearson $r$ vs. Spearman $\rho$ Discrepancy:** Because $>50\%$ of variants saturate the $1.5\text{ h}$ intracellular cleavage assay between `0.95` and `1.05`, rank correlation within the functional plateau is compressed (replicate Spearman $\rho = 0.6914$ vs. replicate Pearson $r = 0.9841$). Always evaluate both $R^2$/Pearson $r$ and Spearman $\rho$.
