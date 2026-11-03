---
model_id: prost5
enzymado_key: prost5
model_name: ProstT5
size_variant: 3B enc-dec (1.21B encoder used)
domain: protein
license: null
created_at: '2026-11-03'
datasets:
  - AFDB (train17M)
  - BFD (via ProtT5 base)
  - UniRef50 (via ProtT5 base)
tags:
  - protein
  - transformer
  - protein_language_model
  - enzymado
base_model: Rostlab/ProstT5
doi: 10.1093/nargab/lqae150
code: https://github.com/mheinzinger/ProstT5
---

# Model Card: ProstT5 (`Rostlab/ProstT5`, encoder only)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and page citation (`[Main, p. X]` or `[SI, p. Y]`). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from paper and SI), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings, so it can be paired with any Dataset Card for mechanistic modeling analysis without leaking benchmark results.

---

## 0. How to think about this model

**ProstT5** (`Rostlab/ProstT5`, Enzymado key `prost5`) is a bilingual sequence–structure T5 model initialized from `ProtT5-XL-U50` (pretrained via span-based denoising on 2.1B BFD metagenomic sequences + UniRef50) and fine-tuned across three sequential stages on `train17M`—17 million high-confidence ($\text{pLDDT} \ge 70$) AlphaFoldDB structures clustered by both MMseqs2 (50% sequence identity) and Foldseek (structural similarity)—to translate bidirectionally between 1D amino-acid sequences (`AA`) and 1D Foldseek local tertiary interaction tokens (`3Di`). In Enzymado, only the 24-layer encoder (`T5EncoderModel`, ~1.21B parameters, $D = 1{,}024$) is run on amino-acid sequences prefixed with the translation direction token `<AA2fold>`, which conditions the encoder to project 1D sequence inputs directly toward their predicted 3Di structural state space.

- **Best-suited dataset regimes:**
  - Globular enzyme properties tightly coupled to 3D fold, secondary/tertiary packing, thermostability, or catalytic scaffold geometry where explicit 3Di structural distillation enriches sequence embeddings beyond 1D evolutionary statistics.
  - Remote-homology and low-sequence-identity regimes (such as `mmseqs2` 50% cluster splits or superfamily-level generalization) where 3Di structural conservation persists after 1D sequence identity decays.
  - Tasks paired with non-linear readout heads (`mlp`, `xgboost`) capable of decoding the multi-channel 3Di structural features prepared by the `<AA2fold>`-conditioned encoder.
- **Expected vulnerabilities & blind spots:**
  - Tasks driven by intrinsically disordered regions (IDRs), flexible loops, or unstructured sorting/localization motifs—the authors explicitly note that filtering AFDB for $\text{pLDDT} \ge 70$ removes most disordered proteins and degrades subcellular localization accuracy relative to base `ProtT5`.
  - Long multi-domain enzymes exceeding 512 residues: ProstT5 fine-tuning truncated all training sequences at $\le 256$ tokens (Phases 1 & 2a) and $\le 512$ tokens (Phase 2b), whereas Enzymado extracts full-length sequences with `truncation=False`.
  - Purely linear probes (`linear`, `attention_linear`) when the target property requires non-linear combinations of the 20-state 3Di structural code that the encoder was trained to feed into a 2-layer CNN or 24-layer T5 decoder.
- **Downstream probe interaction summary:** Because the `<AA2fold>` prefix instructs the encoder to prepare representations for a non-linear 24-layer T5 decoder (or a 2-layer CNN classifier) predicting 20 discrete 3Di states—and because 3Di states exhibit strong class imbalance toward helical/well-structured states—non-linear probes (`mlp`, `xgboost`) can partition and recombine these distributed structural channels much more flexibly than a single global linear hyperplane (`linear`, `attention_linear`).

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Initialized from `ProtT5-XL-U50` (span-based denoising on 2.1B BFD + UniRef50 sequences) and fine-tuned first with T5 span denoising on both AA and 3Di tokens (Phase 1) and then with simultaneous bidirectional translation between 1D amino-acid sequences (`<AA2fold>`) and 1D Foldseek 3Di structural sequences (`<fold2AA>`) (Phase 2). When prompted with `<AA2fold>` (as in Enzymado), the 24-layer encoder transforms 1D amino-acid sequences into a latent space explicitly aligned with local 3D tertiary contact states (`3Di`).
  > *"Both directions were trained simultaneously with the prefixes < fold2AA > and < AA2fold > indicating the directionality of the translation."* `[Main, p. 3]`
  > *"Even without leveraging 3Di-information (ProstT5(AA) used only AA input), bilingual ProstT5 improved over its base model ProtT5"* `[Main, p. 7]`
  > *"tasks strictly related to a protein’s structure or fold are very likely to improve from the new embeddings."* `[Main, p. 11]`
- **Downstream task implications:** Directly enriches the encoder embeddings with fold-level, secondary-structure (`Q3`), and CATH superfamily remote-homology signal (`ROC-AUC = 0.45` vs. `0.49` for experimental structures) from sequence alone, making the representation strongly suited for structure-coupled enzymatic properties (thermostability, fold-dependent catalysis, remote-homology generalization).

### 1.2 Pretraining corpus filtering & sampling
- **Mechanism:** Fine-tuned on `train17M`, curated from 214M AlphaFoldDB (AFDB) structures by clustering at 50% pairwise sequence identity with MMseqs2, clustering structurally with Foldseek, expanding clusters, and strictly filtering out proteins with low AlphaFold confidence ($\text{pLDDT} < 70$), short length ($< 30$ aa), or highly repetitive 3Di strings. The authors explicitly report (`ProstT5 is not a general-purpose pLM`, `[Main, p. 7]`) that filtering for $\text{pLDDT} \ge 70$ biases the training distribution toward well-structured globular proteins (and helical 3Di states `v`, `d`), removing most intrinsically disordered proteins and causing a drop on sequence-motif/function benchmarks such as subcellular localization relative to base `ProtT5`.
  > *"MMseqs2 (35) clustered 214 million (M) UniprotKB (34) protein sequences from AFDB such that no pair had over 50% pairwise sequence identity"* `[Main, p. 2]`
  > *"filtering the AFDB for predictions with high pLDDT removes most intrinsically disordered proteins"* `[Main, p. 11]`
  > *"a clear decrease in predicting subcellular location when inputting exclusively amino acid sequences"* `[Main, p. 7]`
- **Downstream task implications:** Highly advantageous for globular enzymes with well-defined 3D folds (which dominate catalytic and stability benchmarks), but introduces a known representational blind spot for tasks governed by disordered tails, flexible linkers, or unstructured sorting signal peptides.

### 1.3 Training crop window & positional encoding
- **Mechanism:** Uses T5 relative positional embeddings (learned scalar attention biases bucketed by relative token distance). During ProstT5 fine-tuning on `train17M` (mean protein length ~238 residues), sequences were truncated at 256 tokens during Phase 1 and Phase 2a, and increased to 512 tokens during Phase 2b.
  > *"limiting sequence lengths to 256 (truncating longer sequences)"* `[Main, p. 3]`
  > *"increased to 512 for another 600K (6 × 10⁵) steps (20 days)"* `[Main, p. 3]`
- **Downstream task implications:** Enzymado runs ProstT5 extraction with `truncation=False`; single-domain enzymes $\le 512$ residues fall strictly within the Phase 2b translation window, whereas longer multi-domain enzymes ($> 512$ aa) exceed the fine-tuned sequence length cap and rely on T5 relative-position bucket clamping.

### 1.4 Embedding width $D$, pretraining dropout & downstream probe parameterization
- **Mechanism:** Outputs a compact $D = 1{,}024$ per-residue vector from the 24th encoder layer. Crucially, during Phase 2 training under the `<AA2fold>` prefix, the encoder's last hidden state is optimized to be consumed by a non-linear 24-layer T5 decoder (or a 2-layer CNN head in the authors' fast 3Di predictor) that maps continuous embeddings onto 20 discrete, class-imbalanced 3Di structural states.
  > *"we extracted the hidden states of the encoder’s last layer"* `[Main, p. 6]`
  > *"we replaced it by a two-layer CNN trained to predict 3Di states directly from the encoder’s amino acid embeddings."* `[Main, p. 10]`
- **Downstream task implications:** Because the encoder features under `<AA2fold>` are organized as inputs to a non-linear 3Di decoder/CNN rather than a linear classifier, non-linear downstream probes (`mlp` and `xgboost` over the 1,024-D pooled vector) can decode non-linear combinations of 3Di structural channels that a single linear hyperplane (`linear` or `attention_linear`) cannot separate.

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | TUM (Technical University of Munich, Rost lab) & Seoul National University (Steinegger lab) | `reported` | `[Main, p. 1]` *"Department of Informatics, Bioinformatics & Computational Biology, TUM (Technical University of Munich)"* |
| **Publication** | Heinzinger et al. (2024), *NAR Genomics and Bioinformatics* 6(4), lqae150 | `reported` | `[Main, p. 1]` *"https://doi.org/10.1093/nargab/lqae150"* |
| **Code & Weights** | `https://github.com/mheinzinger/ProstT5` & `https://huggingface.co/Rostlab/ProstT5` | `reported` | `[Main, p. 12]` *"example scripts deposited at https://github.com/mheinzinger/ProstT5"*; *"Our model is freely available for all at https://huggingface.co/Rostlab/ProstT5"* |
| **License** | Not stated for weights in paper (article text is CC BY 4.0) | `not_reported` | `[Main, p. 1]` *"Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/)"* (applies to article text; check HF card for weights license) |
| **Base Checkpoint** | `ProtT5-XL-U50` (`Rostlab/prot_t5_xl_uniref50`) | `reported` | `[Main, p. 2]` *"ProtT5 (marked ProtT5-XL-U50 in the original publication (6))"* |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | T5 encoder–decoder (Enzymado loads encoder only: `T5EncoderModel`) | `reported` | `[Main, p. 3]` *"ProstT5 is a T5-based encoder-decoder model initialized with weights of the ProtT5 model"* |
| **Total Parameters** | 3B full encoder–decoder (`1,210M` encoder used in Enzymado) | `reported` | `[Main, p. 2]` *"The 3B (3 × 10⁹) free parameters of ProtT5 were fine-tuned"* |
| **Layers ($L_{\text{layers}}$)** | 24 encoder + 24 decoder layers (24 encoder layers used) | `reported` | `[Main, p. 3]` *"consists of 24 layers in the encoder and decoder"* |
| **Hidden Dim ($D$)** | 1,024 | `reported` | `[Main, p. 3]` *"with an embedding dimension of 1024"* |
| **Attention Heads ($H$) & FFN Dim** | Not restated in paper/SI (`H = 32`, `d_ffn = 16,384`, ReLU inherited from `ProtT5-XL-U50`) | `not_reported` | Inherited unchanged from `ProtT5-XL-U50` (Elnaggar et al. 2021). |
| **Positional Encoding** | T5 relative positional embeddings | `reported` | `[Main, p. 3]` *"relative positional embeddings"* |
| **Training Crop / Truncation Length** | 256 tokens (Phases 1 & 2a) $\rightarrow$ 512 tokens (Phase 2b) | `reported` | `[Main, p. 3]` *"limiting sequence lengths to 256 (truncating longer sequences)"*; *"increased to 512 for another 600K (6 × 10⁵) steps (20 days)"* |
| **Vocabulary (170 tokens)** | 20 uppercase AA + 20 lowercase Foldseek 3Di + 128 span-corruption tokens + 2 direction prefixes (`<AA2fold>`, `<fold2AA>`) | `reported` | `[Main, p. 2]` *"expanded the existing ProtT5 vocabulary consisting of the 20 standard amino acids and 128 special tokens introduced for span-corruption (40) by the 20 3Di tokens"*; *"added two special tokens, namely < fold2AA > and < AA2fold >"* |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objectives** | Inherited: T5 span denoising (BFD 2.1B + UniRef50); Phase 1: AA + 3Di span denoising; Phase 2: simultaneous `<AA2fold>` / `<fold2AA>` translation | `reported` | `[Main, p. 3]` *"we applied the original pre-training objective of ProtT5 (span-based denoising) to both, AAs and 3Di"*; *"Both directions were trained simultaneously with the prefixes < fold2AA > and < AA2fold > indicating the directionality of the translation."* |
| **Fine-Tuning Corpus (`train17M`)** | 17M AFDB proteins ($\times 2$ modalities = 34M samples); 474 val + 474 test proteins held out | `reported` | `[Main, p. 2]` *"The final training set contained 17M proteins"*; *"this corresponded to 34M training samples."* |
| **Corpus Curation & Filtering** | 214M AFDB UniprotKB proteins $\rightarrow$ MMseqs2 50% PIDE clustering $\rightarrow$ Foldseek structural clustering $\rightarrow$ $\text{pLDDT} \ge 70$, length $\ge 30$, non-repetitive 3Di filters | `reported` | `[Main, p. 2]` *"MMseqs2 (35) clustered 214 million (M) UniprotKB (34) protein sequences from AFDB such that no pair had over 50% pairwise sequence identity"*; *"short proteins (length < 30) and (c) proteins with highly repetitive 3Di-strings"* |
| **Batch Size & Gradient Accum.** | Phase 1: batch `16`, accum `5`; Phase 2a: batch `16`, accum `1`; Phase 2b: batch `6`, accum `4` | `reported` | `[Main, p. 2–3]` *"using a batch-size of 16"*; *"gradient accumulation steps (5 steps)"*; *"we had to lower batch-size (from 16 to 6) which we compensated for by increasing the number of gradient accumulation steps (from 1 to 4)."* |
| **Steps & Learning Rate** | Phase 1: `102M` samples (~3 epochs), LR `1e-3`; Phase 2: `100K` + `600K` = `700K` steps (~4 epochs), LR `1e-5` | `reported` | `[Main, p. 2–3]` *"fine-tuned with a learning rate of 10⁻³ for 10 days"*; *"we changed the learning rate to 10⁻⁵ for the initial 100K (10⁵) steps (6 days) and increased to 512 for another 600K (6 × 10⁵) steps (20 days)"* |
| **Hardware, Precision & Distributed** | 8 $\times$ NVIDIA A100 (80 GB) GPUs for 36 days total (`10 + 6 + 20` days); mixed `bf16`; `DeepSpeed (stage-2)` | `reported` | `[Main, p. 2–3]` *"8 Nvidia A100 each with 80GB vRAM"*; *"mixed half-precision (bf16, (44))"*; *"using DeepSpeed (stage-2) (43)"* |
| **Estimated Compute (Fine-Tuning)** | `6,912` A100-80GB-hours; $\le 2.71 \times 10^{21}$ FLOPs ($6ND$ upper bound); $\le 1,904$ H100-eq hours | `estimated` | $8 \times (10 + 6 + 20) \times 24 = 6{,}912\text{ A100-h}$ (excludes base `ProtT5-XL-U50` pretraining) |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 ProstT5 Sequential Training Curriculum (`[Main, pp. 2–3]`)

*Unlike ESM-2, ProstT5 is released as a single model scale (3B encoder–decoder / 1.21B encoder) trained across three curriculum stages on top of `ProtT5-XL-U50`.*

| Stage | Objective | Corpus | Max Length | Learning Rate | Batch / Accum | Duration (Steps / Samples) | Wall-Clock (`8x A100`) |
|:---|:---|:---|---:|:---|:---|:---|:---|
| **Base (`ProtT5-XL-U50`)** | T5 span-based denoising (AA only) | BFD (2.1B) + UniRef50 | 512 | Not reported | Not reported | Inherited checkpoint | Inherited |
| **Phase 1 (Warm-up)** | T5 span-based denoising (AA + 3Di) | `train17M` (34M samples) | 256 | `1e-3` | `16` / `5` | 102M samples (~3 epochs) | 10 days |
| **Phase 2a (Translation)** | Simultaneous `<AA2fold>` & `<fold2AA>` | `train17M` (34M pairs) | 256 | `1e-5` | `16` / `1` | 100K steps | 6 days |
| **Phase 2b (Long Translation)** | Simultaneous `<AA2fold>` & `<fold2AA>` | `train17M` (34M pairs) | 512 | `1e-5` | `6` / `4` | 600K steps (700K total, ~4 epochs) | 20 days |

> *"limiting sequence lengths to 256 (truncating longer sequences)"* `[Main, p. 3]`
> *"increased to 512 for another 600K (6 × 10⁵) steps (20 days)"* `[Main, p. 3]`

### 3.2 Extracted Scientific Claims
- **`[prost5-c1 · bilingual_3di_translation]`** ProstT5-predicted 3Di strings paired with Foldseek reach remote-homology sensitivity close to experimental 3D structures on CATH (`ROC-AUC = 0.45` at superfamily level vs. `0.49` for experimental structures and `0.06` for MMseqs2 sequence search).
  > *"ProstT5 (ROC-AUC on the level of super-family: 0.45) reached performance levels close to experimental structures (ROC-AUC = 0.49)"* `[Main, p. 6]`
- **`[prost5-c2 & prost5-c8 · bilingual_3di_translation]`** Encoder + CNN 3Di prediction for the *M. jannaschii* proteome takes `43.5 s` on a single RTX A5000 GPU compared to `48 h` for ColabFold structure prediction ($> 3$ orders of magnitude speedup; ~`0.1 s` per protein on human proteome).
  > *"43.5s (±2s) on GPU (Nvidia RTX A5000), compared to the 48h structure prediction time on GPU"* `[Main, p. 6]`
  > *"requires around 35m (minutes) or 0.1s (seconds) per protein"* `[Main, p. 9]`
- **`[prost5-c3 & prost5-c9 · transfer_learning]`** On CATH embedding-based annotation transfer (EAT) using amino-acid input only (`ProstT5(AA)`), ProstT5 outperforms base `ProtT5`, `Ankh`, and `ESM-1b` both unsupervised (`73 ± 6` vs. `ProtT5` `68 ± 6`, `Ankh` `70 ± 6`, `ESM-1b` `62 ± 6`) and with contrastive supervision (`81 ± 5` vs. `ProtT5` `76 ± 5`).
  > *"ProstT5(AA) 85 ± 5 74 ± 6 64 ± 6 69 ± 7 73 ± 6"* `[Main, p. 7, Table 1]`
  > *"ProstT5(AA) 89±4 81±5 70±6 83±6 81±5"* `[SI, p. 2, Table S1]`
- **`[prost5-c4 & prost5-c6 · transfer_learning]`** Even when given amino-acid input alone (`ProstT5(AA)`), bilingual training improves secondary-structure (`Q3`) and fold-coupled predictions over base `ProtT5`.
  > *"Even without leveraging 3Di-information (ProstT5(AA) used only AA input), bilingual ProstT5 improved over its base model ProtT5"* `[Main, p. 7]`
  > *"tasks strictly related to a protein’s structure or fold are very likely to improve from the new embeddings."* `[Main, p. 11]`
- **`[prost5-c5 · transfer_learning]`** Conversely, on general function/motif benchmarks—most notably subcellular localization (`LightAttention` on `DeepLoc`)—`ProstT5(AA)` underperforms base `ProtT5`, leading the authors to caution that *"ProstT5 is not a general-purpose pLM"*.
  > *"a clear decrease in predicting subcellular location when inputting exclusively amino acid sequences"* `[Main, p. 7]`
- **`[prost5-c7 · inverse_folding]`** Sequences generated via `3Di -> AA` inverse folding (`<fold2AA>`) share only $21 \pm 3\%$ sequence identity with native proteins while adopting the target fold (`average lDDT = 72` via ESMFold).
  > *"these were all predicted to adopt similar structures (average lDDT = 72)"* `[Main, p. 8]`
- **`[prost5-c10 · ablation]`** A 2-layer CNN trained directly on top of the frozen ProstT5 encoder's amino-acid embeddings outperforms full autoregressive T5 decoding for 3Di prediction on the hardest benchmark (`CASP14`) while running $> 50\times$ faster.
  > *"the CNN trained on top of ProstT5’s encoder was better on the arguably hardest set, CASP14 (lowest bar)"* `[SI, p. 5]`

---

## 4. Paper-Reported Limitations, Biases & Caveats

1. **Bias against intrinsically disordered proteins ($\text{pLDDT} \ge 70$ filter) & dilution of 1D sequence-motif tasks:** Filtering AFDB for $\text{pLDDT} \ge 70$ removes most intrinsically disordered proteins and biases the representation toward well-packed globular folds, degrading subcellular localization and non-structural function tasks relative to base `ProtT5` (the authors suggest concatenating `ProtT5` + `ProstT5` embeddings when both sequence motifs and 3D structure matter).
   > *"filtering the AFDB for predictions with high pLDDT removes most intrinsically disordered proteins"* `[Main, p. 11]`
   > *"a clear decrease in predicting subcellular location when inputting exclusively amino acid sequences"* `[Main, p. 7]`
2. **Severe 3Di token class imbalance:** In `train17M`, two of the 20 Foldseek 3Di states (`v` and `d`, predominantly $\alpha$-helical) account for roughly 40% of all residues (`[SI, p. 1, Fig. S1]`).
   > *"class imbalance remained high"* `[Main, p. 11]`
3. **Training sequence truncation at 256 / 512 tokens:** Because `train17M` has a mean length of ~238 residues and fine-tuning truncated sequences at 256 tokens (Phases 1 & 2a) and 512 tokens (Phase 2b), proteins longer than 512 residues were never seen in full during translation training.
   > *"limiting sequence lengths to 256 (truncating longer sequences)"* `[Main, p. 3]`
4. **Potential circularity on AlphaFold/Foldseek-derived structure benchmarks:** Because `train17M` distills AlphaFold2 predictions discretized by Foldseek, evaluations using Foldseek or AF2-derived labels carry potential circularity.
   > *"Another problem might be the circularity and potential information leakage"* `[Main, p. 11]`

---

## 5. Enzymado Embedding & Probing Specification

*(Status: `code` — deterministic pipeline from `SUPPORTING_CONTEXT_MODELS.md` and `embedding_scripts/prost5_embedding_script.py`)*

- **Enzymado Key & Checkpoint:** `prost5` $\rightarrow$ Hugging Face `Rostlab/ProstT5` (`1,210M` encoder parameters in registry, $D = 1{,}024$; decoder discarded).
- **Model Loader:** `T5EncoderModel.from_pretrained("Rostlab/ProstT5")` and `T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)`.
- **Input Modality & Direction Prefix:** Amino-acid sequence mode (`--is_3Di 0`). Each sanitized sequence is space-separated and prefixed with the AA-to-structure translation token: `seq = "<AA2fold> " + " ".join(list(seq))`.
- **Non-Standard Amino Acid Sanitization:** Rare/ambiguous residues `U`, `Z`, and `O` are mapped to unknown `X` before tokenization (`seq.replace("U", "X").replace("Z", "X").replace("O", "X")`).
- **Tokenization & Special-Token Slicing:** Tokenized via `vocab(seqs, add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")`. Index `0` is the `<AA2fold>` prefix token and index `s_len + 1` is `</s>` (EOS); Enzymado slices `emb = embedding_repr.last_hidden_state[batch_idx, 1 : s_len + 1]` from the final (24th) encoder layer so that strictly the $L$ amino-acid residue representations $\mathbf{E} \in \mathbb{R}^{L \times 1024}$ are retained (matching `[Main, p. 6]`: *"we extracted the hidden states of the encoder’s last layer"*).
- **Storage Format:** Saved in `bfloat16` (`BF16`) safetensors shards both as per-residue matrices $\mathbf{E} \in \mathbb{R}^{L \times 1024}$ (`SafetensorsShardWriter`) and as mean-pooled sequence vectors $\mathbf{e}_{\text{prot}} = \frac{1}{L}\sum_{i=1}^L \mathbf{E}_i \in \mathbb{R}^{1024}$ (`SafetensorsPooledWriter`).
- **Downstream Probing Heads Evaluated (5 heads across 140 runs per PLM):**
  1. `linear` (`pooling="no_attention"`, `head="linear"`): Ridge / L2-regularized linear or logistic/SGD probe over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1024}$.
  2. `mlp` (`pooling="no_attention"`, `head="mlp"`): 2-layer feed-forward neural network with normalization, activation, and dropout over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1024}$.
  3. `xgboost` (`pooling="no_attention"`, `head="xgboost"`): Gradient-boosted decision trees (`XGBRegressor` / `XGBClassifier`) over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1024}$.
  4. `attention_linear` (`pooling="attention"`, `head="linear"`): Learned per-residue attention pooling (`ResidualAttentionAggregation` for global tasks; `AttentionResidueMixer` for residue tasks) trained jointly with a linear readout head over $\mathbf{E} \in \mathbb{R}^{L \times 1024}$.
  5. `attention_mlp` (`pooling="attention"`, `head="mlp"`): Learned per-residue attention pooling trained jointly with a 2-layer MLP readout head over $\mathbf{E} \in \mathbb{R}^{L \times 1024}$.
- **Task-Layout Conditioning:**
  - **Layout A (`protein_global`):** Uses $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1024}$ (concatenated with standardized reaction conditions such as temperature, pH, and buffer one-hot indicators when present).
  - **Layout B (`protein_ligand_global`):** Concatenates $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1024}$ with 200 normalized RDKit 2D substrate descriptors (`descriptastorus` `RDKit2DNormalized`).
  - **Layout C (`mutation_delta_global`):** Uses the wild-type subtraction vector $\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}} \in \mathbb{R}^{1024}$ for point-mutation thermostability ($\Delta T_m$, $\Delta\Delta G$) tasks.
  - **Layout D (`protein_residue`):** Predicts per-residue binary labels $\{0, 1\}^L$ directly from $\mathbf{E} \in \mathbb{R}^{L \times 1024}$.

---

## 6. Provenance Audit

- **Source PDFs:** `model_prost5_1_main.pdf` (15 pages; combined pages 1–15) and `model_prost5_2_suppinfo.pdf` (15 pages; combined pages 16–30 in `model_2_prost5_main_and_si.pdf`).
- **Quote Verification:** `63 / 63` quotes verified (`0` misses) via normalized substring match against `pdftotext` and `pdftotext -layout` text layers of the cited standalone page.
