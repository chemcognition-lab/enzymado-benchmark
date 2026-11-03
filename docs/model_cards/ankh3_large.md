---
model_id: ankh3-large
enzymado_key: ankh3_large
model_name: Ankh3-Large
size_variant: '72L enc-dec (48L encoder used, 1.15B encoder / 1.88B total params, D=1,536)'
domain: protein
license: open_source_huggingface
created_at: '2026-11-03'
datasets:
  - UniRef50
tags:
  - protein
  - transformer
  - t5
  - ul2
  - protein_language_model
  - enzymado
base_model: ElnaggarLab/ankh3-large
doi: 10.48550/arXiv.2505.20052
code: https://huggingface.co/ElnaggarLab/ankh3-large
---

# Model Card: Ankh3-Large (ElnaggarLab/ankh3-large, encoder only)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and 1-based page citation (`[Main, p. X]`) verified against `docs/pdfs/ankh3_alsamkary2025_arxiv.pdf` (8 pp., *arXiv:2505.20052v1*). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from the manuscript), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings, so it can be paired with any Dataset Card for mechanistic modeling analysis without leaking benchmark results.

---

## 0. How to think about this model

**Ankh3-Large (ElnaggarLab/ankh3-large, encoder only)** (Enzymado key `ankh3_large`) is the 48-layer, 1,536-dimensional **SwiGLU** encoder (`1,151,879,680` encoder parameters out of `1,879,049,216` total encoder–decoder parameters; `1,150M` in Enzymado's registry) of a 72-layer T5-style encoder–decoder (48 encoder layers + 24 decoder layers) pre-trained on **59,130,945 UniRef50** sequences (`agemagician/uniref50`) for **≈ 4M steps** (batch size 1,024, sequence length 512) using two UL2-inspired objectives: short-span variable sentinel denoising (`[NLU]`, uniformly sampling **15%, 20%, and 30%** masking probabilities) and 50/50 prefix sequence completion (`[S2S]`). Crucially, `Ankh3-Large` shares the exact layer count (48 encoder + 24 decoder), hidden width ($D = 1{,}536$), FFN width ($d_{\text{ffn}} = 3{,}840$), and head count (16 heads, KV dim 64) of `ankh_large` (~1.15B encoder parameters), replacing Gated-GELU with SwiGLU and single-task 20% denoising with multi-task `[NLU]` (15%/20%/30%) + `[S2S]` (50% completion) pre-training—though the authors explicitly note that `Ankh3-Large` lacked sufficient capacity to handle both tasks simultaneously compared to `Ankh3-XL` (3.48B encoder).

- **Critical Enzymado extraction detail (unprefixed sequences):** In the Ankh3 paper (`[Main, p. 5, Section 3.1]`), downstream benchmarks concatenate either the `[NLU]` or `[S2S]` prefix token at the beginning of each sequence. In contrast, Enzymado's `embedding_scripts/ankh_embedding_script.py` passes `[list(s) for s in seqs]` with `is_split_into_words=True` **without** prepending `[NLU]` or `[S2S]` prefix tokens, and slices `outputs.last_hidden_state[batch_idx, 0:s_len]` (stripping only the trailing `</s>` EOS token).
- **Best-suited dataset regimes:**
  - Controlled architectural comparisons against `ankh_large` (identical 48L, $D = 1{,}536$, $d_{\text{ffn}} = 3{,}840$, 1.15B encoder footprint) and `ankh3_xl` (scaled 3.48B encoder, $D = 2{,}560$) to isolate the effect of UL2 multi-task pre-training (`[NLU]` + `[S2S]`) and SwiGLU activation at fixed 1.15B encoder scale.
  - Protein families well-covered by the expanded 59.13M-sequence UniRef50 corpus (`agemagician/uniref50`), benefiting from uniform exposure to 15%, 20%, and 30% corruption rates across 4M–5M steps.
  - Global enzyme property tasks (`tm`, `topt`, `phopt`, `solubility`, `kcat`, `km`) where Ankh3's own benchmark protocol used **global average pooling** (`Table 4`, unlike Ankh v1's global max-pooling), aligning directly with Enzymado's mean-pooled probes (`linear`, `mlp`, `xgboost`).
- **Expected vulnerabilities & blind spots:**
  - **Unprefixed inference shift in Enzymado:** Because Ankh3 was trained with `[NLU]` and `[S2S]` task prefixes and evaluated with those prefixes in the paper, querying the encoder on raw unprefixed sequences (`[list(s) for s in seqs]`) leaves the task-conditioning slot unpopulated.
  - **Capacity bottleneck under dual-task UL2 pre-training:** The Ankh3 authors explicitly report (`[Main, p. 5, Section 4]`) that `Ankh3-Large` (1.15B encoder) had insufficient capacity to handle both `[NLU]` and `[S2S]` objectives simultaneously, causing it to trail `ankh_large` and `ankh_base` on CASP12 secondary structure (`78.03%` vs. `83.59%` / `80.81%`, `Table 5`) while improving on CASP14, fluorescence, and GB1.
  - **Context window beyond 512 residues:** Pre-trained with `Sequence length: 512` (`Table 3`); proteins longer than 512 residues rely on multi-layer composition of 128-token relative positional biases across 48 layers.
- **Downstream probe interaction summary:** Exposes a $D = 1,536$ per-residue representation from the 48th SwiGLU encoder layer (queried without `[NLU]`/`[S2S]` prefix tokens and with trailing `</s>` stripped via `[0:s_len]`). Unlike Ankh v1 (which used global max-pooling before ConvBERT), the Ankh3 paper used global average pooling for sequence-level tasks (`Table 4`), matching Enzymado's mean-pooling for `linear`, `mlp`, and `xgboost`, while `attention_linear` and `attention_mlp` learn task-specific residue weighting over the $L \times 1,536$ tensor.

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Pre-trained on two of the three UL2 (Tay et al., 2022) objectives: (1) **R-denoiser (`[NLU]`)**, short-span T5 sentinel denoising where each training step uniformly samples a masking probability from **15%, 20%, and 30%**, and (2) **S-denoiser (`[S2S]`)**, sequential completion where the first 50% of a protein sequence is fed to the encoder and the remaining 50% is autoregressively generated by the decoder conditioned on the encoder states. The UL2 X-denoiser (`[NLG]`, extreme long-span corruption) was excluded because long-span masking degraded performance in prior Ankh ablations. Each sequence in a batch is randomly assigned to either MLM or 50% completion on the fly.
  > *"X-denoiser was not used due to previous experiences of long span masking in former Ankh models that resulted in poor performance. Hence, this research only focuses on short-span sequence variable denoising/demasking [NLU] and sequence completion/sequential denoising [S2S]."* `[Main, p. 1]`
  > *"While conventional masked language modeling traditionally utilizes a single masking probability, the training here employed three distinct masking probabilities: 15%, 20%, and 30%. These probabilities governed the proportion of masked tokens and were sampled uniformly to ensure that each masking level had an equal chance of selection in every training step."* `[Main, p. 3]`
  > *"For the multi-task training setup, each sequence from the dataset was randomly assigned to either the MLM or sequence completion task each time it was selected for a training batch."* `[Main, p. 3]`
- **Downstream task implications:** Uniformly sampling 15%, 20%, and 30% masking probabilities exposes the encoder to both fine-grained local conservation (15%) and coarse structural contact constraints (30%), while 50/50 sequence completion forces the encoder representation of the N-terminal half to summarize global fold/family identity sufficient to generate the C-terminal half. Note that in Enzymado, the encoder is queried without the `[NLU]` or `[S2S]` mode-switching prefix token.

### 1.2 Pretraining corpus filtering & sampling
- **Mechanism:** Pre-trained on **59,130,945 distinct sequences** from **UniRef50** (`https://huggingface.co/datasets/agemagician/uniref50`), an expanded ~59.1M-cluster snapshot compared to the 45M-cluster UniRef50 snapshot used for ProtT5 and Ankh v1. Clustered hierarchically from UniRef100 $\to$ UniRef90 (90% identity) $\to$ UniRef50 (50% sequence identity cutoff) to minimize sequence redundancy.
  > *"Inspired by its predecessors Ankh (Elnaggar et al., 2023) and ProtTrans (Elnaggar et al., 2021), the protein language model Ankh3 underwent pre-training with a substantial dataset comprising 59,130,945 distinct sequences sourced from UniRef50."* `[Main, p. 2]`
  > *"Subsequently, the UniRef50 database is created by further clustering the representative sequences from UniRef90 (which is built by clustering UniRef100 sequences at a 90% identity threshold) using a 50% sequence identity cutoff (Suzek et al., 2015)."* `[Main, p. 3]`
  > *"The dataset used for pre-training is available at https://huggingface.co/datasets/agemagician/uniref50."* `[Main, p. 6]`
- **Downstream task implications:** Provides ~31% more non-redundant UniRef50 sequence clusters than Ankh v1 (59.13M vs. 45M) and trains for $\approx 4\text{M}$ (`Ankh3-Large`) or $\approx 5\text{M}$ (`Ankh3-XL`) steps at batch size 1,024 (~69–87 passes over the 59.13M sequences), yielding broad coverage of UniProtKB protein families.

### 1.3 Training crop window & positional encoding
- **Mechanism:** Pre-trained with `Sequence length: 512` (`Table 3`) and T5 relative positional bias configured with `Relative attention max distance: 128` and `Relative attention num buckets: 64` across 48 encoder layers (`Table 1`). Under the `[S2S]` objective, the encoder receives the first 50% of the sequence (up to 256 tokens during pre-training) and the decoder generates the remaining 50%.
  > *"Relative attention max distance 128 128"* `[Main, p. 2, Table 1]`
  > *"Relative attention num buckets 64 64"* `[Main, p. 2, Table 1]`
  > *"Sequence length 512 512"* `[Main, p. 3, Table 3]`
- **Downstream task implications:** Enzymado embeds full-length protein sequences without truncation; relative distances up to 128 residues are resolved directly in each attention head, while longer-range interactions ($>128$ and $>512$ residues) are propagated hierarchically across the 48 encoder layers.

### 1.4 Embedding width $D$, pretraining dropout & downstream probe parameterization
- **Mechanism:** Uses a 48-layer encoder + 24-layer decoder T5 architecture with **SwiGLU** non-linearity (`Feedforward dim: 3840`), key–value head dimension `KV dim: 64`, `16` attention heads, untied word embeddings (`Tie word embeddings: no`), and hidden dimension `Embedding dim: 1536` (`1,151,879,680` encoder parameters, `Tables 1–2`). Critically, adding the second pre-training task (`[S2S]` sequence completion) plus multi-rate masking prevented the scaling collapse previously observed when scaling ProtT5-XL to ProtT5-XXL, allowing `Ankh3-XL` (3.48B encoder) to substantially outperform `Ankh3-Large` (1.15B encoder).
  > *"Finally, because one more task was added during the pre-training, the increase in the capacity of Ankh3-XL model did not lead to performance degradation as in ProtT5 (Elnaggar et al., 2021) when scaled from ProtT5-XL to ProtT5-XXL while all hyperparameters were fixed, including the dataset."* `[Main, p. 2]`
  > *"In the context of scaling Ankh3-Large to Ankh3-XL, Ankh3-XL performed significantly better in both NLU and S2S tasks, indicating that the capacity of Ankh3-Large was insufficient to handle both tasks."* `[Main, p. 5]`
- **Downstream task implications:** With $D = 1{,}536$ and 1.15B encoder parameters, `ankh3_large` matches `ankh_large` in width and depth, but the authors explicitly note that 1.15B encoder parameters proved insufficient to simultaneously absorb both `[NLU]` and `[S2S]` pre-training objectives (`[Main, p. 5]`). Moreover, because Enzymado omits the `[NLU]`/`[S2S]` prefix token at extraction time, comparing `ankh3_large` against `ankh_large` and `ankh3_xl` directly tests both capacity scaling and unprefixed representation quality.

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | Proteinea Inc. | `reported` | `[Main, p. 1]` *"Hazem Alsamkary , Mohamed Elshaffei , Mohamed Elkerdawy, and Ahmed Elnaggar∗ Proteinea Inc."* |
| **Publication** | Alsamkary et al. (2025), *arXiv:2505.20052v1* | `reported` | `[Main, p. 1]` *"arXiv:2505.20052v1 [cs.LG] 26 May 2025"* |
| **Code & Weights** | `https://huggingface.co/ElnaggarLab/ankh3-large` | `reported` | `[Main, p. 6]` *"Model weights of both Ankh3-Large and Ankh3-XL models are available at https://huggingface.co/ ElnaggarLab/ankh3-large and https://huggingface.co/ElnaggarLab/ankh3-xl."* |
| **License** | Open source on Hugging Face (exact SPDX string not printed in paper) | `interpreted` | `[Main, p. 6]` *"Finally, the weights of both Ankh3 models are available open source on Huggingface."* |
| **Base Checkpoint** | None (trained from scratch on UniRef50) | `not_reported` | Pre-trained from scratch in T5x/JAX on TPUv4-64. |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | Asymmetric T5 encoder–decoder Transformer (48L encoder used) | `reported` | `[Main, p. 2]` *"As in preceding Ankh models, Ankh3 is based on T5 architecture (Raffel et al., 2020) with the model configuration mentioned in table 1, an encoder-decoder transformer model with T5 relative positional bias."* |
| **Total Parameters** | `1,151,879,680` encoder / `727,169,536` decoder / `1,879,049,216` total (`1,150M` in Enzymado registry) | `reported` | `[Main, p. 2, Table 2]` *"Ankh3-Large 1,151,879,680 727,169,536 1,879,049,216"* |
| **Layers ($L_{\text{layers}}$)** | 48 encoder layers (+ 24 decoder layers = 72 total) | `reported` | `[Main, p. 2, Table 1]` *"Layers 72 72"*; *"Encoder layers 48 48"*; *"Decoder layers 24 24"* |
| **Hidden Dim ($D$)** | 1,536 | `reported` | `[Main, p. 2, Table 1]` *"Embedding dim 1536 2560"* |
| **FFN Dim ($d_{\text{ffn}}$)** | 3,840 | `reported` | `[Main, p. 2, Table 1]` *"Feedforward dim 3840 6720"* |
| **Attention Heads ($H$)** | 16 (`KV dim: 64`) | `reported` | `[Main, p. 2, Table 1]` *"Num heads 16 32"*; *"KV dim 64 64"* |
| **Activation Function** | `SwiGLU` | `reported` | `[Main, p. 2, Table 1]` *"Non-linearity SwiGLU SwiGLU"* |
| **Positional Encoding** | T5 relative positional bias (`max distance: 128`, `num buckets: 64`) | `reported` | `[Main, p. 2, Table 1]` *"Relative attention max distance 128 128"*; *"Relative attention num buckets 64 64"* |
| **Max Training Length** | 512 tokens | `reported` | `[Main, p. 3, Table 3]` *"Sequence length 512 512"* |
| **Vocabulary & Embeddings** | `Vocab size: 256`, `Num sentinel tokens: 225`, `Tie word embeddings: no` | `reported` | `[Main, p. 2, Table 1]` *"Tie word embeddings no no"*; *"Num sentinel tokens 225 225"*; *"Vocab size 256 256"* |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objective** | UL2 R-denoiser `[NLU]` (15%, 20%, 30% uniform masking) + S-denoiser `[S2S]` (50% sequence completion) | `reported` | `[Main, p. 1]` *"Hence, this research only focuses on short-span sequence variable denoising/demasking [NLU] and sequence completion/sequential denoising [S2S]."* |
| **Pretraining Corpus** | UniRef50 (`59,130,945` distinct sequences, `agemagician/uniref50`) | `reported` | `[Main, p. 2]` *"comprising 59,130,945 distinct sequences sourced from UniRef50."* |
| **Steps & Batch Size** | `≈ 4M` steps $\times$ batch size `1024` at sequence length `512` | `reported` | `[Main, p. 3, Table 3]` *"Num steps ≈ 4M ≈ 5M"*; *"Batch size 1024 1024"* |
| **Optimizer & Schedule** | `Adafactor`, peak LR `1e-2`, `rsqrt decay` scheduler, `10K` warmup steps, weight decay `0.0` | `reported` | `[Main, p. 3, Table 3]` *"Warmup scheduler rsqrt decay rsqrt decay"*; *"Warmup steps 10K 10K"*; *"Learning rate 1e-2 1e-2"*; *"Optimizer Adafactor Adafactor"*; *"Weight decay 0.0 0.0"* |
| **Dropout & Precision** | Backbone pre-training dropout and numeric precision not stated in `Tables 1–3` | `not_reported` | Only downstream ConvBERT dropout (`0.1`) is listed in `Table 4` (`[Main, p. 4]`). |
| **Hardware & Parallelism** | `TPUv4` with `64` chips in `T5x`/`JAX` (no replication or sharding; optimizer states unsharded) | `reported` | `[Main, p. 3]` *"Both Ankh3 models were trained utilizing TPUv4 with 64 chips."*; *"Ankh3-Large was trained without replication or sharding, while Ankh3-XL employed both 2-way data sharding and 2-way model sharding. Optimizer states were not sharded in either model."* |
| **Estimated Compute** | Wall-clock time not reported; $\approx 2.36 \times 10^{22}$ FLOPs ($6ND$ upper bound at length 512); `16,590` H100-eq hours | `estimated` | $6 \times N_{\text{total}} \times (\text{steps} \times 1024 \times 512) \approx 2.36 \times 10^{22}\text{ FLOPs}$ |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 Ankh3 Family Scaling & Prefix Evaluation Summary (`[Main, pp. 2, 3, 5, Tables 1–3 & 5]`)

| Variant | Encoder / Decoder Layers | $D$ (`d_model`) | $d_{\text{ffn}}$ | Heads | Encoder Params | Total Params | Steps (`BS=1024`) | TPUv4-64 Sharding | CASP12 SSP-3 (`[NLU]` / `[S2S]`) | FLIP GB1 (`[NLU]` / `[S2S]`) |
|:---|:---|---:|---:|---:|---:|---:|:---|:---|:---|:---|
| `Ankh3-Large` (`ankh3_large`) | 48 / 24 | 1,536 | 3,840 | 16 | `1,151,879,680` | `1,879,049,216` | `≈ 4M` | None | `78.03%` / `75.49%` | `90.30` / `89.44` |
| `Ankh3-XL` (`ankh3_xl`) | 48 / 24 | 2,560 | 6,720 | 32 | `3,484,799,488` | `5,730,907,136` | `≈ 5M` | 2-way data + 2-way model | `84.40%` / `83.76%` | `89.62` / `90.44` |

> *"Ankh3-Large 1,151,879,680 727,169,536 1,879,049,216"* `[Main, p. 2, Table 2]`
> *"Ankh3-XL 3,484,799,488 2,246,107,648 5,730,907,136"* `[Main, p. 2, Table 2]`
> *"SSP-3 CASP-12 78.03 ± 0.11 75.49 ± 0.21 84.40 ± 0.05 83.76 ± 0.10 80.81 83.59 82.43 83.16 83.43 ± 0.02"* `[Main, p. 5, Table 5]`

### 3.2 Extracted Scientific Claims
- **`[ankh3-c1 · pretraining_objective]`** Ankh3 uses only two of the three UL2 objectives—short-span variable denoising (`[NLU]`) and 50% sequence completion (`[S2S]`)—explicitly omitting the X-denoiser (`[NLG]`) because long-span masking caused poor performance in prior Ankh experiments.
  > *"X-denoiser was not used due to previous experiences of long span masking in former Ankh models that resulted in poor performance. Hence, this research only focuses on short-span sequence variable denoising/demasking [NLU] and sequence completion/sequential denoising [S2S]."* `[Main, p. 1]`
- **`[ankh3-c2 · scaling]`** Combining multiple masking probabilities (`15%`, `20%`, `30%`) and 50% sequence completion prevented the scaling degradation previously observed when scaling ProtT5-XL (`81.4%` CASP12) to ProtT5-XXL (`79.2%` CASP12), enabling `Ankh3-XL` (3.48B encoder) to reach `84.4%` on CASP12 with protein sequences as the sole input.
  > *"for example, CASP12 reached 84.4% in Ankh3-XL compared to 79.2% in ProtT5-XXL (4.8B encoder parameters) and 81.4% in ProtT5-XL (1.2B encoder parameters), this indicates that the addition of multiple masking probabilities and sequence completion were the main contributors to the performance boost while using protein sequences as the only input."* `[Main, p. 5]`
- **`[ankh3-c3 · scaling_capacity]`** `Ankh3-XL` significantly outperformed `Ankh3-Large` across both `[NLU]` and `[S2S]` modes, indicating that `Ankh3-Large`'s 1.15B encoder capacity was insufficient to jointly learn both pre-training tasks.
  > *"In the context of scaling Ankh3-Large to Ankh3-XL, Ankh3-XL performed significantly better in both NLU and S2S tasks, indicating that the capacity of Ankh3-Large was insufficient to handle both tasks."* `[Main, p. 5]`
- **`[ankh3-c4 · prefix_conditioning]`** Neither `[NLU]` nor `[S2S]` prefix conditioning universally outperformed the other across all tasks and model sizes, though `Ankh3-XL` achieved slightly higher scores with `[S2S]` on sequence-level fitness benchmarks (`GB1` `90.44` vs. `89.62`; `Fluorescence` `65.43` vs. `64.23`).
  > *"While NLU and S2S tasks are different, neither of them consistently performed better than the other across all the tasks and model sizes. The preferred objective seems to be task-dependent. One interesting observation that requires deeper investigation is that Ankh3-XL performed better with S2S in sequence classification tasks, such as GB1 and fluorescence (Table 5)."* `[Main, p. 5]`
- **`[ankh3-c5 · multimodal_comparison]`** Compared to the multimodal `ESM3-open` model, sequence-only `Ankh3-XL` achieved competitive accuracy on secondary structure and contact prediction and substantially higher generalization on unseen fitness benchmarks (`TAPE Fluorescence` `65.43` vs. `58.78`; `FLIP GB1` `90.44` vs. `82.91`) with much greater stability across random seeds.
  > *"When compared to ESM3, Ankh3 has competitive performance on tasks such as SSP and CP. However, in tasks that neither Ankh3 nor ESM3 encountered during pre-training, Ankh3-XL performed significantly better, which indicates that Ankh3-XL may have better generalization, especially in tasks like fluorescence and GB1 fitness prediction."* `[Main, p. 5]`

---

## 4. Intended Use, Limitations & Caveats

- **Paper-stated intended uses:** Frozen backbone feature extraction for per-residue and global protein property prediction (secondary structure, contact maps, fluorescence, GB1 epistatic fitness) and conditional sequence completion (`[Main, pp. 2–4]`).
- **Enzymado benchmark role:** Frozen 48-layer encoder feature extractor loaded via `T5EncoderModel.from_pretrained("ElnaggarLab/ankh3-large")` in `embedding_scripts/ankh_embedding_script.py`, producing per-residue ($L \times 1,536$) and mean-pooled ($1,536$) BF16 representations probed across 5 downstream head types.
- **Out-of-scope / unvalidated uses:**
  - Decoder-based 50% sequence completion (`T5ForConditionalGeneration`) is not used in Enzymado.
  - Prefix-conditioned feature extraction (`[NLU]` vs. `[S2S]` token prepending) is not used in Enzymado's `ankh_embedding_script.py`.
- **Paper-stated limitations & caveats:**
  1. **Fixed 50/50 completion split due to compute limits:**
     > *"Although variable completion percentages may enhance the performance of the model, it was not experimented in this work due to the limited computation power."* `[Main, p. 2]`
  2. **Capacity bottleneck in `Ankh3-Large` under multi-task UL2 training:**
     > *"In the context of scaling Ankh3-Large to Ankh3-XL, Ankh3-XL performed significantly better in both NLU and S2S tasks, indicating that the capacity of Ankh3-Large was insufficient to handle both tasks."* `[Main, p. 5]`
  3. **Prefix conditioning (`[NLU]` vs. `[S2S]`) in paper vs. unprefixed sequences in Enzymado:**
     > *"The performance of the PLMs was tested in two different settings: the first setting is concatenating the [NLU] token at the beginning of each sequence, and the second setting is concatenating the [S2S] token at the beginning of each sequence."* `[Main, p. 5]`

---

## 5. Enzymado Evaluation Specification (`status: code`)

| Field | Specification |
|:---|:---|
| **Enzymado Key** | `ankh3_large` |
| **HuggingFace Checkpoint** | `ElnaggarLab/ankh3-large` |
| **Embedding Script** | `embedding_scripts/ankh_embedding_script.py` |
| **Model & Tokenizer Loader** | `T5EncoderModel.from_pretrained("ElnaggarLab/ankh3-large")` + `AutoTokenizer.from_pretrained("ElnaggarLab/ankh3-large")`, `.eval()` on GPU |
| **Extracted Layer & Shape** | Final (48th) encoder layer (`outputs.last_hidden_state`), residue slice `[0:s_len]` $\to$ `(L, 1536)` |
| **Non-Standard AA Handling** | `U`, `Z`, `O` replaced with `X` (`seq.replace("U", "X").replace("Z", "X").replace("O", "X")`) |
| **Input Formatting** | List of single amino-acid characters per sequence (`[list(s) for s in seqs]`) **without** `[NLU]` or `[S2S]` prefix tokens |
| **Tokenization Call** | `tokenizer([list(s) for s in seqs], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")` |
| **Special Token Slicing** | T5 tokenizer appends `</s>` (EOS) at index `s_len` with no leading BOS or task prefix token (`[NLU]`/`[S2S]`); `outputs.last_hidden_state[batch_idx, 0:s_len]` strips the trailing `</s>` token |
| **Saved Storage Format** | BF16 safetensors shards: per-residue `(L, 1536)` via `SafetensorsShardWriter` and mean-pooled `(1536,)` via `SafetensorsPooledWriter` (`emb.mean(dim=0)`) |
| **Consistency with Paper** | Matches the paper's frozen 48-layer encoder extraction and global average pooling for sequence-level tasks (`[Main, pp. 3–4]`), **except** that the Ankh3 paper prepended `[NLU]` or `[S2S]` to each sequence (`[Main, p. 5, Section 3.1]`) whereas `ankh_embedding_script.py` passes raw unprefixed residue lists `[list(s) for s in seqs]` |
| **Probing Heads Evaluated** | `linear`, `mlp`, `xgboost` (over mean-pooled embeddings); `attention_linear`, `attention_mlp` (learned residue-level attention pooling over $L \times 1536$) |
| **Task Layout Conditioning** | Layout A: $f(e_{\text{prot}})$ or $f(e_{\text{prot}}, \text{conditions})$; Layout B: concat with 200-dim `RDKit2DNormalized` ligand features (`kcat`, `km`, `binding`, `cleeveland_galox`); Layout C: $e_{\text{mut}} - e_{\text{wt}}$ (`dtm`, `ddg`); Layout D: per-residue binary (`active_sites`, `binding_site`) |
| **Total Evaluation Runs** | 140 per PLM (10 `properties` tasks $\times$ 2 splits $\times$ 5 heads + 8 `real_case` tasks $\times$ 5 heads) |

---

## 6. Provenance Audit Summary

- **Source PDF:** `docs/pdfs/ankh3_alsamkary2025_arxiv.pdf` (8 pp., *arXiv:2505.20052v1*, cited as `main` pp. 1–8)
- **Automated Quote Verification:** `71 / 71` quotes verified (`0` misses) via normalized text-layer substring matching (`verify_cards.py`).
- **Status Distribution:** `reported=25`, `interpreted=18`, `not_reported=5`, `estimated=2`.
