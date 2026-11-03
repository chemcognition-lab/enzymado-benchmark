---
model_id: ankh-large
enzymado_key: ankh_large
model_name: Ankh_large
size_variant: '72L enc-dec (48L encoder used, ~1.15B encoder params, D=1,536)'
domain: protein
license: CC BY-NC-SA 4.0
created_at: '2026-11-03'
datasets:
  - UniRef50
tags:
  - protein
  - transformer
  - t5
  - protein_language_model
  - enzymado
base_model: ElnaggarLab/ankh-large
doi: 10.48550/arXiv.2301.06568
code: https://github.com/agemagician/Ankh
---

# Model Card: Ankh_large / Ankh (ElnaggarLab/ankh-large, encoder only)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and page citation (`[Main, p. X]` for pp. 1–23 or `[SI, p. Y]` for Section 7 Supplementary Materials pp. 24–29 of `ankh_elnaggar2023_arxiv.pdf`). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from paper and SI), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings, so it can be paired with any Dataset Card for mechanistic modeling analysis without leaking benchmark results.

---

## 0. How to think about this model

**Ankh_large / Ankh (ElnaggarLab/ankh-large, encoder only)** (Enzymado key `ankh_large`) is the 48-layer, 1,536-dimensional bidirectional encoder (~1,150M encoder parameters) of an asymmetric 72-layer T5-style encoder–decoder (48 encoder layers + 24 decoder layers) developed through 23 protein-specific knowledge-guided ablation experiments on Google TPU-v4 pods. Pre-trained for **68 epochs** on **45 million UniRef50** clusters (14B amino acids, maximum length 512) using **1-gram span partial de-masking** (reconstructing consecutive unmasked residues as a single merged sentinel token) at a **20% masking probability** with **Gated-GELU** feed-forward layers ($d_{\text{ffn}} = 3,840$) and untied embeddings, `ankh_large` (referred to in the paper simply as `Ankh`) doubles the hidden dimension of `ankh_base` from $D = 768$ to $D = 1{,}536$ (with 16 attention heads and $d_{\text{ffn}} = 3{,}840$) to maximize structural and functional representation capacity while remaining runnable on a single 40 GB GPU.

- **Best-suited dataset regimes:**
  - Deep structural and fold-sensitive enzyme properties (secondary/tertiary contact geometry, CATH fold recognition, solubility, and epistatic fitness landscapes such as GB1 and GFP) that benefit from an unusually deep 48-layer bidirectional encoder.
  - Natural enzyme families well-represented in UniRef50 (45M clusters trained for 68 full epochs), where 20% single-residue masking with merged unmasked-span reconstruction balances local site conservation (favored by 15% masking) and long-range contact recovery (favored by 30% masking).
  - Downstream probing regimes with sufficient sample size or regularization to exploit a rich $D = 1{,}536$ representation, or per-residue tasks (`active_sites`, `binding_site`) where 48 layers of contextualization refine active-site chemistry.
- **Expected vulnerabilities & blind spots:**
  - Proteins substantially longer than the 512-token pre-training maximum length (`Maximum length: 512`), which must rely on multi-layer composition of 128-token-offset relative positional biases.
  - Orphan metagenomic sequences absent from UniRef50 (unlike `prot5_xl_uniref50`, Ankh was never pre-trained on the 2.1B-sequence BFD metagenomic corpus).
  - Potential pooling-mismatch sensitivity on global tasks: the Ankh authors benchmarked protein-level regression and binary classification using a ConvBERT head preceded by **global max-pooling**, whereas Enzymado evaluates **mean-pooled** vectors (`linear`, `mlp`, `xgboost`) and learned **attention pooling** (`attention_linear`, `attention_mlp`).
- **Downstream probe interaction summary:** Outputs a $D = 1,536$ per-residue representation from a 48-layer Gated-GELU encoder. Because the Ankh paper designed its protein-level benchmark head around a 1-layer ConvBERT network preceded by global max-pooling, Enzymado's learned per-residue attention pooling (`attention_linear`, `attention_mlp`) can capture localized high-activation motifs that uniform global mean-pooling (`linear`, `mlp`, `xgboost`) may average out across sequence length.

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Through ablations over six masking strategies (`Exp. 0–6`, `Table 3`) and four masking probabilities (`10%`, `15%`, `20%`, `30%`; `Exp. 4, 7–9`, `Table 4`), Ankh adopts **1-Gram Span Partial De-masking/Reconstruction** (`Exp. 4`) at a **20% masking probability** (`Exp. 8`) with untied encoder/decoder embeddings (`Exp. 22`). Individual masked residues are corrupted in the encoder input, while consecutive unmasked residues are collapsed into a single merged sentinel token in the decoder target. Notably, the authors found 15% masking excelled on subcellular localization and fold classification while 30% excelled on contact prediction; 20% was selected as the general-purpose compromise.
  > *"The results of the masking strategy experiments promoted utilizing a 1-gram span de-masking strategy (elaborated on in Exp. 4 ). Regarding the masking probability, we used 20% as promoted by Exp. 8."* `[Main, p. 7]`
  > *"Hence, it can be deduced that the top performing version of the six tested versions was the version of Exp. 4, where we reconstruct all the consecutive unmasked tokens as a single merged token."* `[Main, p. 16]`
  > *"Interestingly, it was found that the default value of Exp. 4 (15% masking probability) was outperforming for the sub-cellular localization prediction, fold prediction, as well as some of the secondary structure prediction tasks for datasets such as CB513 and TS115. Nevertheless, it was found that the value of Exp. 9 (30% masking probability) was outperforming for the entirety of the contact prediction tasks"* `[Main, p. 16]`
- **Downstream task implications:** Collapsing unmasked spans on the decoder side focuses cross-attention capacity on reconstructing corrupted residue positions, while the 20% corruption rate balances local motif conservation with long-range 3D contact precision.

### 1.2 Pretraining corpus filtering & sampling
- **Mechanism:** Pre-trained exclusively on **UniRef50** (`45M` proteins, `14B` residues, `16 GB` text; `Table 2`) for **68 full epochs** (`Table 11`). In `Exp. 23` (`Table 10`), the authors explicitly tested training on the larger **UniRef90** corpus (`144M` proteins, `49B` residues) for an equivalent token budget (1 epoch vs. 2 epochs on UniRef50) and found UniRef50 superior across secondary structure, contact prediction, localization, and fold recognition (`68.6%` vs. `67.2%` average) due to lower redundancy.
  > *"For instance, we have reported the superiority of pre-training with UniRef50 over UniRef90, UniRef100, and BFD due to lower redundancy."* `[Main, p. 8]`
  > *"In Table 10, we report the results of the experimentation on the dataset. Nonetheless, the experiment has encouraged the initial direction of proceeding with UniRef50 as representative for efficient high-quality attributes."* `[SI, p. 26]`
  > *"No. of Training Epochs 2 68 68 5"* `[SI, p. 26, Table 11]`
- **Downstream task implications:** Deep multi-epoch convergence (68 epochs) over deduplicated UniRef50 clusters provides strong coverage of canonical natural enzyme families without housekeeping-family over-representation, though rare environmental/metagenomic singletons outside UniProtKB are not seen during pre-training.

### 1.3 Training crop window & positional encoding
- **Mechanism:** Pre-trained with a maximum sequence length of 512 tokens (`Maximum length: 512`, `Table 11`). Uses T5-style shared scalar relative positional embeddings with an embedding offset (maximum relative distance) of 128 tokens and 64 buckets (`embedding dimension of 64`, `Exp. 20`). While any single attention layer clips relative offsets beyond 128 residues to the same bias, stacking 48 encoder layers enables deeper layers to compose relative position information across much longer distances.
  > *"We adopted relative positional embeddings with an embedding offset of 128 and an embedding dimension of 64 as per the results of Exp. 20."* `[Main, p. 7]`
  > *"It is noteworthy that a given layer is insensitive to a relative position beyond 128 tokens, but the following layers can still be sensitive to larger offsets by combining local information from previous layers, enabling the model to accommodate sequence lengths larger than the maximum predetermined length."* `[Main, p. 18]`
  > *"Maximum length 512 512 512 NA"* `[SI, p. 26, Table 11]`
- **Downstream task implications:** Enzymado embeds full-length protein sequences without truncation; single-domain enzymes $\le 512$ residues lie within the pre-training window, while multi-domain enzymes $> 512$ residues rely on 48-layer hierarchical composition of 128-residue relative positional offsets.

### 1.4 Embedding width $D$, pretraining dropout & downstream probe parameterization
- **Mechanism:** Allocates two-thirds of its 72 transformer layers to the encoder (48 encoder layers vs. 24 decoder layers, `Exp. 11`) with Gated-GELU feed-forward networks ($d_{\text{ffn}} = 3,840$, `Exp. 14–15`) and hidden width $D = 1,536$ (16 heads). In `Exp. 13`, widening $D$ from 768 to 1,024 while halving depth to 24 encoder layers degraded median performance (`70.9%` vs. `72.3%`, `Table 6`), motivating the authors to keep the full 48-layer encoder depth in both `ankh_base` ($D = 768$) and `ankh_large` ($D = 1{,}536$).
  > *"For the number of layers, we chose 48 layers for the encoder and 24 layers for the decoder as per the results of Exp. 11. For the activation function, we proceeded with Gated-GELU as per the findings of Exp. 14 and Exp. 15."* `[Main, p. 7]`
  > *"Ankh base has an embedding dimension of 768 as recommended by Exp. 13 while Ankh has an embedding of 1536 as we found that double dimensions are often better performing as verified by Exp. 20."* `[Main, p. 7]`
- **Downstream task implications:** At $D = 1{,}536$, `ankh_large` provides twice the linear readout capacity of `ankh_base` ($D = 768$) and $1.5\times$ that of `prot5_xl_uniref50` ($D = 1{,}024$), benefiting expressive non-linear (`mlp`, `xgboost`) and attention-pooled (`attention_linear`, `attention_mlp`) probes when training data are sufficient.

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | Technical University of Munich (TUM) & Proteinea, Inc. | `interpreted` | `[Main, p. 1]` *"Technical University of Munich (TUM). 2 Proteinea, Inc. 3 Columbia University."* |
| **Publication** | Elnaggar et al. (2023), *arXiv:2301.06568* | `not_reported` | Standalone PDF has no printed DOI string; bibliographic ID is `arXiv:2301.06568`. |
| **Code & Weights** | `https://github.com/agemagician/Ankh` | `reported` | `[Main, p. 19]` *"Both the Ankh and Ankh base models are publicly available for research innovation at our Ankh repository ”https://github.com/agemagician/Ankh”"* (Enzymado loads `ElnaggarLab/ankh-large` via `ankh.load_large_model()`) |
| **License** | `CC BY-NC-SA 4.0` | `reported` | `[Main, p. 19]` *"under an Attribution- NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license."* |
| **Base Checkpoint** | None (trained from random initialization) | `not_reported` | `[SI, p. 26, Table 11]` *"Weight/Bias Initialization random random random random"* |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | Asymmetric T5 encoder–decoder Transformer (48L encoder used) | `reported` | `[Main, p. 9]` *"4.1.2 Pre-trained Model: Encoder-Decoder Transformer"*; *"we utilize the encoder-decoder transformer originally proposed for machine translation"* |
| **Total Parameters** | `<10%` (`Ankh_large`) & `3%` (`Ankh_base`) of 15B training params (`1,150M` encoder in Enzymado registry) | `interpreted` | `[Main, p. 2]` *"with <10% & 3% of the training parameters and 30% & 15% of the embedding dimension, for Ankh and Ankh base respectively (Fig. 1)."* |
| **Layers ($L_{\text{layers}}$)** | 48 encoder layers (+ 24 decoder layers = 72 total) | `reported` | `[SI, p. 26, Table 11]` *"No. of Encoder Layers 36 48 48 NA"*; *"No. of Decoder Layers 36 24 24 NA"* |
| **Hidden Dim ($D$)** | 1,536 | `reported` | `[SI, p. 26, Table 11]` *"Embedding Dimension 768 768 1536 Ankh Emb. Dim."* |
| **FFN Dim ($d_{\text{ffn}}$)** | 3,840 | `reported` | `[SI, p. 26, Table 11]` *"No. of Feed-forward Layers 3072 3072 3840 Emb. Dim./2"* |
| **Attention Heads ($H$)** | 16 | `reported` | `[SI, p. 26, Table 11]` *"No. of Attention Heads 12 12 16 4"* |
| **Activation Function** | `Gated-GELU` | `reported` | `[SI, p. 26, Table 11]` *"Activation Function Gated-GELU Gated-GELU Gated-GELU Gated-GELU"* |
| **Positional Encoding** | Shared scalar relative positional bias (offset 128, 64 buckets) | `interpreted` | `[Main, p. 7]` *"We adopted relative positional embeddings with an embedding offset of 128 and an embedding dimension of 64 as per the results of Exp. 20."* |
| **Max Training Length** | 512 tokens | `reported` | `[SI, p. 26, Table 11]` *"Maximum length 512 512 512 NA"* |
| **Input Modality & Tokens** | 1D amino-acid sequence; single-residue tokens + `[MASK0..N]` sentinels; untied embeddings | `interpreted` | `[Main, p. 9]` *"Sequences in both UniRef50 and UniRef90 are tokenized with a single space between each token"*; `[Main, p. 19]` Exp. 22 untied weights. |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objective** | 1-Gram Span Partial De-masking (`Exp. 4`) at `20%` masking probability (`Exp. 8`) | `reported` | `[Main, p. 7]` *"The results of the masking strategy experiments promoted utilizing a 1-gram span de-masking strategy (elaborated on in Exp. 4 ). Regarding the masking probability, we used 20% as promoted by Exp. 8."* |
| **Pretraining Corpus** | UniRef50 (`45M` proteins, `14B` residues, `16 GB` disk) | `reported` | `[SI, p. 24, Table 2]` *"Number of Proteins [in m] 45 144 216"*; *"Number of Residues [in b] 14 49 88"* |
| **Batch Size & Epochs** | Global batch `1024` (local `16`) $\times$ `68` epochs ($pprox 9.52 \times 10^{11}$ residues) | `reported` | `[SI, p. 26, Table 11]` *"Local Batch Size 16 24 16 1"*; *"Global Batch Size 960 1536 1024 16"*; *"No. of Training Epochs 2 68 68 5"* |
| **Optimizer & Schedule** | `Adafactor`, peak LR `0.002`, `Linear` schedule, `10k` warm-up steps, weight decay `None` | `reported` | `[SI, p. 26, Table 11]` *"Optimizer Adafactor Adafactor Adafactor AdamW"*; *"Learning Rate 0.01 0.004 0.002 0.001"*; *"Warm-Up Steps 10k 10k 10k 1k"* |
| **Dropout & Precision** | Backbone dropout and TPU numeric precision not reported in `Table 11` | `not_reported` | Omitted from `Table 11` (`[SI, p. 26]`). |
| **Hardware & Parallelism** | Google `TPU-v4` Pod (`V4-128`, 128 cores) using Flax/JAX and MegaCore (32 GiB HBM per merged core pair) | `reported` | `[Main, p. 1]` *"trained on Google’s TPU-v4"*; `[SI, p. 26, Table 11]` *"TPUs/GPUs V4-64 V4-128 V4-128 A100"* |
| **Estimated Compute** | Wall-clock time not reported; $\approx 1.07 \times 10^{22}$ FLOPs ($6ND$); `7,515` H100-eq hours | `estimated` | $6 \times N_{\text{total}} \times (68 \times 1.4 \times 10^{10}) \approx 1.07 \times 10^{22}\text{ FLOPs}$ |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 Ankh Family Configuration & Benchmark Summary (`[SI, pp. 26–27, Tables 11 & 13]`)

| Variant | Encoder / Decoder Layers | $D$ (`d_model`) | $d_{\text{ffn}}$ | Heads | Masking Protocol | Epochs | Learning Rate | Global Batch | Hardware | Mean / Median Score (`Table 13`) |
|:---|:---|---:|---:|---:|:---|---:|---:|---:|:---|:---|
| `Baseline (Exp. 0)` | 36 / 36 | 768 | 3,072 | 12 | 15% (full reconstruction) | 2 | `0.01` | 960 | `V4-64` | `64.0%` / `68.9%` (`Table 3`) |
| **`Ankh_base` (`ankh_base`)** | **48 / 24** | **768** | **3,072** | **12** | **20% (1-gram span partial)** | **68** | **`0.004`** | **1,536** | **`V4-128`** | **`70.0%` / `76.0%`** |
| **`Ankh_large` (`ankh_large`)** | **48 / 24** | **1,536** | **3,840** | **16** | **20% (1-gram span partial)** | **68** | **`0.002`** | **1,024** | **`V4-128`** | **`71.43%` / `77.5%`** |

> *"Embedding Dimension 768 768 1536 Ankh Emb. Dim."* `[SI, p. 26, Table 11]`
> *"Avg. 71.43% 70.0% 69.3% 65.3% 66.9% 67.7% 66.6%"* `[SI, p. 27, Table 13]`

### 3.2 Extracted Scientific Claims
- **`[ankh-c1 · scaling_and_efficiency]`** Across seven structure and function benchmarks, `Ankh_large` improved over prior PLM SOTA by 4.8% and `Ankh_base` by 3.4% with $<10\%$ and $3\%$ of the training parameters and $30\%$ and $15\%$ of the embedding dimension of ESM-2 (15B).
  > *"On average, Ankh improved the PLM SOTA performance by 4.8% while Ankh base improved it by 3.4% with <10% & 3% of the training parameters and 30% & 15% of the embedding dimension, for Ankh and Ankh base respectively (Fig. 1)."* `[Main, p. 2]`
- **`[ankh-c2 · efficiency]`** ESM-2 (15B) takes $2.2\times$–$11.7\times$ the feature extraction time of `Ankh_base` and $2.0\times$–$7.1\times$ that of `Ankh_large`, and requires four A100 80 GB GPUs vs. a single A100 40 GB GPU for either Ankh model.
  > *"We can see that ESM-2 (15B) takes minimally 2.2x & 2.0x and maximally 11.7x & 7.1x the feature extraction time for Ankh base and Ankh, respectively."* `[Main, p. 2]`
- **`[ankh-c3 · scaling]`** ESM-2 (15B) did not outperform smaller ESM-2 models across all tasks (e.g., `GB1P` Spearman $0.57$ for ESM-2 15B vs. $0.85$ for `Ankh_base` and $0.84$ for `Ankh_large` in `Table 1`), demonstrating that larger parameter counts are not universally superior on downstream tasks.
  > *"We also observe that ESM-2 (15B) did not outperform the smaller models from the ESM family in all of the tasks in addition to showing inconsistent results across different runs, affirming our hypothesis that bigger models are not necessarily better in all protein modeling tasks and that extensive model sizes bring out its own challenges."* `[Main, p. 3]`
- **`[ankh-c4 · ablation_objective]`** Reconstructing consecutive unmasked tokens as a single merged token (`Exp. 4`, 1-gram span partial de-masking) outperformed full sequence reconstruction and 3-gram span masking across all tasks (`67.9%` vs. `64.0%` baseline in `Table 3`).
  > *"Hence, it can be deduced that the top performing version of the six tested versions was the version of Exp. 4, where we reconstruct all the consecutive unmasked tokens as a single merged token."* `[Main, p. 16]`
- **`[ankh-c5 · ablation_objective]`** Lower masking probability (`15%`) favored subcellular localization and fold recognition, whereas higher masking probability (`30%`) favored contact prediction and CASP12 secondary structure; `20%` (`Exp. 8`) was chosen to balance both regimes.
  > *"Interestingly, it was found that the default value of Exp. 4 (15% masking probability) was outperforming for the sub-cellular localization prediction, fold prediction, as well as some of the secondary structure prediction tasks for datasets such as CB513 and TS115. Nevertheless, it was found that the value of Exp. 9 (30% masking probability) was outperforming for the entirety of the contact prediction tasks in addition to the secondary structure prediction tasks for CASP12 dataset."* `[Main, p. 16]`
- **`[ankh-c6 · ablation_architecture]`** An asymmetric 48-layer encoder / 24-layer decoder (`Exp. 11`) outperformed symmetric 36/36, 54/18, and 24/48 splits across Q8 secondary structure, fold prediction, localization, and overall mean/median.
  > *"It was found that the version with a 48-layer encoder and 24-layer decoder -Exp. 11 - outperformed the version of Exp. 8 in all of the secondary structure prediction tasks (8-states), the fold and sub-cellular localization prediction, and the overall mean and median whilst the other two versions demonstrated a fluctuating performance depending on the task."* `[Main, p. 17]`
- **`[ankh-c7 · ablation_data]`** Pre-training on UniRef50 (`Exp. 20`, mean `68.6%`) outperformed UniRef90 (`Exp. 23`, mean `67.2%`) at matched token budget.
  > *"In Table 10, we report the results of the experimentation on the dataset. Nonetheless, the experiment has encouraged the initial direction of proceeding with UniRef50 as representative for efficient high-quality attributes."* `[SI, p. 26]`
- **`[ankh-c8 · transfer_learning]`** Contextualized last-layer embeddings significantly outperformed attention maps as input to contact prediction across all tested PLMs (`Table 13` vs. `Table 14`).
  > *"Indeed, the results show significant out-performance via embedding-based prediction against attention-based prediction (the full results of attention-based predictions can be found in Table 13."* `[Main, p. 3]`

---

## 4. Paper-Reported Limitations, Biases & Caveats

1. **Parameter overhead of Gated-GELU constrained width-vs.-depth ablations:** Because Gated-GELU introduces substantially more trainable parameters than standard ReLU/GELU, increasing width during fixed-parameter ablations forced compensating reductions in layer depth.
   > *"However, this setting is traced to the nature of the Gated-GELU activation function denoting a significantly larger number of trainable parameters that, in turn, forced us to compensate increases in width by decreases in depth to retain the same total number of trainable parameters and avoid any possible complexity-bias [15]."* `[Main, p. 8]`
2. **Sequential greedy hyperparameter selection without a single composite objective:** Across the 23 ablation experiments, the authors propagated the winning model from each stage to the next using qualitative multi-task judgment rather than a single mathematical selection formula.
   > *"Secondly, our optimization propagated the top-performing model version for each prediction task to the next one. Top performance was not achieved through the version numerically performing top in all tasks, because none such version existed."* `[Main, p. 8]`
3. **Two-epoch budget in pre-design ablations:** All 23 ablation models were evaluated after only 2 epochs of UniRef50 training before scaling the final `Ankh_base` and `Ankh_large` configurations to 68 epochs.
   > *"This entailed promoting generalization across different tasks, especially when the difference amongst results remained numerically negligible, when acknowledging that task-specific customization is of impact, and when all the experimental versions are only trained for two epochs."* `[Main, p. 8]`

---

## 5. Enzymado Embedding & Probing Specification

*(Status: `code` — deterministic pipeline from `SUPPORTING_CONTEXT_MODELS.md`, `scripts/slurm/run_embeddings_cluster.slurm`, and `embedding_scripts/ankh_embedding_script.py`)*

- **Enzymado Key & Checkpoint:** `ankh_large` $\rightarrow$ `ankh.load_large_model()` (Hugging Face `ElnaggarLab/ankh-large`, `1,150M` encoder parameters in registry, $D = 1,536$).
- **Model Loader:** `ankh_embedding_script.py` invokes `ankh.load_large_model()` (which loads `T5EncoderModel` and `AutoTokenizer` from `ElnaggarLab/ankh-large`) and moves the 48-layer encoder to `device` in `.eval()` mode.
- **Non-Standard Amino Acid Sanitization:** Rare/ambiguous residues `U`, `Z`, and `O` are mapped to `X` before tokenization (`seq.replace("U", "X").replace("Z", "X").replace("O", "X")`).
- **Tokenization & Special-Token Slicing:** Sequences are split into character lists `[list(s) for s in seqs]` and tokenized via `tokenizer([list(s) for s in seqs], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")` without truncation. Because Ankh's tokenizer appends `</s>` (EOS) at index `s_len` with no leading BOS/prefix token, Enzymado slices `emb = outputs.last_hidden_state[batch_idx, 0:s_len]` from the final (48th) encoder layer to retain strictly the $L$ amino-acid residue representations $\mathbf{E} \in \mathbb{R}^{L \times 1536}$.
- **Storage Format:** Saved in `bfloat16` (`BF16`) safetensors shards both as per-residue matrices $\mathbf{E} \in \mathbb{R}^{L \times 1536}$ (`SafetensorsShardWriter`) and as mean-pooled sequence vectors $\mathbf{e}_{\text{prot}} = \frac{1}{L}\sum_{i=1}^L \mathbf{E}_i \in \mathbb{R}^{1536}$ (`SafetensorsPooledWriter`).
- **Downstream Probing Heads Evaluated (5 heads across 140 runs per PLM):**
  1. `linear` (`pooling="no_attention"`, `head="linear"`): Ridge / L2-regularized linear or logistic/SGD probe over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1536}$.
  2. `mlp` (`pooling="no_attention"`, `head="mlp"`): 2-layer feed-forward neural network with normalization, activation, and dropout over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1536}$.
  3. `xgboost` (`pooling="no_attention"`, `head="xgboost"`): Gradient-boosted decision trees (`XGBRegressor` / `XGBClassifier`) over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1536}$.
  4. `attention_linear` (`pooling="attention"`, `head="linear"`): Learned per-residue attention pooling (`ResidualAttentionAggregation` for global tasks; `AttentionResidueMixer` for residue tasks) trained jointly with a linear readout head over $\mathbf{E} \in \mathbb{R}^{L \times 1536}$.
  5. `attention_mlp` (`pooling="attention"`, `head="mlp"`): Learned per-residue attention pooling trained jointly with a 2-layer MLP readout head over $\mathbf{E} \in \mathbb{R}^{L \times 1536}$.
- **Task-Layout Conditioning:**
  - **Layout A (`protein_global`):** Uses $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1536}$ (concatenated with standardized reaction conditions such as temperature, pH, and buffer one-hot indicators when present).
  - **Layout B (`protein_ligand_global`):** Concatenates $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1536}$ with 200 normalized RDKit 2D substrate descriptors (`descriptastorus` `RDKit2DNormalized`).
  - **Layout C (`mutation_delta_global`):** Uses the wild-type subtraction vector $\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}} \in \mathbb{R}^{1536}$ for point-mutation thermostability ($\Delta T_m$, $\Delta\Delta G$) tasks.
  - **Layout D (`protein_residue`):** Predicts per-residue binary labels $\{0, 1\}^L$ directly from $\mathbf{E} \in \mathbb{R}^{L \times 1536}$.

---

## 6. Provenance Audit

- **Source PDF:** `ankh_elnaggar2023_arxiv.pdf` (29 pages; `main` = pages 1–23, `si` = Section 7 Supplementary Materials pages 24–29).
- **Quote Verification:** `68 / 68` quotes verified (`0` misses) via normalized substring match against `pdftotext` and `pdftotext -layout` text layers of the cited page.
