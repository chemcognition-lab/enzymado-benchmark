---
model_id: esm2-3b
enzymado_key: esm2_3B
model_name: ESM-2
size_variant: 3B
domain: protein
license: MIT
created_at: '2026-11-03'
datasets:
  - UniRef50
  - UniRef90
tags:
  - protein
  - transformer
  - protein_language_model
  - enzymado
base_model: facebook/esm2_t36_3B_UR50D
doi: 10.1126/science.ade2574
code: https://github.com/facebookresearch/esm
---

# Model Card: ESM-2 3B (`facebook/esm2_t36_3B_UR50D`)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and page citation (`[Main, p. X]` or `[SI, p. Y]`). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from paper and SI), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings, so it can be paired with any Dataset Card for mechanistic modeling analysis without leaking benchmark results.

---

## 0. How to think about this model

**ESM-2 3B** (`facebook/esm2_t36_3B_UR50D`, Enzymado key `esm2_3B`) is a 36-layer, 2,560-dimensional bidirectional Transformer encoder (~2.84B parameters) trained with the exact same 15% masked language modeling (MLM) objective and UniRef50/90 cluster-uniform sampling protocol as `esm2_650M`, and selected by the authors as the frozen language-model trunk of the released ESMFold structure predictor (the largest ESM-2 variant permitting single-GPU inference). Scaling from 650M to 3B lowers UniRef50 validation perplexity from `6.95` to `6.49`, increases unsupervised long-range contact precision (`LR P@L` from `0.52` to `0.54`), and specifically improves structural representation of sequences with low evolutionary depth (few natural homologs / shallow MSAs)—at the cost of doubling the per-residue embedding width to $D = 2{,}560$ with zero pretraining dropout.

- **Best-suited dataset regimes:**
  - Datasets containing divergent, orphan, or low-evolutionary-depth sequences where scaling from 650M to 3B continues to sharpen emergent 3D contact maps and atomic coordinates.
  - Tasks governed by fine-grained evolutionary conservation, mutational deltas ($\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}}$), or global sequence-to-function relationships when paired with regularized mean-pooled probes (Ridge, MLP, or feature-subsampling tree ensembles).
  - Natural enzymatic families well-represented in UniRef90 where lower MLM perplexity (`6.49` vs. `6.95`) yields higher-resolution residue representations.
- **Expected vulnerabilities & blind spots:**
  - Small-$N$ or noisy datasets when paired with high-parameter per-residue attention pooling (`attention_linear`, `attention_mlp`), because $D = 2{,}560$ doubles the downstream parameter count over unregularized (`dropout = 0.0`) token states.
  - *De novo* designed scaffolds (explicitly excluded during UniRef corpus curation) and sequences with high LM perplexity ($> 16$) where evolutionary co-variation is absent.
  - Full-length multidomain proteins exceeding the 1,024-residue pretraining crop window, which depend on RoPE length extrapolation during uncropped Enzymado extraction.
- **Downstream probe interaction summary:** Because ESM-2 removes all hidden and attention dropout during pretraining and scales the hidden dimension to $D = 2{,}560$, downstream probes over mean-pooled embeddings (`linear`, `mlp`, `xgboost`) benefit from higher representational capacity while keeping the sequence-length axis collapsed, whereas learned per-residue attention pooling (`attention_linear`, `attention_mlp` over $L \times 2{,}560$) must optimize a $2\times$ wider unregularized attention projection than on `esm2_650M`.

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Trained with the same 15% masked language modeling (MLM) objective over single sequences as `esm2_650M`, but scaled to 36 layers, 40 attention heads, and $D = 2{,}560$ hidden width (~2.84B parameters) with gradient clipping (`1.0`) under FSDP. Scaling from 650M to 3B lowers validation perplexity from `6.95` to `6.49` and increases unsupervised long-range contact precision (`LR P@L` from `0.52` to `0.54`) and CAMEO structure-projection TM-score (`0.70` to `0.72`), which is why the 3B checkpoint was selected as the frozen sequence trunk of the released ESMFold model (`0.83` mean TM-score on CAMEO, `0.68` on CASP14).
  > *"3B 500K 6.49 0.54 0.81 0.52 0.72"* `[SI, p. 17, Table S1]`
  > *"We use the 3B parameter ESM-2 language model, the largest model that permits inference on a single GPU."* `[SI, p. 5]`
  > *"ESMFold achieves an average TM-score of 0.83 on CAMEO and 0.68 on CASP14."* `[Main, p. 5]`
- **Downstream task implications:** Higher MLM capacity yields finer-grained co-evolutionary and 3D contact representations—particularly beneficial for mutational delta vectors ($\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}}$) and complex sequence–function landscapes when paired with regularized global probes.

### 1.2 Pretraining corpus filtering & sampling
- **Mechanism:** Trained on the identical UniRef50/90 cluster-uniform sampling distribution (~43M UniRef50 clusters, ~138M UniRef90 sequences, ~65M unique sequences seen over 500K updates, excluding artificial/*de novo* sequences). Crucially, the authors show that while high-evolutionary-depth sequences saturate around smaller model scales (150M–650M), low-evolutionary-depth sequences (fewer homologs in UniRef) continue to gain structural accuracy as models scale to 3B and 15B.
  > *"Improvement on sequences with high evolutionary depth thus saturates at lower model scales"* `[Main, p. 3]`
  > *"Each sequence is then replaced with a sequence sampled uniformly from the corresponding UniRef90 cluster."* `[SI, p. 2]`
  > *"De-novo designed proteins are filtered out from the pretraining dataset via two filters."* `[SI, p. 2]`
- **Downstream task implications:** Offers a representational advantage over `esm2_650M` on phylogenetically sparse or divergent enzyme families where shallow evolutionary depth requires larger parameter capacity to internalize weak co-evolutionary signal, while sharing the same blind spot on synthetic *de novo* proteins.

### 1.3 Training crop window & positional encoding
- **Mechanism:** Uses Rotary Position Embedding (RoPE) across all 40 attention heads per layer and trains on random 1,024-token crops with `<cls>` (BOS) and `<eos>` (EOS) boundary markers. RoPE enables relative position awareness and extrapolation beyond the 1,024-token training window.
  > *"In ESM-2, we used Rotary Position Embedding (RoPE) (66) to allow the model to extrapolate beyond the context window it is trained on."* `[SI, p. 4]`
  > *"we cropped long proteins to random 1024 tokens. We used BOS and EOS tokens to signal the beginning and end of a real protein, to allow the model to separate a full-sized protein from a cropped one."* `[SI, p. 4]`
- **Downstream task implications:** Because Enzymado does not truncate sequences during ESM-2 extraction, proteins $\le 1{,}024$ aa operate in-distribution, whereas sequences $> 1{,}024$ aa rely on RoPE extrapolation across 36 layers of 40-head self-attention.

### 1.4 Embedding width $D$, pretraining dropout & downstream probe parameterization
- **Mechanism:** Combines zero pretraining dropout (`dropout = 0.0` in both hidden layers and attention) with a doubled embedding dimension of $D = 2{,}560$ (vs. $D = 1{,}280$ in `esm2_650M` and $D = 1{,}024$ in `prost5`).
  > *"ESM-1b had dropout both in hidden layers and attention which we removed completely to free up more capacity."* `[SI, p. 4]`
  > *"Embedding dim 320 480 640 1280 2560 5120"* `[SI, p. 19, Table S3]`
- **Downstream task implications:** Creates a sharp interaction with downstream probe architecture: mean-pooled probes (`linear` Ridge, `mlp`, and especially feature-subsampling `xgboost`) collapse the sequence axis before fitting and can exploit the richer 2,560-D feature space, whereas learned per-residue attention pooling (`attention_linear`, `attention_mlp` over $L \times 2{,}560$) must fit $2\times$ wider unregularized token representations and faces substantially higher over-parameterization and optimization sensitivity on small or medium sample sizes.

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | FAIR, Meta AI | `reported` | `[Main, p. 1]` *"FAIR, Meta AI, New York, NY, USA."* |
| **Publication** | Lin et al. (2023), *Science* 379, 1123–1130 | `reported` | `[Main, p. 8]` *"10.1126/science.ade2574"* |
| **Code & Weights** | `https://github.com/facebookresearch/esm` | `reported` | `[Main, p. 8]` *"ESM-2 and ESMFold model source code and parameters are available at https://github.com/facebookresearch/esm"* (Enzymado loads Hugging Face `facebook/esm2_t36_3B_UR50D`) |
| **License** | MIT (code & weights); CC-BY (predicted structures) | `reported` | `[Main, p. 8]` *"MIT license for model source code and parameters and CC-BY for predicted structures"* |
| **Base Checkpoint** | None (trained from scratch) | `not_reported` | Trained from random initialization on UniRef50/90. |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | Bidirectional Transformer encoder (BERT-style) | `reported` | `[SI, p. 4]` *"We use a BERT (25) style encoder only transformer architecture (23) with modifications."* |
| **Total Parameters** | 3B (`2,840M` in Enzymado registry; `2.8B` in SI prose) | `reported` | `[SI, p. 19, Table S3]` *"8M 35M 150M 650M 3B 15B"* (`[SI, p. 4]` prose refers to *"the 2.8B and 15B parameter models"*) |
| **Layers ($L_{\text{layers}}$)** | 36 | `reported` | `[SI, p. 19, Table S3]` *"Number of 6 12 30 33 36 48"* |
| **Hidden Dim ($D$)** | 2,560 | `reported` | `[SI, p. 19, Table S3]` *"Embedding dim 320 480 640 1280 2560 5120"* |
| **Attention Heads ($H$)** | 40 (head dim = 64) | `reported` | `[SI, p. 19, Table S3]` *"Attention heads 20 20 20 20 40 40"* |
| **FFN Dim ($d_{\text{ffn}}$) & Activation** | Not reported in paper/SI ($4D = 10{,}240$, GELU in HF config) | `not_reported` | `[SI, p. 4]` states FFN scales with model size citing Table S3, but Table S3 omits $d_{\text{ffn}}$ and activation. |
| **Positional Encoding** | Rotary Position Embedding (RoPE) | `reported` | `[SI, p. 4]` *"In ESM-2, we used Rotary Position Embedding (RoPE) (66) to allow the model to extrapolate beyond the context window it is trained on."* |
| **Training Crop Length** | 1,024 tokens | `reported` | `[SI, p. 4]` *"we cropped long proteins to random 1024 tokens."* |
| **Input Modality & Tokens** | 1D amino-acid sequence; per-residue AA tokens + BOS (`<cls>`), EOS (`<eos>`), `<mask>` | `interpreted` | `[Main, p. 3]` *"Because ESM-2’s training is only on sequences"*; `[SI, p. 4]` *"We used BOS and EOS tokens to signal the beginning and end of a real protein"* |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objective** | Masked Language Modeling (15% mask; 80% `<mask>`, 10% random, 10% kept) | `reported` | `[Main, p. 2]` *"ESM-2 is trained to predict the identity of amino acids that have been randomly masked out of protein sequences"*; *"a randomly generated mask M that includes 15% of positions i in the sequence x"* |
| **Pretraining Corpus** | UniRef50 (Sept 2021, ~43M clusters) $\rightarrow$ uniform UniRef90 member sampling (~138M seqs; ~65M unique seen) | `reported` | `[Main, p. 2]` *"sequences are sampled with even weighting across ∼43 million UniRef50 training clusters from ∼138 million UniRef90 sequences"*; *"the model sees ∼65 million unique sequences"* |
| **Validation Split & Filtering** | 0.5% (~250,000 sequences) held out at 50% identity; *de novo* sequences removed (58,462 proteins) | `reported` | `[SI, p. 2]` *"The training dataset was partitioned by randomly selecting 0.5% (≈ 250,000) sequences to form the validation set."*; *"De-novo designed proteins are filtered out from the pretraining dataset via two filters."* |
| **Batch Size & Updates** | 2M tokens/batch $\times$ 500K updates ($\approx 1.0 \times 10^{12}$ tokens) | `reported` | `[SI, p. 4]` *"All models used 2 million tokens as batch size except the 15B model"*; *"We trained all models for 500K updates except the 15B model which we trained for 270K steps."* |
| **Optimizer & Schedule** | Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$), $L_2$ weight decay `0.01`, warmup `2,000` steps to peak LR `4e-4`, linear decay to `4e-5` over 90% of training | `reported` | `[SI, p. 4]` *"For model training optimization, we used Adam with $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-8}$ and $L_2$ weight decay of 0.01"*; *"warmed up over the first 2,000 steps to a peak value of 4e-4"* |
| **Gradient Clipping & Dropout** | Clip norm `1.0`; Dropout `0.0` (removed from hidden & attention) | `reported` | `[SI, p. 19, Table S3]` *"Clip norm 0 0 0 0 1.0 1.0"*; `[SI, p. 4]` *"ESM-1b had dropout both in hidden layers and attention which we removed completely to free up more capacity."* |
| **Hardware & Distributed** | 512 $\times$ NVIDIA V100 GPUs for 30 days (`FSDP`); numeric precision not reported | `reported` | `[SI, p. 4]` *"We trained each model over 512 NVIDIA V100 GPUs."*; *"The 3B parameter LM took 30 days."*; `[SI, p. 19, Table S3]` *"Distributed DDP DDP DDP DDP FSDP FSDP"* |
| **Estimated Compute** | `368,640` V100-hours; $\approx 1.70 \times 10^{22}$ FLOPs ($6ND$); `11,965` H100-eq hours | `estimated` | $512 \times 30 \times 24 = 368{,}640\text{ V100-h}$; $6 \times (2.84 \times 10^9) \times (1.0 \times 10^{12}) = 1.70 \times 10^{22}\text{ FLOPs}$ |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 ESM-2 Family Scaling Summary (`[SI, pp. 17, 19, Tables S1 & S3]`)

*All models trained for 500K updates on 2M-token batches except 15B (270K updates on 3.2M-token batches).*

| Size Variant | Layers | $D$ (`d_model`) | Heads | Updates | Peak LR | Clip Norm | Parallelism | Val. PPL | Long-Range P@L | CASP14 TM | CAMEO TM |
|:---|---:|---:|---:|---:|:---|:---|:---|---:|---:|---:|---:|
| `8M` | 6 | 320 | 20 | 500K | `4e-4` | `0` | DDP | 10.33 | 0.17 | 0.37 | 0.48 |
| `35M` | 12 | 480 | 20 | 500K | `4e-4` | `0` | DDP | 8.95 | 0.30 | 0.41 | 0.56 |
| `150M` | 30 | 640 | 20 | 500K | `4e-4` | `0` | DDP | 7.75 | 0.44 | 0.49 | 0.65 |
| `650M` (`esm2_650M`) | 33 | 1,280 | 20 | 500K | `4e-4` | `0` | DDP | 6.95 | 0.52 | 0.51 | 0.70 |
| **`3B` (`esm2_3B`)** | **36** | **2,560** | **40** | **500K** | **`4e-4`** | **`1.0`** | **FSDP** | **6.49** | **0.54** | **0.52** | **0.72** |
| `15B` | 48 | 5,120 | 40 | 270K | `1.6e-4` | `1.0` | FSDP | 6.37 | 0.54 | 0.55 | 0.72 |

> *"650M 500K 6.95 0.52 0.79 0.51 0.70"* `[SI, p. 17, Table S1]`
> *"3B 500K 6.49 0.54 0.81 0.52 0.72"* `[SI, p. 17, Table S1]`
> *"15B 270K 6.37 0.54 0.82 0.55 0.72"* `[SI, p. 17, Table S1]`

### 3.2 Extracted Scientific Claims
- **`[esm2-c1 · scaling]`** Validation perplexity improves monotonically with scale from 10.45 (8M) to 6.37 (15B at 270K updates).
  > *"The 8-million-parameter model has a perplexity of 10.45, and the 15 billion model reaches a perplexity of 6.37"* `[Main, p. 2]`
- **`[esm2-c2 · scaling]`** At 500K updates, the 3B model reaches validation perplexity 6.49, unsupervised long-range contact precision P@L of 0.54, and structure-module projection TM-score of 0.72 on CAMEO and 0.52 on CASP14.
  > *"3B 500K 6.49 0.54 0.81 0.52 0.72"* `[SI, p. 17, Table S1]`
- **`[esm2-c3 & esm2-c4 · structure_emergence]`** Atomic-resolution structure can be projected directly from frozen representations, with validation perplexity correlating almost perfectly with structure-projection TM-score across scales ($r = -0.99$ on CASP14, $r = -1.00$ on CAMEO).
  > *"The 15 billion parameter model reaches a TM-score of 0.72 on the CAMEO test set and 0.55 on the CASP14 test set"* `[Main, p. 4]`
  > *"there is a correlation of −0.99 between validation perplexity and CASP14 TM-score and −1.00 between validation perplexity and CAMEO TM-score"* `[Main, p. 4]`
- **`[esm2-c5 · scaling]`** Structural accuracy on sequences with high evolutionary depth saturates at smaller model scales, whereas sequences with low evolutionary depth continue improving as models scale to 3B and 15B.
  > *"Improvement on sequences with high evolutionary depth thus saturates at lower model scales"* `[Main, p. 3]`
- **`[esm2-c6 & esm2-c7 · ablation]`** In 150M/100K-step ablations (Table S5), both Rotary Position Embedding (RoPE) and the updated Sept 2021 UniRef dataset significantly improve unsupervised contact precision (`LR P@L = 0.381` baseline vs. `0.365` without RoPE).
  > *"Unsupervised contact prediction results show that both RoPE and newer data significantly improve the results."* `[SI, p. 4]`
  > *"No RoPE 0.365 0.599 8.62"* `[SI, p. 21, Table S5]`
- **`[esm2-c8 · transfer_learning]`** ESM-2 outperforms the prior ~650M-parameter ESM-1b baseline at comparable parameter count.
  > *"The resulting ESM-2 model family outperforms previously state-of-the-art ESM-1b (a ∼650 million parameter model)"* `[Main, p. 1]`
- **`[esm2-c9 & esm2-c10 · transfer_learning]`** ESMFold uses the frozen 3B ESM-2 model as its sequence trunk (the largest variant fitting on a single GPU for inference) and achieves mean TM-scores of 0.83 on CAMEO and 0.68 on CASP14.
  > *"We use the 3B parameter ESM-2 language model, the largest model that permits inference on a single GPU."* `[SI, p. 5]`
  > *"ESMFold achieves an average TM-score of 0.83 on CAMEO and 0.68 on CASP14."* `[Main, p. 5]`

---

## 4. Paper-Reported Limitations, Biases & Caveats

1. **Degradation on high-perplexity / poorly understood sequences:** Structural projection quality drops sharply when the language model assigns high perplexity to an input sequence (e.g., sequences lacking evolutionary relatives in UniRef).
   > *"The perplexity of the unsuccessful sequence is 16.6, meaning the language model does not understand the input sequence."* `[Main, p. 3]`
2. **Exclusion of *de novo* designed proteins:** Artificial and *de novo* designed proteins were explicitly removed from the pretraining corpus (58,462 sequences via UniProt taxonomy search and Jackhmmer against 154 *de novo* PDB entries), making synthetic scaffolds out-of-distribution.
   > *"De-novo designed proteins are filtered out from the pretraining dataset via two filters."* `[SI, p. 2]`
3. **Unbenchmarked behavior beyond the 1,024-token crop window:** Although RoPE mathematically permits length extrapolation, pretraining cropped all sequences at 1,024 tokens and the paper does not benchmark embedding quality on sequences $> 1{,}024$ residues.
   > *"we cropped long proteins to random 1024 tokens."* `[SI, p. 4]`
4. **Nomenclature & unreported hyperparameter gaps in the SI:** SI Section A.2.4 refers to the 3B model as `"2.8B"`, while Table S3 uses `3B` and the Enzymado registry records `2,840M`. Additionally, language-model training numeric precision (FP16/BF16), feed-forward hidden dimension ($d_{\text{ffn}}$), and activation function are not stated in the paper or SI.
   > *"used sharded data parallelism (FSDP) (68) for the 2.8B and 15B parameter models"* `[SI, p. 4]`

---

## 5. Enzymado Embedding & Probing Specification

*(Status: `code` — deterministic pipeline from `SUPPORTING_CONTEXT_MODELS.md` and `embedding_scripts/esm2_embedding_script.py`)*

- **Enzymado Key & Checkpoint:** `esm2_3B` $\rightarrow$ Hugging Face `facebook/esm2_t36_3B_UR50D` (`2,840M` registry parameters, $D = 2{,}560$).
- **Model Loader:** `transformers.EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D").eval()` and `EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")`.
- **Non-Standard Amino Acid Sanitization:** Rare/ambiguous residues `U`, `Z`, and `O` are mapped to unknown `X` before tokenization (`seq.replace("U", "X").replace("Z", "X").replace("O", "X")`).
- **Tokenization & Special-Token Slicing:** Tokenized via `tokenizer(seqs, add_special_tokens=True, padding="longest", return_tensors="pt")` without explicit truncation. ESM-2 prepends `<cls>` at index `0` and appends `<eos>` at index `s_len + 1`; Enzymado slices `emb = outputs.last_hidden_state[batch_idx, 1 : s_len + 1]` from the final (36th) encoder layer so that strictly the $L$ amino-acid residue representations $\mathbf{E} \in \mathbb{R}^{L \times 2560}$ are retained.
- **Storage Format:** Saved in `bfloat16` (`BF16`) safetensors shards both as per-residue matrices $\mathbf{E} \in \mathbb{R}^{L \times 2560}$ (`SafetensorsShardWriter`) and as mean-pooled sequence vectors $\mathbf{e}_{\text{prot}} = \frac{1}{L}\sum_{i=1}^L \mathbf{E}_i \in \mathbb{R}^{2560}$ (`SafetensorsPooledWriter`).
- **Downstream Probing Heads Evaluated (5 heads across 140 runs per PLM):**
  1. `linear` (`pooling="no_attention"`, `head="linear"`): Ridge / L2-regularized linear or logistic/SGD probe over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{2560}$.
  2. `mlp` (`pooling="no_attention"`, `head="mlp"`): 2-layer feed-forward neural network with normalization, activation, and dropout over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{2560}$.
  3. `xgboost` (`pooling="no_attention"`, `head="xgboost"`): Gradient-boosted decision trees (`XGBRegressor` / `XGBClassifier`) over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{2560}$.
  4. `attention_linear` (`pooling="attention"`, `head="linear"`): Learned per-residue attention pooling (`ResidualAttentionAggregation` for global tasks; `AttentionResidueMixer` for residue tasks) trained jointly with a linear readout head over $\mathbf{E} \in \mathbb{R}^{L \times 2560}$.
  5. `attention_mlp` (`pooling="attention"`, `head="mlp"`): Learned per-residue attention pooling trained jointly with a 2-layer MLP readout head over $\mathbf{E} \in \mathbb{R}^{L \times 2560}$.
- **Task-Layout Conditioning:**
  - **Layout A (`protein_global`):** Uses $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{2560}$ (concatenated with standardized reaction conditions such as temperature, pH, and buffer one-hot indicators when present).
  - **Layout B (`protein_ligand_global`):** Concatenates $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{2560}$ with 200 normalized RDKit 2D substrate descriptors (`descriptastorus` `RDKit2DNormalized`).
  - **Layout C (`mutation_delta_global`):** Uses the wild-type subtraction vector $\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}} \in \mathbb{R}^{2560}$ for point-mutation thermostability ($\Delta T_m$, $\Delta\Delta G$) tasks.
  - **Layout D (`protein_residue`):** Predicts per-residue binary labels $\{0, 1\}^L$ directly from $\mathbf{E} \in \mathbb{R}^{L \times 2560}$.

---

## 6. Provenance Audit

- **Source PDFs:** `model_esm2_1_main.pdf` (9 pages; combined pages 1–9) and `model_esm2_2_suppinfo.pdf` (28 pages; combined pages 10–37 in `model_1_esm2_main_and_si.pdf`).
- **Quote Verification:** `71 / 71` quotes verified (`0` misses) via normalized substring match against `pdftotext` and `pdftotext -layout` text layers of the cited standalone page.
