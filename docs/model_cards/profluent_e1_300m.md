---
model_id: profluent-e1-300m
enzymado_key: profluent_e1_300m
model_name: Profluent E1
size_variant: 300M
domain: protein
license: Permissive license (free research and commercial use) / CC-BY-NC-ND 4.0 (preprint)
created_at: '2026-11-03'
datasets:
  - Profluent Protein Atlas (PPA-1)
  - UniRef Version 2411
tags:
  - protein
  - retrieval_augmented
  - transformer
  - protein_language_model
  - enzymado
base_model: Profluent-Bio/E1-300m
doi: 10.1101/2025.11.12.688125
code: https://github.com/Profluent-AI/E1
---

# Model Card: Profluent E1 300M (`profluent_e1_300m` / `Profluent-Bio/E1-300m`)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and page citation (`[Main, p. X]` in `profluent_e1_jain2025_biorxiv.pdf`, 8 pp.). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from the 8-page paper, such as per-variant `n_layers`, `d_model`, and `n_heads`), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline and HF checkpoint configuration). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings.

---

## 0. How to think about this model

**Profluent E1 300M** (`profluent_e1_300m`, `Profluent-Bio/E1-300m`) is a retrieval-augmented bidirectional Transformer protein encoder (`274M` exact parameters in the Enzymado registry; $D = 1{,}024$ hidden dimension) pretrained for **4 trillion tokens** on multi-sequence homolog instances drawn from the Profluent Protein Atlas (`PPA-1`) and `UniRef Version 2411`. Its architecture alternates standard **intra-sequence self-attention** layers (where residues attend only within the same protein sequence and RoPE position IDs restart at 1) with **global block-causal attention** layers every third block (where residues attend to preceding unaligned homologous sequences in a concatenated multi-sequence prompt of up to 512 sequences / 32,768 tokens). Crucially, in Enzymado's evaluation pipeline (`embedding_scripts/profluent_embedding_script.py`), `profluent_e1_300m` is run in **single-sequence mode** (`batch_preparer.get_batch_kwargs([seq])` with zero retrieved homologs passed at inference). Even in single-sequence mode, `E1 300M` outperforms both `ESM-2 650M` and `ESMC 300M` on ProteinGym zero-shot DMS fitness (`0.416` vs. `0.414` and `0.406`) and unsupervised CAMEO/CASP15 long-range contact prediction (`0.493 / 0.401` vs. `0.425 / 0.342` for `ESMC 300M`), because conditioning on homologs during pretraining offloads phylogenetic sampling noise to the in-context pathway and forces the core weights to internalize cleaner biophysical and co-evolutionary constraints.

- **Best-suited dataset regimes:**
  - Mutational activity, expression, and stability landscapes ($\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}}$ or local variant ranking) on natural enzymes: in single-sequence mode on ProteinGym, `E1 300M` achieves the highest **Activity** DMS Spearman correlation (`0.438`) among all tested sequence-only models—surpassing `ESM-2 650M` (`0.425`), `ESMC 300M` (`0.423`), and even `E1 600M` (`0.415`; `[Main, p. 5, Table 1]`).
  - Low-MSA-depth and taxonomically diverse protein families: in single-sequence mode on Low-MSA-Depth ProteinGym assays, `E1 300M` reaches Spearman `0.367`, outperforming `ESM-2 650M` (`0.338`), `ESMC 300M` (`0.337`), and `E1 600M` (`0.342`; `[Main, p. 5, Table 2]`).
  - Structure-sensitive catalytic and binding tasks where strong single-sequence long-range contact induction (CAMEO `P@L = 0.493`, CASP15 `P@L = 0.401`; `[Main, p. 6, Table 3]`) and a compact $D = 1{,}024$ embedding width benefit downstream probes.
- **Expected vulnerabilities & blind spots:**
  - **Single-sequence inference without retrieved homologs in Enzymado:** Enzymado calls `batch_preparer.get_batch_kwargs([seq])` with only the query sequence. As a result, every third layer's global block-causal attention operates over a single sequence (reducing to intra-sequence attention) and does not exploit the in-context homolog retrieval pathway that lifts ProteinGym Spearman from `0.416` to `0.475` (`[Main, p. 5, Table 1]`).
  - **Viral proteins in single-sequence mode:** Without retrieved homologs at inference time, zero-shot viral DMS correlation is `0.238` (comparable to `ESMC 300M` at `0.242` and below `ESM-2 3B` at `0.294`), whereas adding retrieved homologs raises viral correlation to `0.490` (`[Main, p. 5, Table 2]`).
  - **No explicit 3D coordinate or functional supervision:** E1 is trained strictly on 1D amino-acid sequences and sequence homologs.
- **Downstream probe interaction summary:** Enzymado extracts `outputs.embeddings` and strips the `<bos>1` and `2<eos>` boundary tokens via `~(batch_preparer.get_boundary_token_mask(batch["input_ids"]))`, yielding $\mathbf{E} \in \mathbb{R}^{L \times 1024}$ in `BF16`. At $D = 1{,}024$ (20% narrower than `profluent_e1_600m` and `esm2_650M` at $D = 1{,}280$), `profluent_e1_300m` keeps linear, MLP, XGBoost, and per-residue attention-pooling (`attention_linear`, `attention_mlp`) parameter counts compact across all 18 tasks.

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Trained with a BERT masked language modeling (MLM) objective (80% `<mask>`, 10% random AA, 10% unchanged; noise fraction linearly decreased from 25% to 15% over the first 250B tokens and fixed at 15% up to 4T tokens) over concatenated multi-sequence homolog instances wrapped in `<bos>1...2<eos>` boundary tokens. Two-thirds of the layers use intra-sequence self-attention (attending only within the same protein sequence with local RoPE restarting at 1), while every third layer uses global block-causal attention (allowing each sequence to attend to preceding homologs in the prompt using global RoPE and shared sequence-ID embeddings), followed by a 1-hidden-layer MLP language modeling head (`[Main, p. 3]`).
  > *"Instead, we adopt an alternating attention architecture [19]: global block-causal attention is used every three layers, while all other layers use intra-sequence attention, where residues attend only to other residues within the same protein sequence."* `[Main, p. 3]`
  > *"During training, we linearly decreased the noise fraction (the fraction of tokens replaced in the input) from 25% to 15% for the first 250 billion tokens; after that, it remained fixed at 15%. We followed the standard BERT masking policy: 80% of selected tokens were replaced with a special mask token, 10% were replaced with a random amino acid, and the remaining 10% were left unchanged."* `[Main, p. 3]`
  > *"In Table 1, we observe that E1 models outperform all ESM-2 and ESMC family models in single-sequence mode at comparable model sizes, indicating that E1 can be used as a drop-in replacement for existing single-sequence encoder models without loss of performance."* `[Main, p. 4]`
- **Downstream task implications:** Conditioning on retrieved homologs during pretraining relieves the model weights from memorizing clade-specific phylogenetic biases. Consequently, even when queried with a single sequence in Enzymado, the learned representations encode sharper long-range 3D contacts (CAMEO `P@L = 0.493` vs. `0.425` for `ESMC 300M`) and stronger mutational fitness signals.

### 1.2 Pretraining corpus filtering & two-phase data curriculum
- **Mechanism:** Trained for 4 trillion tokens on homolog sets from the Profluent Protein Atlas (`PPA-1`) and `UniRef Version 2411`, clustered at 50% and 90% sequence identity. For each 50%-ID cluster representative, Diamond searches against all other 50%-ID representatives to build a homolog set. Training instances sample a homolog set with probability inversely proportional to set size, and replace each 50%-ID representative with a member sequence weighted inversely by its 90%-ID subcluster size. The first 1.5 trillion tokens train exclusively on `PPA-1`, followed by a `60:40` `PPA-1:UniRef` mixture for the remaining 2.5 trillion tokens (`[Main, p. 3]`).
  > *"We used sets of homologous sequences derived from the PPA-1 [11] and UniRef Version 2411 [23] datasets. Both PPA-1 and UniRef are clustered at multiple sequence identity thresholds, including at 50% and 90% identity."* `[Main, p. 3]`
  > *"To construct a training instance, we first randomly sample one of these homolog sets (with probability inversely proportional to the size of the set) and then replace each 50% ID cluster representative with a randomly picked sequence from the associated 50% ID sequence cluster (weighted inversely by the size of its 90% ID subcluster)."* `[Main, p. 3]`
  > *"During training, we exclusively trained on instances from PPA-1 for the first 1.5 trillion tokens. Thereafter, we mixed in instances from UniRef in a 60:40 ratio for the remainder of the training duration."* `[Main, p. 3]`
- **Downstream task implications:** Double inverse-cluster-size reweighting (across both homolog sets and 90%-ID subclusters) prevents abundant protein families from dominating gradients, explaining `E1 300M`'s particularly strong single-sequence performance on low-MSA-depth assays (`0.367` Spearman; `[Main, p. 5, Table 2]`).

### 1.3 Training crop window & dual RoPE positional encoding
- **Mechanism:** Encodes positions via Rotary Position Embedding (RoPE) with two distinct position-ID streams (`within_seq_position_ids` restarting at 1 for each protein in intra-sequence layers; `global_position_ids` tracking absolute position across the concatenated multi-sequence prompt in every 3rd layer), under a curriculum that increases total instance length from 8,192 to 32,768 tokens and sequence count from 2 to 512 (`[Main, p. 3]`).
  > *"We use standard Rotary Position Embedding (RoPE) [20] to encode positional information. For layers using intra-sequence attention, each protein sequence restarts position IDs at one, whereas for globalattention layers, the position ID corresponds to the absolute position of the token within the full concatenated multi-sequence input."* `[Main, p. 3]`
  > *"We employed a curriculum learning strategy where we gradually increase the total length and number of sequences in a multi-sequence instance: from 8192 to 32768 and from 2 to 512 respectively."* `[Main, p. 3]`
- **Downstream task implications:** In Enzymado's single-sequence mode (`[seq]`), `within_seq_position_ids` and `global_position_ids` are identical, and all single-chain enzymes fall well inside the 8,192–32,768 token training horizon.

### 1.4 Embedding width $D = 1{,}024$ & probe parameterization
- **Mechanism:** Although the 8-page paper reports only the nominal model scale (`300M`) and omits per-variant layer/width tables (`not_reported`), the released `Profluent-Bio/E1-300m` checkpoint has `274M` parameters and hidden dimension $D = 1{,}024$ (`code` / Enzymado registry), compared with `641M` parameters and $D = 1{,}280$ for `Profluent-Bio/E1-600m`.
  > *"To test whether the model performance scales with number of parameters, we trained three different sizes of E1 models: 150M, 300M, and 600M parameters."* `[Main, p. 3]`
- **Downstream task implications:** At $D = 1{,}024$, `profluent_e1_300m` provides a compact per-residue and mean-pooled feature representation that reduces downstream probe parameter count by 20% relative to `profluent_e1_600m` ($D = 1{,}280$).

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | Profluent Bio | `reported` | `[Main, p. 1]` *"Profluent Bio, Emeryville, CA, USA"* |
| **Publication** | Jain et al. (2025), *bioRxiv* | `reported` | `[Main, p. 1]` *"bioRxiv preprint doi: https://doi.org/10.1101/2025.11.12.688125; this version posted November 13, 2025."* |
| **Code & Weights** | `https://github.com/Profluent-AI/E1` (HF `Profluent-Bio/E1-300m`) | `reported` | `[Main, p. 7]` *"We make inference code and model weights available at https://github.com/Profluent-AI/E1 under a permissive license."* |
| **License** | Permissive license (free research and commercial use; preprint CC-BY-NC-ND 4.0) | `reported` | `[Main, p. 1]` *"we release three models for free research and commercial use at https://github.com/Profluent-AI/E1."* |
| **Base Checkpoint** | None (trained from scratch for 4T tokens) | `not_reported` | Trained from scratch on PPA-1 and UniRef Version 2411. |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | Retrieval-augmented bidirectional Transformer encoder (alternating intra-sequence attention and every-3rd-layer global block-causal attention) | `reported` | `[Main, p. 3]` *"E1 model family is implemented using a standard Transformer-based architecture [17, 18], augmented with a block causal attention mechanism that enables residues in different homologous sequences to attend to one another... global block-causal attention is used every three layers, while all other layers use intra-sequence attention"* |
| **Total Parameters** | `300M` nominal (`274M` exact in Enzymado registry) | `reported` | `[Main, p. 3]` *"we trained three different sizes of E1 models: 150M, 300M, and 600M parameters."* |
| **Layers ($L_{\text{layers}}$)** | Not reported in paper | `not_reported` | Per-variant layer count is not tabulated in the 8-page manuscript. |
| **Hidden Dim ($D$)** | Not reported in paper (`1,024` in HF checkpoint / Enzymado registry) | `not_reported` | `SUPPORTING_CONTEXT_MODELS.md` and `Profluent-Bio/E1-300m` config specify $D = 1{,}024$ (`code`). |
| **Attention Heads ($H$) & FFN Dim ($d_{\text{ffn}}$)** | Not reported in paper | `not_reported` | Per-variant head count, FFN width, and activation function are not stated in the 8-page manuscript. |
| **Output LM Head** | Single-hidden-layer MLP | `reported` | `[Main, p. 3]` *"A language modeling head (a single hidden layer MLP) is then applied on top of the final-layer token representations to predict the probability of the true amino acid at each selected position."* |
| **Positional Encoding** | Dual Rotary Position Embedding (RoPE): sequence-local (restarting at 1) in intra-sequence layers; global multi-sequence position in block-causal layers | `reported` | `[Main, p. 3]` *"We use standard Rotary Position Embedding (RoPE) [20] to encode positional information. For layers using intra-sequence attention, each protein sequence restarts position IDs at one, whereas for globalattention layers, the position ID corresponds to the absolute position of the token within the full concatenated multi-sequence input."* |
| **Max Context Length** | 8,192 $\rightarrow$ 32,768 tokens (2 $\rightarrow$ 512 sequences per instance) | `reported` | `[Main, p. 3]` *"We employed a curriculum learning strategy where we gradually increase the total length and number of sequences in a multi-sequence instance: from 8192 to 32768 and from 2 to 512 respectively."* |
| **Tokenization & Boundary Tokens** | Each sequence wrapped in `<bos>1...2<eos>` (`<bos>1MLFH2<eos>`) + shared per-sequence ID embedding | `reported` | `[Main, p. 3]` *"The model takes as input a sequence of protein sequences (for example, MLFH,MIIVR,MFHK) with each individual sequence wrapped in special tokens (<bos>1MLFH2<eos><bos>1MIIVR2<eos><bos>1MFHK2<eos>) to mark the start and end of the sequence."* |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objective & Masking** | BERT MLM (80% mask, 10% random AA, 10% unchanged); noise rate annealed from `25%` to `15%` over first 250B tokens, then fixed at `15%` | `reported` | `[Main, p. 3]` *"During training, we linearly decreased the noise fraction (the fraction of tokens replaced in the input) from 25% to 15% for the first 250 billion tokens; after that, it remained fixed at 15%. We followed the standard BERT masking policy: 80% of selected tokens were replaced with a special mask token, 10% were replaced with a random amino acid, and the remaining 10% were left unchanged."* |
| **Pretraining Corpus** | Profluent Protein Atlas (`PPA-1`) and `UniRef Version 2411` (clustered at 50% and 90% ID; Diamond homolog sets) | `reported` | `[Main, p. 3]` *"We used sets of homologous sequences derived from the PPA-1 [11] and UniRef Version 2411 [23] datasets. Both PPA-1 and UniRef are clustered at multiple sequence identity thresholds, including at 50% and 90% identity."* |
| **Data Mixture Curriculum** | First `1.5 trillion tokens` exclusively `PPA-1`; remaining `2.5 trillion tokens` `60:40` `PPA-1:UniRef` | `reported` | `[Main, p. 3]` *"During training, we exclusively trained on instances from PPA-1 for the first 1.5 trillion tokens. Thereafter, we mixed in instances from UniRef in a 60:40 ratio for the remainder of the training duration."* |
| **Batch Size & Total Tokens** | `2^20` (`1,048,576`) tokens/batch for `4 trillion tokens` ($\approx 3.81\text{M}$ steps) | `reported` | `[Main, p. 3]` *"All three E1 models were trained for 4 trillion tokens (batch size = 220 tokens) using a warmup-stable-decay learning rate schedule [21] and Stable AdamW optimizer [22], on clusters of H100/H200 GPUs – for example, E1 600M was trained on a cluster of 64 H100s for 25 days."* |
| **Optimizer & LR Schedule** | `Stable AdamW` optimizer; `warmup-stable-decay` (`WSD`) LR schedule | `reported` | `[Main, p. 3]` *"using a warmup-stable-decay learning rate schedule [21] and Stable AdamW optimizer [22]"* |
| **Hardware** | Clusters of `H100/H200 GPUs` (exact GPU count and wall-clock time not stated for 300M; 600M used 64 H100s for 25 days) | `reported` | `[Main, p. 3]` *"on clusters of H100/H200 GPUs – for example, E1 600M was trained on a cluster of 64 H100s for 25 days."* |
| **Estimated Compute** | $\approx 6.58 \times 10^{21}$ FLOPs ($6ND$ at 274M params; $7.20 \times 10^{21}$ at 300M); `4,619` H100-eq hours | `estimated` | $6 \times (2.74 \times 10^8) \times (4.0 \times 10^{12}) = 6.58 \times 10^{21}\text{ FLOPs}$ |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 Profluent E1 Family Scaling Summary (`[Main, pp. 3, 5, 6, Tables 1–3]`)

*All three E1 variants are trained for 4 trillion tokens (batch size $2^{20} = 1{,}048{,}576$ tokens) with a 1.5T `PPA-1` $\rightarrow$ 2.5T `60:40 PPA-1:UniRef` curriculum and 8,192 $\rightarrow$ 32,768 context length.*

| Model Variant | Nominal Params | Enzymado Registry Params ($D$) | Training Tokens | ProteinGym Spearman (Seq-Only / +Homologs) | ProteinGym NDCG@10 (Seq-Only / +Homologs) | ProteinGym Activity / Stability (Seq-Only) | CAMEO LR P@L (Seq-Only / +Homologs) | CASP15 LR P@L (Seq-Only / +Homologs) |
|:---|---:|:---|:---|:---|:---|:---|:---|:---|
| `E1 150M` | 150M | *(not in Enzymado)* | 4T | 0.401 / 0.473 | 0.744 / 0.785 | 0.426 / 0.532 | 0.466 / 0.510 | 0.387 / 0.406 |
| **`E1 300M` (`profluent_e1_300m`)** | **300M** | **274M ($D = 1{,}024$)** | **4T** | **0.416 / 0.475** | **0.748 / 0.787** | **0.438 / 0.537** | **0.493 / 0.526** | **0.401 / 0.415** |
| `E1 600M` (`profluent_e1_600m`) | 600M | 641M ($D = 1{,}280$) | 4T (64 H100s, 25d) | 0.420 / 0.477 | 0.749 / 0.788 | 0.415 / 0.548 | 0.512 / 0.541 | 0.425 / 0.436 |

> *"E1 300M Sequence Only 0.416 0.438 0.332 0.430 0.346 0.537 0.748"* `[Main, p. 5, Table 1]`
> *"E1 300M Sequence + Homologs 0.475 0.501 0.410 0.468 0.474 0.523 0.787"* `[Main, p. 5, Table 1]`
> *"E1 300M Sequence Only 0.493 0.401"* `[Main, p. 6, Table 3]`
> *"E1 300M Sequence + Homologs 0.526 0.415"* `[Main, p. 6, Table 3]`

### 3.2 Extracted Scientific Claims
- **`[e1-300m-c1 · single_sequence_drop_in_superiority]`** In single-sequence mode (query sequence only), `E1 300M` (`0.416` Spearman, `0.748` NDCG@10) outperforms `ESM-2 650M` (`0.414`), `ESM-2 3B` (`0.406`), `ESMC 300M` (`0.406`), and `ESMC 600M` (`0.405`) across 217 ProteinGym v1.3 substitution DMS assays.
  > *"In Table 1, we observe that E1 models outperform all ESM-2 and ESMC family models in single-sequence mode at comparable model sizes, indicating that E1 can be used as a drop-in replacement for existing single-sequence encoder models without loss of performance."* `[Main, p. 4]`
- **`[e1-300m-c2 · activity_and_low_msa_depth_strength]`** In single-sequence mode, `E1 300M` achieves the highest ProteinGym **Activity** Spearman correlation (`0.438`, beating `E1 600M` at `0.415` and `ESM-2 650M` at `0.425`) and the highest **Low-MSA-Depth** correlation (`0.367`, beating `E1 600M` at `0.342` and `ESM-2 650M` at `0.338`).
  > *"E1 300M Sequence Only 0.416 0.438 0.332 0.430 0.346 0.537 0.748"* `[Main, p. 5, Table 1]`
  > *"E1 300M Sequence Only 0.466 0.513 0.444 0.238 0.367 0.396 0.524"* `[Main, p. 5, Table 2]`
- **`[e1-300m-c3 · unsupervised_contact_prediction]`** Using the Categorical Jacobian method on CAMEO and CASP15 targets, single-sequence `E1 300M` reaches long-range contact `P@L` of `0.493` (CAMEO) and `0.401` (CASP15), outperforming `ESM-2 3B` (`0.434 / 0.339`) and `ESMC 300M` (`0.425 / 0.342`), and improving further to `0.526 / 0.415` with retrieved homologs.
  > *"We observe from Table 3 that E1 models outperform the ESM family of models at all scales when tested in single-sequence mode. Moreover, we see consistent gains in performance when including homologous sequences during inference, indicating that the model is able to leverage in-context evolutionary information to identify putative 3D contacts in a protein."* `[Main, p. 5]`
- **`[e1-300m-c4 · retrieval_augmentation_gains]`** Adding unaligned homologs at inference time lifts `E1 300M` ProteinGym Spearman from `0.416` to `0.475` (surpassing PoET at `0.470` and MSA Pairformer at `0.45`), with the largest jumps on viral proteins (`0.238 -> 0.490`) and low-MSA-depth proteins (`0.367 -> 0.471`).
  > *"When evaluated with homologs at inference time, the E1 models substantially outperform corresponding single-sequence metrics and achieve state of the art performance relative to similar publicly available models, i.e., models that only take homologous sequences as additional context during inference, like MSA Pairformer and PoET∗ ."* `[Main, p. 4]`

---

## 4. Paper-Reported Limitations, Biases & Caveats

1. **Per-variant architectural hyperparameters not tabulated in the 8-page paper:** The manuscript states the three nominal model sizes (`150M, 300M, and 600M parameters`; `[Main, p. 3]`) but omits a table of per-variant `n_layers`, `d_model`, `n_heads`, `d_ffn`, and optimizer learning-rate/beta values (`not_reported` in paper; $D = 1{,}024$ and `274M` exact parameters are documented from the released `Profluent-Bio/E1-300m` checkpoint and `SUPPORTING_CONTEXT_MODELS.md`).
2. **Single-sequence vs. retrieval-augmented gap on viral and low/medium-MSA-depth proteins:** In single-sequence mode (which Enzymado uses), viral DMS correlation is `0.238` vs. `0.490` with retrieval augmentation (`[Main, p. 5, Table 2]`).
   > *"In Table 2, we further observe that switching from single-sequence to retrieval-augmented mode yields consistent improvements for assays with low and medium MSA depth."* `[Main, p. 4]`
3. **Open mechanistic attribution and sequence-only scope up to 600M parameters:** The authors note that further work is needed to disentangle weight-encoded knowledge from in-context homolog retrieval, and that pretraining was limited to sequence data up to 600M parameters (`[Main, pp. 6–7]`).
   > *"In particular, further analysis is needed to disentangle how much E1 is relying on the information encoded in the model weights during pre-training versus that derived from homologous sequences provided at inference time."* `[Main, p. 6]`
   > *"Scaling laws seem to exist as we increase the model parameter count for our zero-shot evaluation tasks. However, we only extended this study to 600M parameters. Also, within the broader context of protein representation learning, we studied only sequence-based models to focus on the effects of retrieval augmentation."* `[Main, p. 7]`

---

## 5. Enzymado Embedding & Probing Specification

*(Status: `code` — deterministic pipeline from `SUPPORTING_CONTEXT_MODELS.md` and `embedding_scripts/profluent_embedding_script.py`)*

- **Enzymado Key & Checkpoint:** `profluent_e1_300m` $\rightarrow$ `Profluent-Bio/E1-300m` (`274M` registry parameters, $D = 1{,}024$).
- **Model Loader:** `E1.modeling.E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-300m", dtype=torch.bfloat16).to(device).eval()`.
- **Single-Sequence Input Mode & Tokenization:** Gap characters (`-`) are stripped during FASTA loading. Enzymado runs E1 in **single-sequence mode** by calling `batch = batch_preparer.get_batch_kwargs([seq], device=str(device))` with a single-element list `[seq]` and **zero retrieved homologs**. The model is invoked under `torch.autocast(device.type, dtype=torch.bfloat16)` with `input_ids`, `within_seq_position_ids`, `global_position_ids`, and `sequence_ids`, returning `outputs.embeddings`.
- **Special-Token Slicing:** E1 wraps each sequence in `<bos>1` and `2<eos>` boundary tokens (`<bos>1MLFH2<eos>`). Enzymado strips these boundary tokens via `residue_selector = ~(batch_preparer.get_boundary_token_mask(batch["input_ids"]))` and `emb = embeddings[0, residue_selector[0]]`, retaining strictly the $L$ amino-acid residue representations $\mathbf{E} \in \mathbb{R}^{L \times 1024}$.
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

- **Source PDF:** `profluent_e1_jain2025_biorxiv.pdf` (8 pages, *bioRxiv* posted November 13, 2025).
- **Quote Verification:** `52 / 52` quotes verified (`0` misses) via normalized substring match against `pdftotext` and `pdftotext -layout` text layers of the cited page.
