---
model_id: esmc-600m
enzymado_key: esmc_600m
model_name: ESM Cambrian (ESMC)
size_variant: 600M
domain: protein
license: Cambria Non-Commercial License (weights) / CC-BY 4.0 (preprint)
created_at: '2026-11-03'
datasets:
  - UniRef
  - MGnify
  - Joint Genome Institute (JGI) Metagenome database
tags:
  - protein
  - transformer
  - protein_language_model
  - enzymado
base_model: EvolutionaryScale/esmc-600m-2024-12
doi: 10.64898/2026.06.03.729735
code: https://github.com/evolutionaryscale/esm
---

# Model Card: ESM Cambrian 600M (`esmc_600m` / `EvolutionaryScale/esmc-600m-2024-12`)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and page citation (`[Main, p. X]` or `[SI, p. Y]` in `esmc_candido2026_biorxiv.pdf`). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from paper and SI), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings, so it can be paired with any Dataset Card for mechanistic modeling analysis without leaking benchmark results.

---

## 0. How to think about this model

**ESM Cambrian 600M** (`esmc_600m`, `EvolutionaryScale/esmc-600m-2024-12`) is a 36-layer, 1,152-dimensional pre-norm bidirectional Transformer encoder (`575.0M` parameters) trained purely on 1D amino-acid sequences via 15% masked language modeling (MLM) across **2.806 billion** sequences from UniRef, MGnify, and the Joint Genome Institute (JGI) Metagenome database clustered at 70% sequence identity (~2.455B clusters). By combining a two-stage training curriculum (1M steps at context length 512 weighted 65% toward metagenomes, followed by 500K steps at context length 2,048 weighted 63% toward curated UniRef) with QK-LayerNorm, bias-free SwiGLU layers, RoPE, and $\mu\text{P}$ hyperparameter transfer, `esmc_600m` achieves unsupervised long-range contact precision (`P@L-LR = 0.589 ± 0.002`) statistically indistinguishable from the $5\times$ larger `esm2_3B` model (`0.587 ± 0.002`) and approaching ESM-2 15B (`0.593 ± 0.002`), while maintaining a moderate per-residue embedding width ($D = 1{,}152$ vs. $D = 2{,}560$ for `esm2_3B`).

- **Best-suited dataset regimes:**
  - Broad microbial, environmental, and metagenome-enriched enzyme families (prokaryotic and extremophilic biocatalysts) well-covered by the 2.37B JGI + MGnify metagenomic clusters seen during Stage 1 pretraining.
  - Structure-sensitive enzyme fitness, remote-homology, and mutational landscapes where 3B-scale emergent tertiary contact precision (`P@L-LR = 0.589`) is beneficial without incurring the $>2.2\times$ probe parameter inflation of $D = 2{,}560$ embeddings.
  - Multi-domain enzymes between 1,024 and 2,048 residues, which fall strictly inside ESMC's 500K-step Stage 2 training window (whereas ESM-2 cropped training sequences at 1,024 tokens).
- **Expected vulnerabilities & blind spots:**
  - **Final-layer logit specialization vs. penultimate-layer functional features:** The authors explicitly report (`[SI, p. 34]`) that ESMC's penultimate transformer layer significantly outperforms the final layer on linear-probe folding stability ($\Delta G$) regression because the final layer is projected directly to amino-acid token logits, and that EC functional clustering peaks in middle-to-late layers (`[Main, p. 4]`). Because the official `esm` SDK (`logits_output.embeddings`) returns the final layer, purely linear probes over `esmc_600m` embeddings may experience token-reconstruction interference that non-linear probes (`mlp`, `attention_mlp`, `xgboost`) can better untangle.
  - **Viral protein mutational landscapes:** On 25 ProteinGym viral DMS assays, the authors report that ESMC models perform worse than or similar to prior baselines (`[SI, p. 110]`).
- **Downstream probe interaction summary:** Enzymado extracts the final post-LayerNorm 36th-layer representation ($D = 1{,}152$). Compared with `esmc_300m` ($D = 960$, 30 layers), `esmc_600m` provides 20% wider and 6-layer deeper representations with $+0.037$ higher long-range contact P@L, while keeping downstream probe parameter counts slightly smaller than `esm2_650M` ($D = 1{,}280$) and less than half of `esm2_3B` ($D = 2{,}560$).

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Trained on single 1D protein sequences with a 15% MLM objective in which selected tokens are replaced directly with `[MASK]` under a per-sequence weighted cross-entropy loss so that every sequence contributes equally to the gradient regardless of length or mask count. Across transformer layers, structural contact precision rises sharply in later layers (peaking in the penultimate layer), whereas enzyme functional organization (EC number kNN clustering and topology-independent linear directions) peaks earlier in the stack, with structure and function aligning along nearly orthogonal axes in latent space.
  > *"At each training step, 15% of input tokens are randomly selected and replaced with a special [MASK] token."* `[SI, p. 28]`
  > *"The training loss is a per-sequence weighted cross-entropy, ensuring that each sequence contributes equally to the gradient regardless of its length or number of masked positions."* `[SI, p. 28]`
  > *"In this setting, function and structure appear along nearly orthogonal axes of variation (principal components) in the latent space, and linear probes fit to predict EC number on a subset of topologies generalize to predict unseen topologies."* `[Main, p. 4]`
  > *"Comparing performance across layers, the second to last embedding from the ESMC transformer stack consistently performed best (Figure S5B), significantly outperforming the final layer (which is projected to final sequence logits)."* `[SI, p. 34]`
- **Downstream task implications:** The emergence of fold-independent EC functional directions and linear $\Delta G$ stability directions makes ESMC representations directly informative for catalytic class and thermostability tasks. However, because `esmc_embedding_script.py` extracts the final layer (`logits_output.embeddings`) rather than the penultimate layer, non-linear probes (`mlp`, `attention_mlp`, `xgboost`) have a mechanistic advantage when separating functional signal from masked-token reconstruction features.

### 1.2 Pretraining corpus filtering & two-stage curriculum
- **Mechanism:** Expands pretraining sequence volume by nearly two orders of magnitude over ESM-2 to 2.806 billion sequences from UniRef release 2023_02 (156M sequences; 83M clusters), MGnify version 2023_02 (621M sequences; 372M clusters), and JGI (2.029B sequences; 2B clusters), all clustered at 70% sequence identity. Stage 1 (1M steps, length 512) weights metagenomic sources heavily (54% JGI, 11% MGnify, 36% UniRef), while Stage 2 (500K steps, length 2,048) shifts sampling to 63% curated UniRef, 31% JGI, and 6% MGnify.
  > *"The ESMC training data includes 156 million sequences from UniRef release 2023 02 (28), 2.029 billion from JGI (downloaded 7/2023) (29), and 621 million from MGnify version 2023 02 (30), expanding the number of training sequences to 2.806 billion."* `[Main, p. 3]`
  > *"This two-stage strategy allows the model to first learn broad sequence statistics from the large and diverse metagenomic corpus at shorter context lengths, before refining its representations on higher-quality curated sequences at longer context lengths."* `[SI, p. 28]`
- **Downstream task implications:** Massive metagenomic Stage 1 exposure prevents representation saturation as model capacity doubles from 300M to 600M, allowing `esmc_600m` to match `esm2_3B`'s structural contact precision while Stage 2 recalibrates the embedding space toward curated full-length UniRef proteins.

### 1.3 Training crop window & positional encoding
- **Mechanism:** Applies Rotary Position Embeddings (RoPE) in every self-attention layer and trains for 1M steps at context length 512 (Stage 1) followed by 500K steps (~2.1T tokens) at an extended context length of 2,048 tokens (Stage 2).
  > *"The attention module uses Rotary Position Embeddings (RoPE) (88) to encode positional information, enabling the model to generalize to varying sequence lengths."* `[SI, p. 28]`
  > *"In Stage 2, training continues for an additional 500K steps with an extended context length of 2048 tokens."* `[SI, p. 28]`
- **Downstream task implications:** By natively training up to 2,048 residues in Stage 2 (twice ESM-2's 1,024-token crop and four times ProstT5's 512-token translation cap), `esmc_600m` embeds long multi-domain enzymes without relying on unseen positional extrapolation.

### 1.4 Embedding width $D$, modernization & probe parameterization
- **Mechanism:** Uses a 36-layer pre-norm Transformer stack ($D = 1{,}152$, $H = 18$ heads of dimension 64, bias-free SwiGLU FFN of width $3{,}072 = \frac{8}{3}D$) with query-key LayerNorm (`QK-LN`) and $\sqrt{n_{\text{layers}}}$ residual scaling.
  > *"Model dimension (dmodel ) 960 1,152 2,560"* `[SI, p. 29, Table S1]`
  > *"ESMC 600M (0.589 ± 0.002) is statistically indistinguishable from ESM2-3B (0.587 ± 0.002) and approaches ESM2-15B (0.593 ± 0.002)."* `[SI, p. 30]`
- **Downstream task implications:** Compresses 3B-grade structural contact representations into $D = 1{,}152$ dimensions—10% narrower than `esm2_650M` ($D = 1{,}280$) and 55% narrower than `esm2_3B` ($D = 2{,}560$)—keeping both mean-pooled and per-residue attention probes well-regularized across moderate-sized enzyme datasets.

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | Biohub / EvolutionaryScale | `reported` | `[Main, p. 1]` *"1 Biohub 2 work performed at Biohub or EvolutionaryScale"* |
| **Publication** | Candido et al. (2026), *bioRxiv* | `reported` | `[Main, p. 1]` *"bioRxiv preprint doi: https://doi.org/10.64898/2026.06.03.729735; this version posted June 4, 2026."* |
| **Code & Weights** | Open release (`https://github.com/evolutionaryscale/esm`, HF `EvolutionaryScale/esmc-600m-2024-12`) | `interpreted` | `[SI, p. 109]` *"(ESMC 600 million and 300 million parameter model weights have already been open to the scientific community.)"* |
| **License** | CC-BY 4.0 (preprint); Cambria Non-Commercial License (600M weights) | `interpreted` | `[Main, p. 1]` *"It is made available under a CC-BY 4.0 International license."* |
| **Base Checkpoint** | None (trained from scratch) | `not_reported` | Trained from random initialization across Stages 1 and 2. |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | Pre-norm bidirectional Transformer encoder | `reported` | `[SI, p. 28]` *"The ESM Cambrian (ESMC) protein language models are based on a Transformer architecture adapted for protein sequences."* |
| **Total Parameters** | `575.0 M` (`575M` in Enzymado registry) | `reported` | `[SI, p. 29, Table S4]` *"600M 36 1152 575.0 M 1,000,000"* |
| **Layers ($L_{\text{layers}}$)** | 36 | `reported` | `[SI, p. 29, Table S1]` *"Layers (nlayers ) 30 36 80"* (`[Main, p. 2]` prose has a typo stating *"16, 24, and 80"*) |
| **Hidden Dim ($D$)** | 1,152 | `reported` | `[SI, p. 29, Table S1]` *"Model dimension (dmodel ) 960 1,152 2,560"* |
| **Attention Heads ($H$)** | 18 (head dim = 64) | `reported` | `[SI, p. 29, Table S1]` *"Attention heads (nheads ) 15 18 40"*; *"Head dimension (dhead ) 64 64 64"* |
| **FFN Dim ($d_{\text{ffn}}$) & Activation** | 3,072 (`SwiGLU`, bias-free); 2-layer `GELU` MLP output head | `reported` | `[SI, p. 29, Table S1]` *"FFN hidden dimension 2,560 3,072 6,912"*; `[SI, p. 28]` *"The feed-forward network uses the SwiGLU activation function. All linear layers throughout the model are bias-free."* |
| **Normalization & Scaling** | Pre-LayerNorm, QK-LayerNorm, $\sqrt{n_{\text{layers}}}$ residual scaling, final LayerNorm | `reported` | `[SI, p. 28]` *"We apply layer normalization to the query and key projections prior to computing attention scores, which stabilizes training at scale."* |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) | `reported` | `[SI, p. 28]` *"The attention module uses Rotary Position Embeddings (RoPE) (88) to encode positional information, enabling the model to generalize to varying sequence lengths."* |
| **Training Context Length** | 512 (Stage 1, 1M steps) $\rightarrow$ 2,048 (Stage 2, 500K steps) | `reported` | `[SI, p. 29, Table S3]` *"Stage 1 context length 512"*; *"Stage 2 context length 2,048"* |
| **Vocabulary & Modalities** | 1D amino-acid sequence; vocabulary size 64 (20 standard AAs + special tokens incl. `[MASK]`, BOS, EOS) | `reported` | `[SI, p. 29, Table S1]` *"Vocabulary size 64 64 64"*; `[SI, p. 28]` *"Input amino acid sequences are tokenized to a one hot embedding, which includes the 20 standard amino acids as well as special tokens."* |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objective** | Masked Language Modeling (15% replaced with `[MASK]`, per-sequence weighted cross-entropy) | `reported` | `[SI, p. 28]` *"All ESMC models are trained with a masked language modeling (MLM) objective (25). At each training step, 15% of input tokens are randomly selected and replaced with a special [MASK] token."* |
| **Pretraining Corpus** | 2.806B sequences across UniRef 2023_02 (156M seqs, 83M clusters), JGI (2.029B seqs, 2B clusters), MGnify 2023_02 (621M seqs, 372M clusters) clustered at 70% ID | `reported` | `[Main, p. 3]` *"The ESMC training data includes 156 million sequences from UniRef release 2023 02 (28), 2.029 billion from JGI (downloaded 7/2023) (29), and 621 million from MGnify version 2023 02 (30), expanding the number of training sequences to 2.806 billion."* |
| **Stage 1 / Stage 2 Mixture** | Stage 1 (1M steps, len 512): 36% UniRef, 11% MGnify, 54% JGI; Stage 2 (500K steps, len 2048): 63% UniRef, 6% MGnify, 31% JGI | `reported` | `[SI, p. 28]` *"The data mixture is shifted to emphasize UniRef sequences, which constitute approximately 63% of the training data, while JGI and MGnify are reduced to 31% and 6%, respectively."* |
| **Batch Size & Steps** | ~4.2M tokens/batch ($2^{22} = 4{,}194{,}304$) $\times$ 1,500,000 total steps ($\approx 6.29 \times 10^{12}$ tokens) | `reported` | `[SI, p. 29]` *"All model sizes use approximately 4.2M tokens per batch across both stages: Stage 1 uses batch size 8,192 at sequence length 512, and Stage 2 uses batch size 2,048 at sequence length 2,048."* |
| **Optimizer & $\mu\text{P}$ Scaling** | AdamW ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$), selective weight decay via $\mu\text{P}$, gradient clip norm `1.0` | `reported` | `[SI, p. 28]` *"We optimize all models using AdamW with $\beta_1 = 0.9, \beta_2 = 0.95$, and $\epsilon = 10^{-8}$."*; *"Learning rates are scaled across model sizes using Maximal Update Parameterization ($\mu$P)."* |
| **LR Schedule** | Warmup-Stable-Decay (WSD): 1,000-step linear warmup, constant peak LR in Stage 1 (1M steps), decay over Stage 2 (500K steps) to $0.1\times$ peak LR | `reported` | `[SI, p. 28]` *"We use a Warmup-Stable-Decay (WSD) learning rate schedule. The learning rate increases linearly over a 1,000-step warmup, remains constant at the peak through Stage 1, and decays over all of Stage 2 (500,000 steps, $\approx$1/3 of the 1,500,000-step total) to a minimum value of 0.1$\times$ the peak learning rate."* |
| **Precision & Hardware** | `bf16-mixed` precision; GPU type, count, and wall-clock time not reported | `reported` | `[SI, p. 29, Table S3]` *"300M/600M: bf16-mixed; 6B: FP8-mixed (TE)"* |
| **Estimated Compute** | $\approx 2.17 \times 10^{22}$ FLOPs ($6ND$; `2.75e+22` with quadratic attention term); `15,239` H100-eq hours | `estimated` | $6 \times (5.75 \times 10^8) \times (6.291 \times 10^{12}) = 2.17 \times 10^{22}\text{ FLOPs}$ |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 ESMC Family Scaling Summary (`[SI, pp. 28–30, Tables S1–S4]`)

*All released ESMC models are trained for 1M Stage 1 steps (context 512, 65% metagenomic) + 500K Stage 2 steps (context 2,048, 63% UniRef) at ~4.2M tokens/batch.*

| Model Variant | Exact Params | Layers ($n_{\text{layers}}$) | Width ($d_{\text{model}}$) | Heads ($n_{\text{heads}}$) | Head Dim | FFN Dim | Precision | Stage 1 Steps @ Len | Stage 2 Steps @ Len | Long-Range Contact P@L-LR |
|:---|---:|---:|---:|---:|---:|---:|:---|:---|:---|---:|
| `ESMC 300M` (`esmc_300m`) | 333.0 M | 30 | 960 | 15 | 64 | 2,560 | `bf16-mixed` | 1M @ 512 | 500K @ 2,048 | 0.552 $\pm$ 0.002 |
| **`ESMC 600M` (`esmc_600m`)** | **575.0 M** | **36** | **1,152** | **18** | **64** | **3,072** | **`bf16-mixed`** | **1M @ 512** | **500K @ 2,048** | **0.589 $\pm$ 0.002** |
| `ESMC 6B` | 6.35 B | 80 | 2,560 | 40 | 64 | 6,912 | `FP8-mixed (TE)` | 1M @ 512 | 500K @ 2,048 | 0.725 $\pm$ 0.002 |

> *"300M 30 960 333.0 M 1,000,000"* `[SI, p. 29, Table S4]`
> *"600M 36 1152 575.0 M 1,000,000"* `[SI, p. 29, Table S4]`
> *"ESMC 300M (P@L-LR = 0.552 ± 0.002) achieves contact prediction performance comparable to ESM2-650M (0.566±0.002) despite roughly half the parameters. ESMC 600M (0.589 ± 0.002) is statistically indistinguishable from ESM2-3B (0.587 ± 0.002) and approaches ESM2-15B (0.593 ± 0.002)."* `[SI, p. 30]`

### 3.2 Extracted Scientific Claims
- **`[esmc-c1 · scaling]`** Unsupervised long-range contact precision scales log-linearly with pretraining compute ($\text{P@L-LR} = 0.115 \log_{10}(\text{FLOPs}) - 1.982$, $R^2 = 0.99$ on scaling sweep, $R^2 = 0.98$ across ESMC 300M, 600M, and 6B).
  > *"ESMC exhibits a scaling law, a log-linear relationship between training compute and the emergence of information about protein structure in the model’s representations."* `[Main, p. 3]`
- **`[esmc-c2 · scaling]`** Expanding the training corpus from ~50M UniRef50 proteins (ESM-2) to 2.806B UniRef + JGI + MGnify sequences eliminates the scaling plateau observed in ESM-2 between 650M and 15B parameters.
  > *"As the parameter count of the released ESMC models increases from 300 million to 6 billion parameters, contact prediction accuracy improves log-linearly (Figure 1C). This differs from ESM2, where diminishing returns from scale were observed between 650 million and 15 billion parameters."* `[Main, p. 3]`
- **`[esmc-c3 · parameter_efficiency]`** ESMC 600M (`P@L-LR = 0.589 ± 0.002`) matches the 5x larger ESM-2 3B model (`0.587 ± 0.002`) and approaches ESM-2 15B (`0.593 ± 0.002`) in unsupervised long-range contact precision.
  > *"ESMC 600M (0.589 ± 0.002) is statistically indistinguishable from ESM2-3B (0.587 ± 0.002) and approaches ESM2-15B (0.593 ± 0.002)."* `[SI, p. 30]`
- **`[esmc-c4 · representation_geometry]`** Structure and function peak at distinct depths in the transformer stack: EC functional kNN accuracy peaks in layers 50–60 of ESMC 6B, whereas long-range contact precision peaks in the penultimate layer.
  > *"Long range contact precision is low in the first 40 layers of the model, and rises sharply in the later layers, peaking in the penultimate layer."* `[Main, p. 4]`
- **`[esmc-c5 · representation_geometry]`** On the structure-controlled EC-CATH benchmark, enzyme function and fold topology align along nearly orthogonal principal components, and linear probes trained to predict EC numbers generalize to held-out CATH topologies.
  > *"In this setting, function and structure appear along nearly orthogonal axes of variation (principal components) in the latent space, and linear probes fit to predict EC number on a subset of topologies generalize to predict unseen topologies."* `[Main, p. 4]`
- **`[esmc-c6 · transfer_learning]`** On Megascale and held-out FireProt folding stability ($\Delta G$ / $\Delta\Delta G$), Ridge regression on penultimate-layer embeddings significantly outperforms final-layer embeddings and transfers zero-shot across datasets (Spearman $\approx 0.5$ on FireProt vs. $\approx 0.2$ for pseudoperplexity).
  > *"Comparing performance across layers, the second to last embedding from the ESMC transformer stack consistently performed best (Figure S5B), significantly outperforming the final layer (which is projected to final sequence logits)."* `[SI, p. 34]`
- **`[esmc-c7 · limitations]`** On 25 viral DMS datasets in ProteinGym, ESMC models perform worse than or similar to existing tools.
  > *"The results show ESMC models perform worse than or similar to existing tools for variant effect prediction in viral proteins (Figure S48)."* `[SI, p. 110]`

---

## 4. Paper-Reported Limitations, Biases & Caveats

1. **Final-layer vs. penultimate-layer probing degradation:** In Appendix A.1.4.4 (`[SI, p. 34]`), the authors demonstrate that the penultimate layer of ESMC consistently outperforms the final layer on linear Ridge stability prediction because the final layer is projected to amino-acid token logits.
   > *"Comparing performance across layers, the second to last embedding from the ESMC transformer stack consistently performed best (Figure S5B), significantly outperforming the final layer (which is projected to final sequence logits)."* `[SI, p. 34]`
2. **Wildtype pseudoperplexity dependence:** Downstream stability classifier accuracy depends strongly on the language model pseudoperplexity of the wildtype sequence.
   > *"we find that the performance of stability classifiers depends on the language model pseudoperplexity (pppl) of the wildtype sequence (Figure S5A)."* `[SI, p. 34]`
3. **Underperformance on viral variant-effect prediction:** Across 25 viral DMS landscapes in ProteinGym, ESMC models do not improve over ESM-2 650M, ESM3 1.4B Open, or EVMutation.
   > *"The results show ESMC models perform worse than or similar to existing tools for variant effect prediction in viral proteins (Figure S48)."* `[SI, p. 110]`
4. **Main-text vs. SI layer-count discrepancy & unreported compute hardware:** Main text p. 2 states that the 300M, 600M, and 6B models have `"16, 24, and 80 transformer layers"`, whereas SI Table S1 and Table S4 (p. 29) and the released checkpoints have `30, 36, and 80` layers. Additionally, GPU hardware type, accelerator count, wall-clock training time, and numeric base learning rate are not reported in the manuscript.
   > *"The 300 million, 600 million, and 6 billion parameter ESMC models have 16, 24, and 80 transformer layers, respectively."* `[Main, p. 2]`

---

## 5. Enzymado Embedding & Probing Specification

*(Status: `code` — deterministic pipeline from `SUPPORTING_CONTEXT_MODELS.md` and `embedding_scripts/esmc_embedding_script.py`)*

- **Enzymado Key & Checkpoint:** `esmc_600m` $\rightarrow$ `EvolutionaryScale/esmc-600m-2024-12` (`575M` registry parameters, $D = 1{,}152$).
- **Model Loader:** `esm.models.esmc.ESMC.from_pretrained("esmc_600m").to(device).eval()`.
- **FASTA Parsing & Tokenization:** Gap characters (`-`) are stripped during FASTA loading. Each sequence is wrapped in `protein = ESMProtein(sequence=seq)`, encoded via `protein_tensor = client.encode(protein)`, and passed through `logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))`.
- **Special-Token Slicing:** ESMC prepends BOS at index `0` and appends EOS at index `-1`; Enzymado slices `emb = logits_output.embeddings[0, 1:-1, :]` from the final (36th) transformer layer so that strictly the $L$ amino-acid residue representations $\mathbf{E} \in \mathbb{R}^{L \times 1152}$ are retained.
- **Storage Format:** Saved in `bfloat16` (`BF16`) safetensors shards both as per-residue matrices $\mathbf{E} \in \mathbb{R}^{L \times 1152}$ (`SafetensorsShardWriter`) and as mean-pooled sequence vectors $\mathbf{e}_{\text{prot}} = \frac{1}{L}\sum_{i=1}^L \mathbf{E}_i \in \mathbb{R}^{1152}$ (`SafetensorsPooledWriter`).
- **Downstream Probing Heads Evaluated (5 heads across 140 runs per PLM):**
  1. `linear` (`pooling="no_attention"`, `head="linear"`): Ridge / L2-regularized linear or logistic/SGD probe over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1152}$.
  2. `mlp` (`pooling="no_attention"`, `head="mlp"`): 2-layer feed-forward neural network with normalization, activation, and dropout over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1152}$.
  3. `xgboost` (`pooling="no_attention"`, `head="xgboost"`): Gradient-boosted decision trees (`XGBRegressor` / `XGBClassifier`) over mean-pooled $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1152}$.
  4. `attention_linear` (`pooling="attention"`, `head="linear"`): Learned per-residue attention pooling (`ResidualAttentionAggregation` for global tasks; `AttentionResidueMixer` for residue tasks) trained jointly with a linear readout head over $\mathbf{E} \in \mathbb{R}^{L \times 1152}$.
  5. `attention_mlp` (`pooling="attention"`, `head="mlp"`): Learned per-residue attention pooling trained jointly with a 2-layer MLP readout head over $\mathbf{E} \in \mathbb{R}^{L \times 1152}$.
- **Task-Layout Conditioning:**
  - **Layout A (`protein_global`):** Uses $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1152}$ (concatenated with standardized reaction conditions such as temperature, pH, and buffer one-hot indicators when present).
  - **Layout B (`protein_ligand_global`):** Concatenates $\mathbf{e}_{\text{prot}} \in \mathbb{R}^{1152}$ with 200 normalized RDKit 2D substrate descriptors (`descriptastorus` `RDKit2DNormalized`).
  - **Layout C (`mutation_delta_global`):** Uses the wild-type subtraction vector $\mathbf{e}_{\text{mut}} - \mathbf{e}_{\text{wt}} \in \mathbb{R}^{1152}$ for point-mutation thermostability ($\Delta T_m$, $\Delta\Delta G$) tasks.
  - **Layout D (`protein_residue`):** Predicts per-residue binary labels $\{0, 1\}^L$ directly from $\mathbf{E} \in \mathbb{R}^{L \times 1152}$.

---

## 6. Provenance Audit

- **Source PDF:** `esmc_candido2026_biorxiv.pdf` (111 pages; main text & references on pages 1–27, Appendix A Supplementary Information on pages 28–111).
- **Quote Verification:** `69 / 69` quotes verified (`0` misses) via normalized substring match against `pdftotext` and `pdftotext -layout` text layers of the cited page.
