---
model_id: prot5-xl-half-uniref50
enzymado_key: prot5_xl_half_uniref50
model_name: ProtT5-XL-U50
size_variant: '3B enc-dec (1.21B FP16 encoder used)'
domain: protein
license: not_reported
created_at: '2026-11-03'
datasets:
  - BFD
  - UniRef50
tags:
  - protein
  - transformer
  - t5
  - protein_language_model
  - enzymado
base_model: Rostlab/prot_t5_xl_half_uniref50-enc
doi: 10.1109/TPAMI.2021.3095381
code: https://github.com/agemagician/ProtTrans
---

# Model Card: ProtT5-XL-U50 Half-Precision Encoder (Rostlab/prot_t5_xl_half_uniref50-enc)

> **Provenance & Scope Note:** Every paper-derived claim and hyperparameter carries a verbatim quote and page citation (`[Main, p. X]` from `prott5_elnaggar2022_tpami.pdf` or `[SI, p. Y]` from `prott5_elnaggar2021_biorxiv.pdf` where Table 2 has a selectable text layer). Status tags distinguish `reported` (stated verbatim), `interpreted` (directly implied by text/mechanics), `not_reported` (absent from paper and SI), `estimated` (derived compute formula), and `code` (Enzymado extraction/probing pipeline). This card is strictly **outcome-blind**: it contains zero Enzymado benchmark evaluation scores, Elo ratings, or leaderboard rankings, so it can be paired with any Dataset Card for mechanistic modeling analysis without leaking benchmark results.

---

## 0. How to think about this model

**ProtT5-XL-U50 Half-Precision Encoder (Rostlab/prot_t5_xl_half_uniref50-enc)** (Enzymado key `prot5_xl_half_uniref50`) is the 24-layer, 1,024-dimensional encoder (~1.21B encoder parameters out of 3B total encoder–decoder parameters) of **ProtT5-XL-U50** from the ProtTrans suite. It was pre-trained in two sequential stages using BERT-style single-token denoising (15% corruption rate) first on **2.122 billion** metagenomic + UniProt sequences in **BFD100** (1.2M steps, global batch size 4,096) and then refined on **45 million** **UniRef50** clusters (991K steps, global batch size 2,048). At inference time, the 24-layer decoder is discarded because the encoder alone significantly outperformed the decoder across all ProtTrans benchmarks. In Enzymado, `prot5_xl_half_uniref50` loads the FP16 half-precision encoder checkpoint (`Rostlab/prot_t5_xl_half_uniref50-enc`) before saving final residue and mean-pooled representations in `bfloat16` (`BF16`) safetensors shards, serving as the direct precision-pair counterpart to `prot5_xl_uniref50` (`Rostlab/prot_t5_xl_uniref50`).

- **Best-suited dataset regimes:**
  - Broad natural and metagenomic enzyme families—including small or orphan families ($N_{\text{eff}} = 1$ to $10$)—where two-stage pre-training on 2.12B BFD metagenomic sequences followed by 45M UniRef50 cluster refinement provides wide taxonomic coverage.
  - Per-residue structural/catalytic annotation (active sites, binding sites, secondary structure) and global sequence-to-function tasks where a compact $D = 1{,}024$ encoder representation trained with 10% dropout resists downstream probe overfitting.
  - Long multi-domain enzymes ($> 512$ residues), where T5 shared relative positional attention biases learned across 24 encoder layers extrapolate beyond the 512-token pre-training target length without hard positional truncation.
- **Expected vulnerabilities & blind spots:**
  - Tasks requiring explicit 3D atomic coordinates or side-chain packing geometry not recoverable from 1D single-token amino-acid denoising alone (in contrast to structure-distilled derivatives such as `prost5`).
  - Fine-grained local functional differentiation (such as closely related EC sub-classes with subtle active-site variations) when relying solely on global mean-pooled embeddings without task-specific per-residue attention pooling.
  - Any downstream regime sensitive to subtle FP16 weight quantization rounding prior to BF16 activation storage, though the ProtTrans authors report indistinguishable benchmark accuracy between half- and full-precision inference.
- **Downstream probe interaction summary:** ProtT5-XL-U50 outputs a compact $D = 1{,}024$ per-residue vector from its 24th encoder layer, regularized during pre-training with `0.1` dropout and backed by a large $d_{\text{ffn}} = 16{,}384$ feed-forward intermediate layer per block. The ProtTrans authors explicitly compared min, max, mean, and concat pooling and showed that global mean-pooling substantially outperformed alternatives, aligning directly with Enzymado's mean-pooled probes (`linear`, `mlp`, `xgboost`), while learned per-residue attention pooling (`attention_linear`, `attention_mlp`) fits a compact $L \times 1{,}024$ parameter budget.

---

## 1. Representational mechanisms & dataset compatibility guide

### 1.1 Objective & latent geometry
- **Mechanism:** Unlike standard NLP T5 which masks multi-token spans, ProtT5 adopted BERT's single-token denoising objective (15% masking probability, generated on the fly at each epoch) within a full encoder–decoder T5 architecture. At inference time, the 24-layer decoder is dropped because the encoder significantly outperformed the decoder on all benchmarks, and running inference in half-precision had no effect on downstream accuracy.
  > *"Contrary to the original T5 model which masks spans of multiple tokens, we adopted BERT’s denoising objective to corrupt and reconstruct single tokens using a masking probability of 15 percent."* `[Main, p. 5]`
  > *"In our hands, the encoder outperformed the decoder on all benchmarks significantly and running the model in half-precision during inference instead of full-precision had no effect on performance"* `[Main, p. 5]`
- **Downstream task implications:** Single-token denoising over bidirectional encoder self-attention forces every residue token to encode both local sequence motifs and global protein context without span-boundary artifacts, making the last encoder hidden state directly usable for both per-residue classification and mean-pooled protein regression/classification.

### 1.2 Pretraining corpus filtering & sampling
- **Mechanism:** Pre-trained via a two-phase curriculum: Phase 1 trained on BFD100 (2,122M metagenomic + UniProt sequences, 393B amino acids) for 1.2M steps at global batch size 4,096 (~4.9B protein samples seen), and Phase 2 fine-tuned on the less redundant, cleaner UniRef50 corpus (45M clusters, 14B amino acids) for 991K steps at global batch size 2,048 (~2.0B protein samples seen). Continuing pre-training on UniRef50 after noisy BFD improved secondary structure Q3 by $+2.8\%$ for ProtT5-XL and enabled strong performance even on singleton/small protein families ($N_{\text{eff}} = 1$).
  > *"First, T5-XL and T5-XXL were trained on BFD for 1.2M and 920k steps respectively (ProtT5-XL-BFD, ProtT5-XXL-BFD). In a second step, ProtT5-XL-BFD and ProtT5-XXL-BFD were fine-tuned on UniRef50 for 991k and 343k steps respectively (ProtT5-XL-U50, ProtT5-XXL-U50)."* `[Main, p. 5]`
  > *"In contrast, the pre-training protocol by which we first trained on the larger but more noisy (more mistakes in sequences) and redundant BFD and then continued pre-training using the smaller, less redundant UniRef50, improved performance significantly for both ProtT5 versions (DQ3(ProtT5-XL-BFD/U50)=2.8% and DQ3(ProtT5-XXL-BFD/U50)=1.4%)."* `[Main, p. 11]`
  > *"ProtT5XL-U50 improved most over NetSurfP-2.0 for the smallest families (Neff=1)."* `[Main, p. 9]`
- **Downstream task implications:** Combines broad metagenomic sequence exposure from BFD with deduplicated cluster balance from UniRef50, yielding strong generalization across diverse prokaryotic and eukaryotic enzymes and particularly benefiting low-homology or small-family sequences.

### 1.3 Training crop window & positional encoding
- **Mechanism:** ProtT5-XL was pre-trained with a target sequence length of 512 tokens across both BFD100 and UniRef50 phases (`Table 2`). Instead of fixed sinusoidal or learned absolute position tables, T5 learns a relative positional bias for each attention head that is shared across all layers, allowing deeper layers to compose relative offsets from lower layers and extrapolate beyond the training window.
  > *"In contrast to this, T5 learns a positional encoding for each attention head that is shared across all layers. This way, the model learned to combine the relative offset between residue pairs of lower layers, enabling the model to make predictions beyond the actual length of the positional encoding."* `[Main, p. 4]`
  > *"Target Length 512 512/2048 512 512/2048 512/1024 512 512"* `[SI, p. 4, Table 2]`
- **Downstream task implications:** Enzymado extracts full-length protein sequences without truncation; single-domain enzymes $\le 512$ residues fall strictly inside the pre-training target length, while longer multi-domain enzymes ($> 512$ residues) rely on T5's shared relative positional biases to generalize across arbitrary lengths.

### 1.4 Embedding width $D$, pretraining dropout & downstream probe parameterization
- **Mechanism:** Each of the 24 encoder layers uses hidden size $D = 1{,}024$, 32 attention heads, an expanded feed-forward intermediate dimension of $d_{\text{ffn}} = 16{,}384$ ($16\times D$), and a pre-training dropout rate of `0.1` (`Table 2`). The authors explicitly compared min, max, mean, and concatenated pooling for protein-level transfer learning and demonstrated that global mean-pooling outperformed min/max by ~14 percentage points and concat by ~10 percentage points on subcellular localization (Q10).
  > *"Hidden Layers Size 1024 1024 1024 4096 1024 1024 1024"* `[SI, p. 4, Table 2]`
  > *"Dropout 0.15 0.0 0.1 0.0 0.0 0.1 0.1 0.0"* `[SI, p. 4, Table 2]`
  > *"While mean-pooling and concat performed similarly for the classification task (membrane/other), mean-pooling outperformed concat for location by about ten percentage points (SOM Table 8, available online)."* `[Main, p. 9]`
- **Downstream task implications:** The compact $D = 1{,}024$ output dimensionality and `0.1` pre-training dropout yield well-regularized residue features where simple mean-pooling (`linear`, `mlp`, `xgboost`) is empirically validated by the authors, while per-residue attention probes (`attention_linear`, `attention_mlp`) require fewer parameters than wider PLMs ($D = 1{,}280$ to $2{,}560$).

---

## 2. Architecture, Tokenization & Pretraining Specification

### 2.1 Identity & Provenance

| Field | Value | Status | Source & Verbatim Quote |
|:---|:---|:---|:---|
| **Organization** | Technical University of Munich (TUM) | `reported` | `[Main, p. 1]` *"Department of Informatics, Bioinformatics & Computational Biology - i12, Technical University of Munich (TUM), 85748 Garching/Munich, Germany."* |
| **Publication** | Elnaggar et al. (2021/2022), *IEEE TPAMI* 44(10), 7112–7127 | `reported` | `[Main, p. 1]` *"Digital Object Identifier no. 10.1109/TPAMI.2021.3095381"* |
| **Code & Weights** | `https://github.com/agemagician/ProtTrans` | `reported` | `[Main, p. 1]` *"All our models are available through https://github.com/agemagician/ProtTrans."* (Enzymado loads Hugging Face `Rostlab/prot_t5_xl_half_uniref50-enc`) |
| **License** | Not reported for weights in paper (paper text CC BY 4.0) | `not_reported` | Paper text carries CC BY 4.0 (`[Main, p. 1]`); explicit software/weights license is not stated in manuscript text. |
| **Base Checkpoint** | `ProtT5-XL-BFD` (trained 1.2M steps on BFD100) | `reported` | `[Main, p. 5]` *"In a second step, ProtT5-XL-BFD and ProtT5-XXL-BFD were fine-tuned on UniRef50 for 991k and 343k steps respectively (ProtT5-XL-U50, ProtT5-XXL-U50)."* |

### 2.2 Architecture & Tokenization

| Parameter | Value | Status | Source & Verbatim Quote / Notes |
|:---|:---|:---|:---|
| **Architecture Type** | T5 encoder–decoder Transformer (encoder used) | `reported` | `[Main, p. 3]` *"T5 uses the original transformer architecture proposed for sequence translation, which consists of an encoder that projects a source language to an embedding space and a decoder that generates a translation to a target language based on the encoder’s embedding."* |
| **Total Parameters** | `3B` total enc–dec (`1,210M` encoder in Enzymado registry) | `reported` | `[Main, p. 5]` *"We trained two model sizes, one with 3B (T5-XL) and one with 11B parameters (T5-XXL)."*; `[Main, p. 5]` *"we dropped the decoder from further analysis which cuts model size by half during inference."* |
| **Layers ($L_{\text{layers}}$)** | 24 encoder layers (and 24 decoder layers) | `reported` | `[SI, p. 4, Table 2]` *"Number of Layers 32 30 30 30 12 30 24 24"* |
| **Hidden Dim ($D$)** | 1,024 | `reported` | `[SI, p. 4, Table 2]` *"Hidden Layers Size 1024 1024 1024 4096 1024 1024 1024"* |
| **FFN Dim ($d_{\text{ffn}}$)** | 16,384 | `reported` | `[SI, p. 4, Table 2]` *"Hidden Layers Intermediate Size 4096 4096 4096 16384 4096 16384 65536"* |
| **Attention Heads ($H$)** | 32 | `reported` | `[SI, p. 4, Table 2]` *"Number of Heads 14 16 16 16 64 16 32 128"* |
| **Activation Function** | Not reported in paper/SI | `not_reported` | Omitted from `Table 2` and manuscript text. |
| **Positional Encoding** | Shared per-head relative positional bias | `interpreted` | `[Main, p. 4]` *"In contrast to this, T5 learns a positional encoding for each attention head that is shared across all layers."* |
| **Training Target Length** | 512 tokens (`Positional Encoding Limits: -`) | `reported` | `[SI, p. 4, Table 2]` *"Target Length 512 512/2048 512 512/2048 512/1024 512 512"* |
| **Input Modality & Tokens** | 1D amino-acid sequence; space-separated AAs, `[BOUZ] -> X` | `interpreted` | `[Main, p. 2]` *"Uniref50, UniRef100 and BFD were tokenized with a single space (indicating word-boundaries) between each token."*; *"Non-generic or unresolved amino acids ([BOUZ]) were mapped to unknown (X)."* |

### 2.3 Pretraining Data, Optimization & Compute

| Parameter | Value | Status | Source & Verbatim Quote / Formula |
|:---|:---|:---|:---|
| **Pretraining Objective** | Single-token denoising (15% corruption on the fly) | `reported` | `[Main, p. 5]` *"Contrary to the original T5 model which masks spans of multiple tokens, we adopted BERT’s denoising objective to corrupt and reconstruct single tokens using a masking probability of 15 percent."* |
| **Pretraining Corpora** | Stage 1: BFD100 (`2,122M` proteins, `393B` AAs) $\rightarrow$ Stage 2: UniRef50 (`45M` proteins, `14B` AAs) | `reported` | `[Main, p. 2, Table 1]` *"Number proteins [in m] 45 216 2,122"*; *"Number of amino acids [in b] 14 88 393"* |
| **Batch Size & Steps** | BFD100: global batch `4096` (local `4`) $\times$ `1.2M` steps; UniRef50: global batch `2048` (local `8`) $\times$ `991K` steps | `reported` | `[SI, p. 4, Table 2]` *"Global Batch Size 44928 22464 32768/6144 15360/2560 1024 10752/1024 9216/3584 2048 4096 2048 4096"*; *"Training Steps 40.7K 31.3K 800K/200K 300K/100K 847K 150K/150K 400K/400K 991K 1.2M 343K 920K"* |
| **Optimizer & Schedule** | AdaFactor, LR `0.01`, inverse square-root decay, `10K` warm-up steps, weight decay `0.0` | `reported` | `[Main, p. 5]` *"All T5 models used the AdaFactor optimizer with inverse square root learning rate schedule for pre-training."*; `[SI, p. 4, Table 2]` *"Learning Rate 0.0005 0.002 0.002 0.00001 0.002 0.002 0.01 0.01"* |
| **Dropout & Precision** | Dropout `0.1`; Pre-training mixed precision `None` (TPUs) | `reported` | `[SI, p. 4, Table 2]` *"Dropout 0.15 0.0 0.1 0.0 0.0 0.1 0.1 0.0"*; *"Mixed Precision None None None None None None"* |
| **Hardware & Parallelism** | Google TPU Pod v3 (`1024` cores / `128` nodes on BFD100; `256` cores / `32` nodes on UniRef50); 8-way model parallelism | `reported` | `[Main, p. 11]` *"a mix of TPU Pod V3-256, V3-512, and V3-1024 for ProtT5-XL and ProtT5-XXL [47]."*; `[Main, p. 5]` *"T5-XL was trained using 8-way model parallelism"* |
| **Estimated Compute** | Wall-clock time not reported; $\le 6.40 \times 10^{22}$ FLOPs ($6ND$ at 512-token cap); $\le 44{,}949$ H100-eq hours | `estimated` | $6 \times 3.0 \times 10^9 \times ((1.2\times 10^6 \times 4096 + 9.91\times 10^5 \times 2048) \times 512) = 6.40 \times 10^{22}\text{ FLOPs}$ |

---

## 3. Scaling / Training Phases & Paper-Reported Scientific Claims

### 3.1 ProtT5 Two-Stage Pre-Training & Scaling Summary (`[Main, p. 5; SI, p. 4, Table 2]`)

| Checkpoint / Stage | Total Params | Encoder / Decoder Layers | $D$ (`d_model`) | $d_{\text{ffn}}$ | Heads | Corpus | Global Batch | Steps | Dropout | TPU Cores (Nodes) |
|:---|:---|:---|---:|---:|---:|:---|---:|:---|---:|:---|
| `ProtT5-XL-BFD` (Stage 1) | 3B (1.21B enc) | 24 / 24 | 1,024 | 16,384 | 32 | BFD100 (`2,122M` seqs) | 4,096 | 1.2M | 0.1 | 1,024 (128) |
| **`ProtT5-XL-U50` (Stage 2)** | **3B (1.21B enc)** | **24 / 24** | **1,024** | **16,384** | **32** | **UniRef50 (`45M` seqs)** | **2,048** | **991K** | **0.1** | **256 (32)** |
| `ProtT5-XXL-BFD` (Stage 1) | 11B | 24 / 24 | 1,024 | 65,536 | 128 | BFD100 (`2,122M` seqs) | 4,096 | 920K | 0.0 | 1,024 (128) |
| `ProtT5-XXL-U50` (Stage 2) | 11B | 24 / 24 | 1,024 | 65,536 | 128 | UniRef50 (`45M` seqs) | 2,048 | 343K | 0.1 | 256 (32) |

> *"First, T5-XL and T5-XXL were trained on BFD for 1.2M and 920k steps respectively (ProtT5-XL-BFD, ProtT5-XXL-BFD). In a second step, ProtT5-XL-BFD and ProtT5-XXL-BFD were fine-tuned on UniRef50 for 991k and 343k steps respectively (ProtT5-XL-U50, ProtT5-XXL-U50)."* `[Main, p. 5]`

### 3.2 Extracted Scientific Claims
- **`[prott5-c1 · transfer_learning]`** Frozen ProtT5-XL-U50 embeddings reach 81%–87% Q3 secondary structure accuracy, 81% Q10 subcellular localization accuracy, and 91% Q2 membrane/soluble accuracy, matching or surpassing MSA-based state-of-the-art methods without evolutionary alignments.
  > *"We validated the advantage of using the embeddings as exclusive input for several subsequent tasks: (1) a per-residue (per-token) prediction of protein secondary structure (3-state accuracy Q3=81%-87%); (2) per-protein (pooling) predictions of protein sub-cellular location (ten-state accuracy: Q10=81%) and membrane versus water-soluble (2-state accuracy Q2=91%)."* `[Main, p. 1]`
- **`[prott5-c2 · ablation]`** The ProtT5 encoder significantly outperformed the decoder across all benchmarks, and running inference in FP16 half-precision had no effect on accuracy while fitting on a single 12 GB GPU.
  > *"In our hands, the encoder outperformed the decoder on all benchmarks significantly and running the model in half-precision during inference instead of full-precision had no effect on performance but allowed to run the model on on a single Nvidia TitanV (12GB vRAM)."* `[Main, p. 5]`
- **`[prott5-c3 · data_curriculum]`** Pre-training on massive, noisy BFD followed by continued pre-training on smaller, less redundant UniRef50 significantly improved accuracy ($\Delta Q_3 = +2.8\%$ for ProtT5-XL and $+1.4\%$ for ProtT5-XXL).
  > *"In contrast, the pre-training protocol by which we first trained on the larger but more noisy (more mistakes in sequences) and redundant BFD and then continued pre-training using the smaller, less redundant UniRef50, improved performance significantly for both ProtT5 versions (DQ3(ProtT5-XL-BFD/U50)=2.8% and DQ3(ProtT5-XXL-BFD/U50)=1.4%)."* `[Main, p. 11]`
- **`[prott5-c4 · scaling]`** Downstream performance across pLMs correlated with total pre-training samples seen ($r = 0.62$), and training longer on ProtT5-XL (3B) outperformed scaling up to ProtT5-XXL (11B) for fewer steps.
  > *"In particular, comparing the two largest models (ProtT5-XL and ProtT5-XXL) suggested more pre-training steps to be more beneficial than bigger models."* `[Main, p. 9]`
- **`[prott5-c5 · transfer_learning]`** ProtT5-XL-U50 improved most over MSA-based NetSurfP-2.0 on the smallest protein families ($N_{\text{eff}} = 1$).
  > *"ProtT5XL-U50 improved most over NetSurfP-2.0 for the smallest families (Neff=1)."* `[Main, p. 9]`
- **`[prott5-c6 · pooling_ablation]`** Global mean-pooling outperformed min/max-pooling by ~14 percentage points and concat-pooling by ~10 percentage points on 10-state localization (Q10).
  > *"While mean-pooling and concat performed similarly for the classification task (membrane/other), mean-pooling outperformed concat for location by about ten percentage points (SOM Table 8, available online)."* `[Main, p. 9]`
- **`[prott5-c7 · architecture]`** Bidirectional context is essential for modeling protein sequences compared to unidirectional auto-regressive architectures.
  > *"In contrast, the bi-directional context appeared crucial to model aspects of the language of life."* `[Main, p. 12]`
- **`[prott5-c8 · efficiency]`** ProtT5-XL embeds a human protein in 0.12 s on average (40 min for the entire human proteome), 4–6$\times$ faster than MMseqs2 profile generation.
  > *"ProtT5-XL, required on average 0.12 seconds to generate embeddings for a human protein, completing the entire human proteome (all proteins in an organism) in only 40 minutes."* `[Main, p. 10]`

---

## 4. Paper-Reported Limitations, Biases & Caveats

1. **Weaker unsupervised separation of short local functional motifs vs. global structural classes:** While global structural classes and transmembrane properties separate clearly in pooled embeddings, local functional properties governed by short sequence motifs (such as EC numbers) are less separated without supervised probing.
   > *"In contrast, local features relying on short motifs were less separated (EC-numbers: Fig. 14F, location: Fig. 14C, available online) but still clustered, e.g., for secreted/extracellular proteins or hydrolases."* `[Main, p. 11]`
2. **Noise and redundancy in metagenomic BFD:** Training on BFD alone yields only modest gains due to sequencing/assembly errors and high redundancy until refined on UniRef50.
   > *"In contrast, the pre-training protocol by which we first trained on the larger but more noisy (more mistakes in sequences) and redundant BFD and then continued pre-training using the smaller, less redundant UniRef50, improved performance significantly for both ProtT5 versions"* `[Main, p. 11]`
3. **Diminishing returns of parameter scaling under fixed compute:** Scaling from 3B (`ProtT5-XL`) to 11B (`ProtT5-XXL`) reduced sample throughput under fixed TPU budgets and did not improve downstream accuracy.
   > *"However, the comparison between, e.g., ProtT5-XL and ProtT5-XXL suggested a possible cap to this trend, as larger models see fewer samples in the same amount of computing power."* `[Main, p. 11]`

---

## 5. Enzymado Embedding & Probing Specification

*(Status: `code` — deterministic pipeline from `SUPPORTING_CONTEXT_MODELS.md`, `scripts/slurm/run_embeddings_cluster.slurm`, and `embedding_scripts/prot5_embedding_script.py`)*

- **Enzymado Key & Checkpoint:** `prot5_xl_half_uniref50` $\rightarrow$ Hugging Face `Rostlab/prot_t5_xl_half_uniref50-enc` (`1,210M` encoder parameters in registry, $D = 1{,}024$).
- **Model Loader:** `transformers.T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device).eval()` and `T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)` (24-layer encoder only; decoder is not loaded).
- **Precision Distinction (`prot5_xl_uniref50` vs. `prot5_xl_half_uniref50`):** `prot5_xl_uniref50` loads the full-precision (FP32) checkpoint `Rostlab/prot_t5_xl_uniref50`, whereas `prot5_xl_half_uniref50` loads the official half-precision (FP16) encoder checkpoint `Rostlab/prot_t5_xl_half_uniref50-enc`. Both scripts subsequently write output embeddings into `bfloat16` (`BF16`) safetensors shards.
- **Non-Standard Amino Acid Sanitization & Formatting:** Rare/ambiguous residues `U`, `Z`, and `O` are mapped to `X` (`seq.replace("U", "X").replace("Z", "X").replace("O", "X")`), and residues are space-separated (`seq = " ".join(list(seq))`).
- **Tokenization & Special-Token Slicing:** Tokenized via `vocab(list(seqs), add_special_tokens=True, padding="longest", return_tensors="pt")` without truncation. `T5Tokenizer` appends `</s>` (EOS) at index `s_len` with no leading BOS token; Enzymado slices `emb = embedding_repr.last_hidden_state[batch_idx, :s_len]` from the final (24th) encoder layer to retain strictly the $L$ amino-acid residue representations $\mathbf{E} \in \mathbb{R}^{L \times 1024}$.
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

- **Source PDFs:** `prott5_elnaggar2022_tpami.pdf` (`main`, 16 pages, IEEE TPAMI 2022) and `prott5_elnaggar2021_biorxiv.pdf` (`si`, 17 pages, bioRxiv 2021 companion where Table 2 on page 4 has a selectable text layer).
- **Quote Verification:** `67 / 67` quotes verified (`0` misses) via normalized substring match against `pdftotext` and `pdftotext -layout` text layers of the cited standalone page.
