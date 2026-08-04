---
name: killarney-cluster
description: Guidelines, configurations, module loading instructions, storage policies, and Slurm templates for running jobs on the Killarney Standard (L40S) and Performance (H100) tiers.
version: 1.0.0
---

# SciNet Killarney Cluster Execution & Guide

Use this skill when developing scripts, configuring environments, structuring file storage, or preparing Slurm batch submission pipelines for the **Killarney** AI cluster.

* **Documentation Source:** [Alliance wiki - Killarney](https://docs.alliancecan.ca/wiki/Killarney/en)
* **System Status:** [Alliance Status Page](https://status.alliancecan.ca/)

---

## 1. Architectural Selection Taxonomy

The Killarney cluster consists of two compute tiers. Select the appropriate tier using GPU specifiers:

| Tier | GPUs per Node | System RAM | CPU per Node | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Tier** | 4 x NVIDIA L40S (48 GB) | 512 GB | 2 x Intel Xeon 6338 | Standard deep learning batch training, single-node HPO sweeps. |
| **Performance Tier** | 8 x NVIDIA H100 SXM (80 GB) | 2,048 GB | 2 x Intel Xeon 6442Y | Large scale AI / foundation models, distributed multi-node sweeps. |

---

## 2. Cluster SSH Access & Security

* **Login Node Hostname:** `killarney.alliancecan.ca`
* **Authentication:** SSH keys only.
* **Geographic Blocking:** Killarney restricts login access based on the Government of Canada's National Cyber Threat Assessment. You must connect from a Canadian institutional network or utilize the **Vector Institute VPN** (or UofT campus VPN) if connecting from off-campus.

---

## 3. Storage Directory Policies

* **`$HOME` (Max 100 GB):** Writable, but has tight space and inode (file count) quotas. Never output logs, checkpoints, or pip/conda caches here.
* **`$PROJECT`:** Read-write on compute nodes. Useful for code repositories, shared datasets, and configuration files.
* **`$SCRATCH`:** Read-write on compute nodes. Recommended directory for running jobs, writing output results, and managing large conda/micromamba environments.
* **`$SLURM_TMPDIR`:** Node-local SSD space. Always stage dataset `.tar` files here at the beginning of a job to reduce filesystem latency.

---

## 4. Module Stack & Environment Setup

Killarney utilizes the standard Digital Research Alliance of Canada (DRAC) software stack. Load standard environments, compilers, and CUDA versions as follows:

```bash
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2  # Match PyTorch/Triton CUDA 12 runtimes
```

### Cache Redirection
Add the following exports to your Slurm script to keep pip, PyTorch compile, and matplotlib cache files out of `$HOME`:

```bash
export XDG_CACHE_HOME=$SCRATCH/.cache
export XDG_CONFIG_HOME=$SCRATCH/.config
export PYTHONUSERBASE=$SCRATCH/.local
export TRITON_CACHE_DIR=$SCRATCH/.triton
export MPLCONFIGDIR=$SCRATCH/.config/matplotlib
```

---

## 5. Slurm Submission Rules

* **GPU Allocation Specifiers:** Killarney supports tier selection via model tags:
  * For Standard Tier (L40S 48GB): `#SBATCH --gpus-per-node=l40s:1` (bare `#SBATCH --gpus-per-node=1` also defaults to L40S)
  * For Performance Tier (H100 SXM 80GB): `#SBATCH --gpus-per-node=h100:1`
* **CPU and Memory Requirements:** Unlike Balam, you must explicitly request host resources on Killarney to fit your requirements:
  * `#SBATCH --cpus-per-task=8`
  * `#SBATCH --mem=32G`

---

## 6. Dataset Staging Template

Staging your datasets to local compute node memory (`$SLURM_TMPDIR`) significantly reduces cluster file system contention:

```bash
# Inside your Slurm script:
DATA_SRC=$PROJECT/shared/datasets/my_dataset.tar
DATA_DST=$SLURM_TMPDIR/dataset

echo "=== Staging dataset to local SSD ==="
mkdir -p $DATA_DST
tar -xf $DATA_SRC -C $DATA_DST

# Point your Python data loaders to the local directory:
python train.py --data_dir $DATA_DST
```
