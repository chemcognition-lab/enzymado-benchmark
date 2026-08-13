---
name: hpc-cluster
description: Guidelines and configurations for selecting HPC cluster hardware (Trillium, Killarney, Balam), managing storage partitions ($HOME, $PROJECT, $SCRATCH, $SLURM_TMPDIR), setting up virtual environments, preparing Slurm jobs, executing parallel hyperparameter sweeps using Optuna's JournalStorage, and tracking experiments offline.
version: 1.0.0
---

# High-Performance Computing (HPC) Cluster Access & Execution Guide

Use this skill when you need to write code, configure shell scripts, structure datasets, or submit training and hyperparameter search jobs to University of Toronto (UofT) and Digital Research Alliance of Canada (formerly Compute Canada) high-performance computing clusters: **Trillium**, **Killarney**, or **Balam**.

* **Alliance System Status:** [Alliance Status Page](https://status.alliancecan.ca/)
* **SciNet System Status:** [SciNet Status Page](https://docs.scinet.utoronto.ca/)

---

## 1. Cluster Architectural Selection Taxonomy

Refer to this taxonomy to select the correct partition based on the hardware requirements of the workload:

| Cluster / Subcluster | GPUs per Node | System RAM | Interconnect | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Nibi** | 288 x H100 (80 GB) / MIG | 2,048 GB | 400 Gbps InfiniBand | Distributed HPO, MIG fast-queue scheduling |
| **Trillium GPU** | 4 x H100 SXM (80 GB) | 755 GiB | 800 Gbps InfiniBand NDR | Large scale LLM / Foundation Model training |
| **Killarney Performance** | 8 x H100 SXM (80 GB) | 2,048 GB | 400 Gbps InfiniBand | Large scale AI / Foundation Models |
| **Killarney Standard** | 4 x L40S (48 GB) | 512 GB | 100 Gbps InfiniBand HDR100 | Standard deep learning batch training, single-node HPO |
| **Balam** | 4 x A100 (40 GB) | 1,024 GB | InfiniBand Coherent Link | GNNs or materials discovery pipelines |
| **Trillium CPU** | None | 755 GiB | 400 Gbps InfiniBand NDR | Highly parallel CPU-only simulation workloads |

### Slurm GPU Specifiers by Cluster

> [!NOTE]
> GPU request specifiers vary by cluster partition rules and Slurm GRES configurations:

| Cluster | Recommended Slurm Directive | Notes / Constraints |
| :--- | :--- | :--- |
| **Nibi** | `#SBATCH --gpus-per-node=h100:1` or `nvidia_h100_80gb_hbm3_1g.10gb:1` | **Mandatory Model Tag**: Bare `--gpus-per-node=1` fails on Nibi with `"There is no 1 GPU-type"`. Must specify `h100:1` or exact MIG slice string. |
| **Killarney** | `#SBATCH --gpus-per-node=l40s:1` (Standard) or `h100:1` (Perf) | Bare `--gpus-per-node=1` also works (defaults to Standard L40S tier). |
| **Balam** | `#SBATCH --gpus-per-node=1` | SciNet wrapper expects bare count `--gpus-per-node=1` (rejects `--gres=gpu:1`). |
| **Trillium GPU** | `#SBATCH --gpus-per-node=h100:1` | Standard Alliance `--gpus-per-node=h100:1` request. |
| **TamIA** | `#SBATCH --gpus-per-node=4` | Whole-node allocation (4 x A100 per node). |
| **Vulcan** | `#SBATCH --gpus-per-node=l40s:1` or `--gpus-per-node=1` | Standard L40S cluster partition. |

> [!TIP]
> **MIG Scheduling Priority & Cluster Matrix**:
> If your workload fits within smaller GPU memory limits, requesting MIG slices results in significantly shorter queue wait times.
> * **MIG Supported Clusters**: **Nibi**, **Narval**, **Fir**, **Rorqual** (see [Alliance Multi-Instance GPU](https://docs.alliancecan.ca/wiki/Multi-Instance_GPU)).
> * **Non-MIG Clusters**: **Trillium GPU** (does not support MIG; use NVIDIA MPS for GPU sharing), **Killarney**, **Balam**, **TamIA**.
> * **Check MIG on Login Node**: Run `sinfo -o "%G" | grep gpu | sed 's/gpu://g' | sed 's/),/\n/g' | cut -d: -f1 | sort | uniq`.

### Slurm Memory Allocation Rule

> [!TIP]
> On Alliance Slurm clusters (Nibi, Killarney, etc.), always specify memory using explicit Megabytes (`M`), e.g., `#SBATCH --mem=32768M` instead of `#SBATCH --mem=32G`. Slurm converts `G` to binary MiB ($32 \times 1024 = 32768\text{M}$); using explicit `M` prevents Slurm wrapper warnings.

---

## 2. Storage & File System I/O Best Practices

HPC filesystems (GPFS and VAST Data) are optimized for large block I/O. Follow these constraints:

### Storage Directory Rules
- **`$HOME`**: Max 100 GB. **Read-only on Trillium compute nodes.** Never write logs, weights, checkpoints, or pip/conda caches here.
- **`$PROJECT`**: Read-only on Trillium compute nodes. Writable on Balam/Killarney. Use for code repositories and sharing static master datasets.
- **`$SCRATCH`**: Read-write on all clusters. Use this as the main directory for launching jobs, saving checkpoints, and writing logs. Files inactive for >60 days are purged on national systems.
- **`$SLURM_TMPDIR`**: A virtual RAM disk on the node's local memory (up to 70% of physical node RAM, ~528 GB on standard Trillium nodes). Extremely low latency.

### Mitigating Filesystem Contention
1. **Consolidate Small Files**: Do not read or write individual images, text files, or graph caches over GPFS/Lustre. This exhausts file quotas (inodes) and causes severe metadata server latency. Consolidate them into single-file containers such as `.tar` archives, SQLite databases, HDF5, LMDB, or TFRecord formats.
2. **RAM Disk Staging**: Stage datasets to the local node SSD or RAM disk (`$SLURM_TMPDIR`) at the beginning of a job, extract them, and run your model data loader locally.
3. **Dataset Streaming (for large collections)**: If your dataset is too large to fit in `$SLURM_TMPDIR` (e.g. >500 GB), do not stream individual files node-by-node. Instead, organize data into large chunked files (e.g. using WebDataset, MosaicML StreamingDataset, or HDF5 chunking) and stream these chunk-by-chunk to the compute nodes.
4. **Data Transfers**: Use Globus endpoints (e.g., `alliancecan#trillium`) for datasets >10 GB. Use dedicated login/datamover nodes for smaller transfers.

---

## 3. Environment & Package Management

Standard Anaconda/Micromamba installations create tens of thousands of individual files, causing metadata contention. Use these pre-configured paths:

### Path A: Virtual Environments and Pre-Compiled Wheels (SciNet Recommended)
Load system-level modules and install precompiled packages using `--no-index` to link directly to cluster-optimized math libraries:
```bash
module load python/3.13.1 gcc/12.3
virtualenv --no-download --system-site-packages ~/.virtualenvs/my_env
source ~/.virtualenvs/my_env/bin/activate
pip install --no-index torch torchvision
```

### Path B: Scinet-Conda Container
For packages requiring complex dependency resolution:
```bash
source /scinet/conda/etc/profile.d/scinet-conda.sh
scinet-conda create -n my_env python==3.10
scinet-conda activate my_env
scinet-conda install -c conda-forge rdkit
```

### Path C: Micromamba Cache Redirection
To use Micromamba, install the binary in `$HOME/.local/bin` (so it is permanent) and redirect environment and package cache directories to `$SCRATCH` to prevent `$HOME` inode exhaustion:
```bash
# Append to ~/.condarc to redirect cache and envs to $SCRATCH
micromamba config append pkgs_dirs /scratch/username/micromamba_cache
micromamba config append envs_dirs /scratch/username/envs
micromamba config set channel_priority strict

# Initialize & Create
eval "$(micromamba shell hook --shell=bash)"
micromamba create --prefix /scratch/username/envs/my_env python=3.10 numpy
micromamba activate /scratch/username/envs/my_env
```

---

## 4. Hyperparameter Sweeps with Optuna

When running distributed hyperparameter sweeps with Optuna across multiple nodes on a parallel filesystem, **never use SQLite databases directly (`sqlite:///study.db`)**. File-locking overhead on GPFS/Lustre will cause `SQLITE_BUSY` crashes.

### Journal File Storage
Use Optuna's **`JournalStorage`** with an append-only log backend:
```python
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

shared_log_path = "/scratch/username/hpo_study_journal.log"
storage = JournalStorage(JournalFileBackend(shared_log_path))

study = optuna.create_study(
    study_name="distributed_sweep",
    direction="minimize",
    storage=storage,
    load_if_exists=True
)
study.optimize(objective, n_trials=5)
```

Submit sweeps using a **Slurm Array Job** where each worker runs an independent trial task querying the append-only journal file.

---

## 5. Experiment Tracking: Offline Logging

Compute nodes do not have internet access. If logging to Weights & Biases (wandb) or MLflow, configure them to run offline:

### Weights & Biases
Add the following environment variables to your Slurm script:
```bash
export WANDB_MODE=offline
export WANDB_DIR=/scratch/username/wandb_logs
```
After the job completes, upload the logged metrics to the cloud by running the following command **on a login node** (which has internet access):
```bash
wandb sync /scratch/username/wandb_logs/offline-run*
```

### Local MLflow
Configure MLflow to log directly to a local tracking SQLite database file on `$SCRATCH`:
```python
import mlflow
mlflow.set_tracking_uri("sqlite:////scratch/username/mlruns.db")
mlflow.set_experiment("experiment_name")
```

---

## 6. Cluster-Specific Execution Constraints

For specific compiler versions, partition rules, GPU resource allocations, and read-only directory workarounds, load and refer to the appropriate machine-specific skill:
* **Balam Cluster:** Use the `balam-cluster` skill.
* **Killarney Cluster:** Use the `killarney-cluster` skill.
* **Trillium Cluster:** Use the `trillium-cluster` skill.

---

## 7. Containerized Environments (Apptainer)

When you need highly customized system libraries or prepackaged environments (e.g., RAPIDS) that are difficult to resolve or build in standard virtual environments, use **Apptainer** (formerly Singularity). **Docker is not supported** on Alliance clusters due to root privilege requirements.
* **GPU Driver Binding:** When running an Apptainer container that requires GPU acceleration, you must pass the `--nv` flag to bind the host system's GPU driver and libraries into the container:
  ```bash
  apptainer run --nv my_container.sif python train.py
  ```

---

## 8. GPU Monitoring & Visualization (TensorBoard & nvtop)

* **TensorBoard:** TensorBoard is recommended for tracking metrics and profiling code. **Never run TensorBoard on login nodes.** Always launch it inside a batch job on a compute node or via the cluster's interactive JupyterLab interface.
* **GPU Utilization Monitoring:** During interactive debug sessions, use these commands on compute nodes to monitor real-time GPU compute and memory consumption:
  * `nvidia-smi` (Standard query utility)
  * `nvtop` (Interactive graphical monitor displaying GPU utilization and memory history)
