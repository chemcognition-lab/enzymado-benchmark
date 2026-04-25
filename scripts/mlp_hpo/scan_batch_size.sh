#!/bin/bash
#SBATCH --account=aip-bensl                 
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --gres=gpu:l40s:1
#SBATCH --tasks-per-node=1                     # Number of processes
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=128G                       # Memory (RAM) per node
#SBATCH --time=00-04:00:00             # Time limit (DD-HH:MM:SS) - here 1 hour

# 1. Load modules
module load python/3.10 cuda/12.2

# 2. Run the Python script
cd /project/6111759/williamc/enzymado-benchmark
uv run python scripts/mlp_hpo/batch_size_scan.py
