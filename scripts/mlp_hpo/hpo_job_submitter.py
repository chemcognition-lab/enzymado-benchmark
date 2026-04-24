"""
Batch submitter for HPO configurations.
Reads a JSON list, submits a batch of SLURM jobs using the existing 
run_training_cluster.slurm file, and tracks progress.
"""

import json
import subprocess
import argparse
from pathlib import Path

INDEX_FILE = Path("last_submitted_index.txt")
DEFAULT_CONFIG_FILE = Path("configs_400_lhs.json")
SLURM_SCRIPT = "scripts/run_training_cluster.slurm"

def load_last_index() -> int:
    """Reads the last submitted index from the tracking file."""
    if not INDEX_FILE.exists():
        return 0
    try:
        with open(INDEX_FILE, "r") as f:
            return int(f.read().strip())
    except ValueError:
        print(f"[WARNING] {INDEX_FILE} is corrupted. Starting from 0.")
        return 0

def save_last_index(index: int):
    """Saves the current index to the tracking file."""
    with open(INDEX_FILE, "w") as f:
        f.write(str(index))

def run_local(config_idx: int, run_id: str, emb_name: str, dry_run: bool = False):
    """Runs a single config locally, bypassing SLURM."""
    
    # Constructing the local command using standard paths from the SLURM script
    local_cmd = [
        "uv", "run", "models/train.py",
        "benchmark_card.json",
        "embeddings_models_output.json",
        "results/predictions",
        "--emb_name", emb_name,
        "--results_card", "results_card.json",
        "--model_prefix", run_id
    ]
    cmd_str = " ".join(local_cmd)

    print(f"\n{'='*60}")
    print(f"LOCAL RUN — Config index {config_idx} (Run ID: {run_id})")
    print(f"{'='*60}")
    print(f"Command:\n  {cmd_str}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution. Command printed above.")
        return

    print("Starting local execution...")
    result = subprocess.run(cmd_str, shell=True)
    
    if result.returncode != 0:
        print(f"\n[WARNING] Process exited with return code {result.returncode}")
    else:
        print("\n[SUCCESS] Local run completed.")

def main():
    parser = argparse.ArgumentParser(description="Batch submit HPO configurations to SLURM.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of jobs to submit.")
    parser.add_argument("--config-file", type=str, default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--emb-name", type=str, default="esm2_650M", help="Target embedding.")

    # Local testing args
    parser.add_argument("--local", action="store_true", help="Run a single config locally instead of via sbatch.")
    parser.add_argument("--local-index", type=int, default=None, help="Index to run locally. Implies --local.")
    parser.add_argument("--dry-run", action="store_true", help="Print local command without executing. Implies --local.")

    args = parser.parse_args()

    # --dry-run and --local-index imply --local
    if args.dry_run or args.local_index is not None:
        args.local = True

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"[ERROR] Config file '{config_path}' not found.")
        return

    # Load configurations
    with open(config_path, "r") as f:
        configs = json.load(f)

    total_configs = len(configs)

    # Run local test
    if args.local:
        index = args.local_index if args.local_index is not None else load_last_index()
        if index >= total_configs:
            print(f"[ERROR] Config index {index} is out of range (0–{total_configs-1}).")
            return
            
        config = configs[index]
        config_idx = config.get("config_index", index)
        run_id = f"mlp_hpo_{config_idx:03d}"
        
        run_local(config_idx, run_id, args.emb_name, dry_run=args.dry_run)
        return
    
    # Determine indices
    start_index = load_last_index()
    if start_index >= total_configs:
        print(f"[INFO] All {total_configs} configurations have already been submitted. Exiting.")
        return

    end_index = min(start_index + args.batch_size, total_configs)
    jobs_to_submit = end_index - start_index

    print(f"--- Starting HPO Batch Submission ---")
    print(f"Configs File : {config_path.name} ({total_configs} total)")
    print(f"Target Emb   : {args.emb_name}")
    print(f"Submitting   : Indices {start_index} to {end_index - 1} ({jobs_to_submit} jobs)")
    print("-" * 40)

    successful_submissions = 0

    # Submission loop
    for i in range(start_index, end_index):
        config = configs[i]
        run_id = f"mlp_hpo_{i:03d}"

        # Construct sbatch command supported by SLURM template
        sbatch_cmd = [
            "sbatch",
            f"--export=ALL,EMB_NAME={args.emb_name},MODEL_PREFIX={run_id}",
            f"--output=slurm_logs/%j/cv_runner-%j.out",
            f"--error=slurm_logs/%j/cv_runner-%j.err",
            SLURM_SCRIPT
        ]

        try:
            # Execute command
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            
            # Parse the job_id
            job_id = output.split()[-1] if "Submitted batch job" in output else "Unknown_ID"
            
            print(f"[{i+1}/{total_configs}] {run_id} -> SLURM Job ID: {job_id}")
            successful_submissions += 1
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit {run_id}: {e.stderr.strip()}")
            break

    # Update tracking file
    new_index = start_index + successful_submissions
    save_last_index(new_index)
    
    print("-" * 40)
    print(f"[INFO] Successfully submitted {successful_submissions} jobs.")
    if new_index >= total_configs:
        print("[INFO] You have submitted all available configurations!")

if __name__ == "__main__":
    main()