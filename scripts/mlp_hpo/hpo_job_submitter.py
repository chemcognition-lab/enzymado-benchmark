"""
Batch submitter for HPO configurations.
Reads a JSON list, submits a batch of SLURM jobs using the existing 
run_training_cluster.slurm file, and tracks progress.

run `uv run  hpo_job_submitter.py --account [cluster account]` from scripts/mlp_hpo/.

Embedding options:
ankh3_large,  ankh_base,   esm2_3B,    esm3_open,  esmc_600m,          profluent_e1_600m,  prot5_xl_half_uniref50,
ankh3_xl,     ankh_large,  esm2_650M,  esmc_300m,  profluent_e1_300m,  prost5,             prot5_xl_uniref50
"""

import json
import subprocess
import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_FILE = Path("last_submitted_index.txt")
DEFAULT_CONFIG_FILE = Path("configs_400_lhs.json")
SLURM_SCRIPT = REPO_ROOT / "scripts/run_training_cluster.slurm"
EMBEDDINGS_DATA_ROOT = "/scratch/chemist/enzymado/data"
TASK_NAMES = "km"
CARD_PATH = REPO_ROOT / "benchmark_card.json"

def load_last_index() -> int:
    if not INDEX_FILE.exists():
        return 0
    try:
        with open(INDEX_FILE, "r") as f:
            return int(f.read().strip())
    except ValueError:
        return 0

def save_last_index(index: int):
    with open(INDEX_FILE, "w") as f:
        f.write(str(index))

def run_local(config_idx: int, run_id: str, emb_name: str, config_file_path: Path, dry_run: bool = False):
    """Executes the SLURM bash script locally in your terminal."""
    
    # We pass the variables via the environment, exactly how --export does it
    env = os.environ.copy()
    env["EMB_NAME"] = emb_name
    env["MODEL_PREFIX"] = run_id
    env["HPO_CONFIG_PATH"] = str(config_file_path.resolve())
    env["TASK_NAMES"] = TASK_NAMES
    env["EMBEDDINGS_DATA_ROOT"] = EMBEDDINGS_DATA_ROOT
    env["CARD_PATH"] = CARD_PATH

    print(f"\n{'='*60}")
    print(f"LOCAL BASH RUN — Config index {config_idx} (Run ID: {run_id})")
    print(f"{'='*60}")
    print(f"Simulating SLURM --export: EMB_NAME={emb_name}, MODEL_PREFIX={run_id}")
    print(f"Executing: bash {SLURM_SCRIPT}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return

    print("Starting local execution...\n")
    
    # Run the bash script directly. Output will stream directly to your terminal.
    result = subprocess.run(["bash", SLURM_SCRIPT], env=env)
    
    if result.returncode != 0:
        print(f"\n[WARNING] Process exited with return code {result.returncode}")
    else:
        print("\n[SUCCESS] Local run completed.")

def main():
    parser = argparse.ArgumentParser(description="Batch submit HPO configurations to SLURM.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of jobs to submit.")
    parser.add_argument("--config-file", type=str, default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--emb-name", type=str, default="esmc_300m", help="Target embedding.")
    parser.add_argument("--account", type=str, help="SLURM account to bill")

    # Local testing args
    parser.add_argument("--local", action="store_true", help="Run the slurm script locally in terminal.")
    parser.add_argument("--local-index", type=int, default=None, help="Index to run locally. Implies --local.")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing.")

    args = parser.parse_args()

    if args.dry_run or args.local_index is not None:
        args.local = True

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"[ERROR] Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        configs = json.load(f)

    total_configs = len(configs)

    # Local run for testing
    if args.local:
        index = args.local_index if args.local_index is not None else load_last_index()
        if index >= total_configs:
            print(f"[ERROR] Config index {index} is out of range.")
            return
            
        config = configs[index]
        config_idx = config.get("config_index", index)
        run_id = f"mlp_hpo_{config_idx:03d}"
        
        run_local(config_idx, run_id, args.emb_name, config_path , dry_run=args.dry_run)
        return
    
    # Slurm batch submission
    start_index = load_last_index()
    if start_index >= total_configs:
        print(f"[INFO] All configurations submitted.")
        return

    end_index = min(start_index + args.batch_size, total_configs)
    jobs_to_submit = end_index - start_index

    print(f"--- Starting HPO Batch Submission ---")
    print(f"Configs File : {config_path.name}")
    print(f"Target Emb   : {args.emb_name}")
    print(f"Submitting   : Indices {start_index} to {end_index - 1} ({jobs_to_submit} jobs)")
    print("-" * 40)

    # Create log directory
    log_dir = Path("slurm_logs")
    log_dir.mkdir(exist_ok=True)

    successful_submissions = 0

    for i in range(start_index, end_index):
        config = configs[i]
        run_id = f"mlp_hpo_{i:03d}"

        run_log_dir = REPO_ROOT / "slurm_logs" / f"{run_id}"
        run_log_dir.mkdir(parents=True, exist_ok=True)

        sbatch_cmd = [
            "sbatch",
            f"--account={args.account}",

            f"--chdir={REPO_ROOT}",

            "--job-name=enzymado_hpo",
            "--gres=none",
            "--gpus=h100_1g.10gb:1",
            "--cpus-per-task=1",
            "--mem=8G",
            "--time=00:10:00",

            f"--output={run_log_dir}/%j/job_id_%j.out",
            f"--error={run_log_dir}/%j/job_id_%j.err",

            f"--export=ALL,EMB_NAME={args.emb_name},MODEL_PREFIX={run_id},HPO_CONFIG_PATH={config_path.resolve()},TASK_NAMES={TASK_NAMES},EMBEDDINGS_DATA_ROOT={EMBEDDINGS_DATA_ROOT},CARD_PATH={CARD_PATH}",
            str(SLURM_SCRIPT)
        ]

        if args.dry_run:
            print(f"[{i+1}/{total_configs}] [DRY RUN] {' '.join(sbatch_cmd)}")
            continue

        try:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT)
            output = result.stdout.strip()
            job_id = output.split()[-1] if "Submitted batch job" in output else "Unknown_ID"
            
            print(f"[{i+1}/{total_configs}] {run_id} -> SLURM Job ID: {job_id}")
            successful_submissions += 1
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to submit {run_id}: {e.stderr.strip()}")
            break

    if not args.dry_run:
        new_index = start_index + successful_submissions
        save_last_index(new_index)
        print("-" * 40)
        print(f"[INFO] Successfully submitted {successful_submissions} jobs.")

if __name__ == "__main__":
    main()