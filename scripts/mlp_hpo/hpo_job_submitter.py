"""Submit the 400 Appendix B configurations as a SLURM array.

Each array task calls run_hpo.py with its original config index. Paths are
arguments, the account defaults to aip-bensl, and --dry-run has no side effects.
Use --start-index and --n-configs to resubmit an explicit subset.

Example:
    python scripts/mlp_hpo/hpo_job_submitter.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the cluster and local command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file", type=Path, default=SCRIPT_DIR / "configs_400_lhs.json"
    )
    parser.add_argument(
        "--embeddings-json",
        type=Path,
        default=REPO_ROOT / "embeddings_models_output_mean_pooled.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "mlp_hpo"
    )
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--account", default="aip-bensl")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-configs", type=int, default=400)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--gpus", default="l40s:1", help="GPU type:count for --gres.")
    parser.add_argument(
        "--exclude", default="", help="Nodes to avoid, e.g. kn051,kn092."
    )
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", default="8G")
    parser.add_argument("--time", default="00:10:00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    """Build an argument vector without invoking a shell or creating files."""
    configs = json.loads(args.config_file.read_text(encoding="utf-8"))
    if not isinstance(configs, list) or not configs:
        raise ValueError("Expected a non-empty list of configurations.")
    if any(config.get("config_index") != i for i, config in enumerate(configs)):
        raise ValueError("Configuration indices must match their list positions.")
    training_command = [
        # absolute(), not resolve(): a venv's python is a symlink to the base
        # interpreter, and resolving it would drop the venv's packages.
        str(args.python.absolute()),
        "-u",
        str(SCRIPT_DIR / "run_hpo.py"),
        "--config-file",
        str(args.config_file.resolve()),
        "--embeddings-json",
        str(args.embeddings_json.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--seed",
        str(args.seed),
    ]
    if args.local_index is not None:
        if not 0 <= args.local_index < len(configs):
            raise ValueError("--local-index is outside the configuration list.")
        return [*training_command, "--config-index", str(args.local_index)]
    stop = args.start_index + args.n_configs
    if not 0 <= args.start_index < stop <= len(configs):
        raise ValueError("Requested array range is outside the configuration list.")
    if args.max_concurrent <= 0 or args.cpus_per_task <= 0:
        raise ValueError("Concurrency and CPUs per task must be positive.")
    logs = args.output_dir.resolve() / "slurm_logs"
    return [
        "sbatch",
        "--parsable",
        f"--account={args.account}",
        f"--chdir={REPO_ROOT}",
        "--job-name=enzymado_mlp_hpo",
        f"--array={args.start_index}-{stop - 1}%{args.max_concurrent}",
        f"--gres=gpu:{args.gpus}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--mem={args.mem}",
        f"--time={args.time}",
        *([f"--exclude={args.exclude}"] if args.exclude else []),
        f"--output={logs}/%A_%a.out",
        f"--error={logs}/%A_%a.err",
        str(SCRIPT_DIR / "run_hpo.slurm"),
        *training_command,
    ]


def main() -> None:
    """Print, submit or run the selected configuration range."""
    args = build_parser().parse_args()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.local_index is None:
        (args.output_dir / "slurm_logs").mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
