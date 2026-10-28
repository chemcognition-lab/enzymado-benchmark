#!/usr/bin/env python
"""Compute behind the benchmark: run time, memory and hardware of every run.

Reads the published runs (seed 42) and the two extra seeds. A run's time is the span
from its manifest, written when training starts, to its training log, closed when
training ends, so it leaves out job setup and the later bootstrap. Peak memory and
the hardware come from SLURM accounting of the job named in each manifest; a job can
train several runs, so its memory is a peak over them. Runs from the overflow
cluster (Nibi) are timed but have no accounting here.

Example:
    python scripts/paper/compute_requirements.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from enzymado import paths_and_constants

_P = paths_and_constants.Paths()
SCALAR = {"kcat", "km", "ph_opt", "dtm", "ddg", "binding", "solubility"}
CLUSTERS = {"6111759": "Killarney", "6111712": "Nibi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_P.results_dir / "paper")
    return parser.parse_args()


def task_group(scope: str, task: str) -> str:
    match task:
        case _ if scope == "real_case":
            return "real case"
        case "ec_number":
            return "EC number"
        case "active_sites" | "binding_site":
            return "residue"
        case _ if task in SCALAR:
            return "scalar property"
        case _:
            raise ValueError(f"Unknown task {task}.")


def load_runs() -> pd.DataFrame:
    trees = {42: _P.results_dir / "predictions"}
    trees |= {
        s: _P.results_dir / "noise_seeds" / f"seed_{s}" / "predictions"
        for s in (43, 44)
    }
    rows = []
    for seed, tree in trees.items():
        for manifest in tree.glob("*/*/*/*/*/*/run_manifest.json"):
            run = manifest.parent
            scope, pool, split, head, task, emb = run.relative_to(tree).parts
            env = json.loads(manifest.read_text())["environment"]
            device = json.loads((run / "hparams.json").read_text())["params"].get(
                "device"
            )
            project = re.search(r"/project/(\d+)/", env["python_executable"]).group(1)
            rows.append(
                dict(
                    seed=seed, scope=scope, pool=pool, split=split, head=head,
                    task=task, emb=emb, group=task_group(scope, task),
                    job=str(env["slurm_job_id"]), cluster=CLUSTERS[project],
                    cpu=env["platform"].split("x86_64-")[-1].split("-with")[0],
                    device=device or "cuda",
                    minutes=((run / "training_log.csv").stat().st_mtime
                             - manifest.stat().st_mtime) / 60,
                )
            )  # fmt: skip
    return pd.DataFrame(rows)


def to_gib(value: str) -> float:
    """SLURM memory such as 4643796K or 124G, in GiB."""
    if not value:
        return float("nan")
    scale = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1.0, "T": 2**10}
    return float(value[:-1]) * scale[value[-1]]


def accounting(jobs: list[str]) -> pd.DataFrame:
    """Elapsed time, requested and peak memory, GPU and node of each job."""
    fields = "JobIDRaw,ElapsedRaw,MaxRSS,ReqMem,AllocTRES,NodeList"
    lines = []
    for start in range(0, len(jobs), 200):
        batch = ",".join(jobs[start : start + 200])
        out = subprocess.run(
            ["sacct", "-j", batch, "-o", fields, "-P", "-n", "-S", "2026-01-01"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        lines += [line.split("|") for line in out.splitlines() if line]
    steps = pd.DataFrame(lines, columns=fields.split(","))
    steps["job"] = steps.JobIDRaw.str.split(".").str[0]
    steps["rss_gib"] = steps.MaxRSS.map(to_gib)
    head = steps[~steps.JobIDRaw.str.contains(r"\.")]
    head = head.drop_duplicates("job", keep="last").set_index(
        "job"
    )  # requeued jobs repeat
    gpu = head.AllocTRES.str.extract(r"gres/gpu:(\w+)=(\d+)")
    return pd.DataFrame(
        {
            "hours": head.ElapsedRaw.astype(float) / 3600,
            "req_gib": head.ReqMem.map(to_gib),
            "peak_gib": steps.groupby("job").rss_gib.max(),
            "gpu": gpu[0],
            "n_gpu": gpu[1].astype(float),
            "cpus": head.AllocTRES.str.extract(r"cpu=(\d+)")[0].astype(float),
            "node": head.NodeList,
        }
    )


def main() -> None:
    args = parse_args()
    runs = load_runs()
    print(f"runs {len(runs):,}: {runs.groupby('seed').size().to_dict()}")
    print("cluster and CPU:", runs.groupby(["cluster", "cpu"]).size().to_dict())
    print(
        "device per head:",
        runs.groupby(["group", "head"]).device.unique().map(list).to_dict(),
    )

    keys = ["group", "pool", "head"]
    timing = runs.groupby(keys).minutes.describe(percentiles=[0.5, 0.9])
    timing["total_h"] = runs.groupby(keys).minutes.sum() / 60
    print("\nMinutes per run:")
    print(timing[["count", "50%", "90%", "max", "total_h"]].round(1).to_string())

    killarney = runs[runs.cluster == "Killarney"]
    jobs = accounting(sorted(killarney.job.unique()))
    print(f"\njobs with accounting: {len(jobs)} of {killarney.job.nunique()}")
    print("GPUs:", jobs.groupby("gpu", dropna=False).size().to_dict())
    gpu_hours = (jobs.hours * jobs.n_gpu.fillna(0)).sum()
    print(f"job hours {jobs.hours.sum():,.0f}, GPU-hours {gpu_hours:,.0f}")
    groups = killarney.groupby("job")[keys].apply(lambda d: d.drop_duplicates())
    memory = groups.reset_index("job").join(jobs[["req_gib", "peak_gib"]], on="job")
    peak = memory.groupby(keys)[["req_gib", "peak_gib"]].agg(["median", "max"])
    print("\nMemory per job that trained each kind of run (GiB):")
    print(peak.round(1).to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "compute_runs.csv", index=False)
    jobs.to_csv(args.out_dir / "compute_jobs.csv")
    print(f"wrote compute_runs.csv and compute_jobs.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
