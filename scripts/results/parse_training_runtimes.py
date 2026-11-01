"""Parse enzymado_train_*/enzymado_attn_*.out logs into a long-format CSV
recording, per (job, embedding, task) block: how long it took and what it
depended on (embedding dim, dataset size, split, route, model families).

Cross-references `sacct` for the ground-truth job outcome (COMPLETED /
TIMEOUT / OUT_OF_MEMORY / ...). Output feeds
results/analysis/README_training_runtime.md.

Usage: run from the repo root, after `source .venv/bin/activate`:
    python scripts/results/parse_training_runtimes.py
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

from enzymado import paths_and_constants

_P = paths_and_constants.Paths
REPO = _P.project_root

with (REPO / "embeddings_models_output.json").open(encoding="utf-8") as _emb_fh:
    EMB_DIMS = {k: v["dim"] for k, v in json.load(_emb_fh).items()}

TS_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]")
HEADER_EMB_RE = re.compile(r"^Embedding model: (\S+)")
HEADER_SPLIT_RE = re.compile(r"^Split variant: (\S+)")
HEADER_PREFIX_RE = re.compile(r"^Model prefix: (\S+)")
HEADER_MODES_RE = re.compile(r"^Embedding modes: (\S+)")
SECTION_RE = re.compile(r"^=== Embedding: (\S+) \(dim=(\d+)\) ===")
TASK_RE = re.compile(r"Task: (\S+) \| type: (\S+) \| split: (\S+)")
ROUTE_RE = re.compile(r"\[(\w+)\] n_train=([\d,]+), n_test=([\d,]+) \| route=(\S+)")
ROWS_RE = re.compile(r"Rows loaded: train=(\d+), val=(\d+), test=(\d+)")
ERROR_RE = re.compile(r"\[ERROR\] Task '(\S+)' failed: (.+)")
SKIP_RE = re.compile(r"(\d+)/(\d+) models already done: \[(.*)\]")


def family_of(model_name: str) -> str:
    """Bucket a specific model_name key into its broad family."""
    if "xgb" in model_name:
        return "xgboost"
    if "mlp" in model_name:
        return "mlp"
    if "linear" in model_name:
        return "linear"
    return model_name


def parse_ts(ts_str, day_anchor):
    return datetime.combine(day_anchor, datetime.strptime(ts_str, "%H:%M:%S").time())


def parse_log(path: Path, launcher: str):
    """Yield one dict per Task-block found in a single .out log."""
    lines = path.read_text(errors="replace").splitlines()
    day_anchor = datetime.fromtimestamp(path.stat().st_mtime).date()

    header = {
        "embedding": None,
        "split_variant": None,
        "model_prefix": None,
        "embedding_modes": None,
    }
    emb_dim = None
    cur_task = None
    last_ts = None
    blocks = []

    def close_task(end_ts):
        if (
            cur_task is not None
            and cur_task.get("start_ts") is not None
            and end_ts is not None
        ):
            dur = (end_ts - cur_task["start_ts"]).total_seconds()
            if dur < 0:
                dur += 86400  # midnight wrap
            cur_task["duration_s"] = dur
        if cur_task is not None:
            blocks.append(cur_task)

    for line in lines:
        m = HEADER_EMB_RE.match(line)
        if m:
            header["embedding"] = m.group(1)
            continue
        m = HEADER_SPLIT_RE.match(line)
        if m:
            header["split_variant"] = m.group(1)
            continue
        m = HEADER_PREFIX_RE.match(line)
        if m:
            header["model_prefix"] = m.group(1)
            continue
        m = HEADER_MODES_RE.match(line)
        if m:
            header["embedding_modes"] = m.group(1)
            continue
        m = SECTION_RE.match(line)
        if m:
            emb_dim = int(m.group(2))
            continue

        ts_m = TS_RE.match(line)
        ts = None
        if ts_m:
            ts = parse_ts(ts_m.group(1), day_anchor)
            last_ts = ts

        # NOTE: the timestamp for a "Task:" line is on the *preceding* blank
        # log line, not the Task line itself -- fall back to the last seen ts.
        m = TASK_RE.search(line)
        if m:
            task_ts = ts if ts is not None else last_ts
            close_task(task_ts)
            cur_task = {
                "log_file": path.name,
                "launcher": launcher,
                "embedding": header["embedding"],
                "embedding_dim": emb_dim,
                "split_variant_hdr": header["split_variant"],
                "model_prefix": header["model_prefix"],
                "embedding_modes": header["embedding_modes"],
                "task_name": m.group(1),
                "task_type": m.group(2),
                "split": m.group(3),
                "start_ts": task_ts,
                "n_train": None,
                "n_test": None,
                "route": None,
                "models_trained": [],
                "models_skipped": None,
                "error": None,
                "duration_s": None,
            }
            continue

        if cur_task is None:
            continue

        m = SKIP_RE.search(line)
        if m:
            cur_task["models_skipped"] = m.group(3).replace("'", "").split(", ")
            continue

        m = ROWS_RE.search(line)
        if m:
            cur_task["n_train_raw"] = int(m.group(1))
            cur_task["n_val_raw"] = int(m.group(2))
            cur_task["n_test_raw"] = int(m.group(3))
            continue

        m = ROUTE_RE.search(line)
        if m:
            cur_task["n_train"] = int(m.group(2).replace(",", ""))
            cur_task["n_test"] = int(m.group(3).replace(",", ""))
            cur_task["route"] = m.group(4)
            cur_task["models_trained"].append(m.group(1))
            continue

        m = ERROR_RE.search(line)
        if m:
            cur_task["error"] = m.group(2)[:200]
            continue

    close_task(last_ts)
    return blocks


def load_sacct(user: str, since: str):
    """Map (job_id, array_idx) to sacct state, memory, and elapsed time."""
    proc = subprocess.run(
        [
            "sacct",
            "-u",
            user,
            f"--starttime={since}",
            "--format=JobID,State,ExitCode,Elapsed,ReqMem",
            "-X",
            "-P",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = proc.stdout.splitlines()
    header = lines[0].split("|")
    out = {}
    for line in lines[1:]:
        rec = dict(zip(header, line.split("|")))
        m = re.match(r"^(\d+)_(\d+)$", rec.get("JobID", ""))
        if not m:
            continue
        job_id, array_idx = m.group(1), m.group(2)
        h, mi, s = (rec.get("Elapsed") or "0:00:00").split(":")
        out[(job_id, array_idx)] = {
            "state": rec.get("State", ""),
            "req_mem": rec.get("ReqMem", ""),
            "job_elapsed_s": int(h) * 3600 + int(mi) * 60 + int(s),
        }
    return out


def main():
    import getpass

    sacct = load_sacct(getpass.getuser(), since="2026-01-01")
    rows = []
    for pattern, launcher in [
        ("enzymado_train_*.out", "no_attention"),
        ("enzymado_attn_*.out", "attention"),
    ]:
        for path in sorted(REPO.glob(pattern)):
            m = re.match(r"enzymado_(?:train|attn)_(\d+)_(\d+)\.out", path.name)
            if not m:
                continue
            job_id, array_idx = m.group(1), m.group(2)
            try:
                blocks = parse_log(path, launcher)
            except Exception as e:  # noqa: BLE001 - continue parsing other logs.
                print(f"parse error {path}: {e}")
                continue
            sacct_rec = sacct.get((job_id, array_idx), {})
            for b in blocks:
                emb = b["embedding"] or "?"
                dim = b["embedding_dim"] or EMB_DIMS.get(emb)
                families = sorted(
                    {family_of(name) for name in b["models_trained"]}
                ) or ([b["model_prefix"].rstrip("_")] if b["model_prefix"] else [])
                rows.append(
                    {
                        "job_id": job_id,
                        "array_idx": array_idx,
                        "launcher": launcher,
                        "slurm_state": sacct_rec.get("state", ""),
                        "slurm_req_mem": sacct_rec.get("req_mem", ""),
                        "slurm_job_total_elapsed_s": sacct_rec.get("job_elapsed_s", ""),
                        "embedding": emb,
                        "embedding_dim": dim,
                        "task_name": b["task_name"],
                        "task_type": b["task_type"],
                        "split": b["split"],
                        "model_prefix_filter": b["model_prefix"] or "",
                        "families_trained_this_block": ";".join(families),
                        "models_skipped_already_done": ";".join(
                            b["models_skipped"] or []
                        ),
                        "n_train_rows": b.get("n_train_raw"),
                        "n_train_residues_concat": b.get("n_train"),
                        "route": b.get("route"),
                        "duration_seconds": round(b["duration_s"], 1)
                        if b["duration_s"] is not None
                        else None,
                        "error": b.get("error") or "",
                    }
                )

    out_path = _P.results_dir / "analysis" / "training_runtime_breakdown.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
