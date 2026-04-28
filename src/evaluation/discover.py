"""Walk results/predictions/ and group experiments into rating tracks.

An experiment is one trained model identified by the 5-tuple
(pool_type, split, model_family, task, embedding_model). The on-disk layout is

    results/predictions/<pool_type>/<split>/<model_family>/<task>/<embedding_model>/

Tracks aggregate experiments for a single global leaderboard:

  - track="mean":      pool_type ∈ {mean, per_residue}, task is a benchmark property
  - track="attention": pool_type == "attention",       task is a benchmark property
  - track="real_case": pool_type == "mean",            task is in benchmark_card.real_case
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

POOL_TYPES = {"mean", "attention", "per_residue"}


@dataclass(frozen=True)
class Experiment:
    pool_type: str
    split: str
    model_family: str
    task: str
    embedding_model: str
    run_dir: Path

    def task_type(self) -> str | None:
        """Read task_type from the run's metrics.json if present."""
        path = self.run_dir / "metrics.json"
        if not path.exists():
            return None
        try:
            md = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        for split in ("test", "val", "train"):
            block = md.get(split)
            if isinstance(block, dict) and "task_type" in block:
                return block["task_type"]
        return None


@dataclass(frozen=True)
class TrackKey:
    track: str
    split: str
    model_family: str

    def filename_stem(self) -> str:
        return f"{self.track}__{self.split}__{self.model_family}"


def discover_experiments(predictions_root: Path) -> list[Experiment]:
    """Enumerate every experiment dir matching the canonical 5-level layout."""
    out: list[Experiment] = []
    if not predictions_root.exists():
        return out
    for pool_dir in sorted(predictions_root.iterdir()):
        if not pool_dir.is_dir() or pool_dir.name not in POOL_TYPES:
            continue
        for split_dir in sorted(pool_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for fam_dir in sorted(split_dir.iterdir()):
                if not fam_dir.is_dir():
                    continue
                for task_dir in sorted(fam_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    for emb_dir in sorted(task_dir.iterdir()):
                        if not emb_dir.is_dir():
                            continue
                        out.append(
                            Experiment(
                                pool_type=pool_dir.name,
                                split=split_dir.name,
                                model_family=fam_dir.name,
                                task=task_dir.name,
                                embedding_model=emb_dir.name,
                                run_dir=emb_dir,
                            )
                        )
    return out


def assign_track(exp: Experiment, real_case_tasks: set[str]) -> str | None:
    """Map an experiment to its leaderboard track or None if it does not fit."""
    if exp.task in real_case_tasks:
        if exp.pool_type == "mean":
            return "real_case"
        # Attention real_case is not a planned track today; ignore.
        return None
    if exp.pool_type == "attention":
        return "attention"
    if exp.pool_type in {"mean", "per_residue"}:
        return "mean"
    return None


def group_into_tracks(
    experiments: list[Experiment],
    real_case_tasks: set[str],
) -> dict[TrackKey, list[Experiment]]:
    """Bucket experiments by (track, split, model_family)."""
    groups: dict[TrackKey, list[Experiment]] = {}
    for exp in experiments:
        track = assign_track(exp, real_case_tasks)
        if track is None:
            continue
        key = TrackKey(track=track, split=exp.split, model_family=exp.model_family)
        groups.setdefault(key, []).append(exp)
    return groups


def task_signature(exp: Experiment) -> tuple[str, str, str, str]:
    """Identifier shared by experiments that should share bootstrap resample indices."""
    return (exp.pool_type, exp.split, exp.model_family, exp.task)


def load_real_case_tasks(benchmark_card_path: Path) -> set[str]:
    if not benchmark_card_path.exists():
        return set()
    card = json.loads(benchmark_card_path.read_text())
    real_case = card.get("real_case") or {}
    if isinstance(real_case, dict):
        return set(real_case.keys())
    if isinstance(real_case, list):
        return set(real_case)
    return set()
