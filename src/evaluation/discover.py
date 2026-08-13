"""Walk results/predictions/ and group experiments into rating tracks.

An experiment is one trained model identified by the 6-tuple
(task_scope, pool_type, split, model_family, task, embedding_model). The
on-disk layout is

    results/predictions/<task_scope>/<pool_type>/<split>/<model_family>/<task>/<embedding_model>/

``task_scope`` ("properties" or "real_case") is the top-level bucket, matching
``TaskConfig.task_scope`` in TASK_REGISTRY. Tracks aggregate experiments for a
single global leaderboard:

  - track="no_attention":        task_scope == "properties", pool_type == "no_attention"
  - track="attention":           task_scope == "properties", pool_type == "attention"
  - track="real_case":           task_scope == "real_case",  pool_type == "no_attention"
  - track="real_case_attention": task_scope == "real_case",  pool_type == "attention"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src import data_loading

from .loaders import derive_task_type, read_hparams, read_metrics

TASK_SCOPES = {"properties", "real_case"}
POOL_TYPES = {"attention", "no_attention"}


@dataclass(frozen=True)
class Experiment:
    task_scope: str
    pool_type: str
    split: str
    model_family: str
    task: str
    embedding_model: str
    run_dir: Path

    def task_type(self) -> str | None:
        """Infer task_type, preferring hparams.json (written immediately after training).

        Falls back to metrics.json for older runs whose hparams.json predates the
        task_config block.
        """
        task_type = derive_task_type(read_hparams(self.run_dir))
        if task_type is not None:
            return task_type
        md = read_metrics(self.run_dir)
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
    """Enumerate every experiment dir matching the canonical 6-level layout."""
    out: list[Experiment] = []
    if not predictions_root.exists():
        return out
    for scope_dir in sorted(predictions_root.iterdir()):
        if not scope_dir.is_dir() or scope_dir.name not in TASK_SCOPES:
            continue
        for pool_dir in sorted(scope_dir.iterdir()):
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
                                    task_scope=scope_dir.name,
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
    """Map an experiment to its leaderboard track or None if it does not fit.

    ``real_case_tasks`` is used only as a consistency check against the
    physical ``task_scope`` folder (the authoritative signal); a mismatch
    means the on-disk layout and TASK_REGISTRY have drifted apart.
    """
    scope_says_real_case = exp.task in real_case_tasks
    if exp.task_scope == "real_case" and not scope_says_real_case:
        raise ValueError(
            f"Task '{exp.task}' lives under a real_case/ dir but TASK_REGISTRY "
            "does not tag it task_scope='real_case'."
        )
    if exp.task_scope == "properties" and scope_says_real_case:
        raise ValueError(
            f"Task '{exp.task}' lives under a properties/ dir but TASK_REGISTRY "
            "tags it task_scope='real_case'."
        )

    if exp.task_scope == "real_case":
        return "real_case" if exp.pool_type == "no_attention" else "real_case_attention"
    if exp.pool_type == "attention":
        return "attention"
    if exp.pool_type == "no_attention":
        return "no_attention"
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


def task_signature(exp: Experiment) -> tuple[str, str, str]:
    """Identify experiments evaluated on the same ordered test samples."""
    return (exp.task_scope, exp.split, exp.task)


def load_real_case_tasks() -> set[str]:
    return {
        name
        for name, cfg in data_loading.TASK_REGISTRY.items()
        if cfg.task_scope == "real_case"
    }
