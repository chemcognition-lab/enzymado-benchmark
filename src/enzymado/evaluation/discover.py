"""Discover benchmark experiments and identify shared resampling groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        """Infer task type, preferring hparams written before training.

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


def discover_experiments(predictions_root: Path) -> list[Experiment]:
    """Enumerate every experiment dir matching the canonical 6-level layout.

    Globbing one scope at a time keeps the walk pruned to the two known scopes,
    and sorting the name tuples reproduces the lexicographic order a nested
    ``sorted(iterdir())`` descent produced.
    """
    if not predictions_root.exists():
        return []
    layouts = sorted(
        run_dir.relative_to(predictions_root).parts
        for scope in TASK_SCOPES
        for run_dir in (predictions_root / scope).glob("*/*/*/*/*")
        if run_dir.is_dir()
    )
    out: list[Experiment] = []
    for scope, pool, split, family, task, embedding in layouts:
        if pool not in POOL_TYPES:
            continue
        out.append(
            Experiment(
                task_scope=scope,
                pool_type=pool,
                split=split,
                model_family=family,
                task=task,
                embedding_model=embedding,
                run_dir=predictions_root.joinpath(
                    scope, pool, split, family, task, embedding
                ),
            )
        )
    return out


def task_signature(exp: Experiment) -> tuple[str, str]:
    """Identifier shared by experiments that should share bootstrap resample indices.

    Every competitor for one task and split is scored on the same test rows, so they
    all belong in one resampling group: pairing the bootstrap is what lets a paired
    difference cancel the shared sampling noise. Including the model family or pool
    type here would split that group and force most comparisons to use independent
    indices, inflating their variance for no reason. ``paired_by_sample_id`` in
    compute_bootstrap_metrics still verifies each run's ordered sample fingerprint
    before sharing indices, so a genuinely different test set falls back safely.
    """
    return (exp.split, exp.task)
