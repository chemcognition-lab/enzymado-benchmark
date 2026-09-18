"""Resolve evaluation task types from benchmark metadata.

Example:
    python -m enzymado.evaluation.task_types benchmark_card.json ec_number
"""

from __future__ import annotations

import argparse
import json
import pathlib


def from_label_kind(label_kind: str | None) -> str | None:
    """Map a dataset label representation to its evaluation task type."""
    if label_kind in {
        "binary",
        "multilabel_ec",
        "residue_binary_vector",
    }:
        return "classification"
    if label_kind == "scalar_regression":
        return "regression"
    return None


def from_hparams(hparams: dict[str, object]) -> str | None:
    """Resolve a task type from one experiment's hyperparameters."""
    task_config = hparams.get("task_config") or {}
    if not isinstance(task_config, dict):
        return None
    label_kind = task_config.get("label_kind")
    return from_label_kind(label_kind if isinstance(label_kind, str) else None)


def from_benchmark_card(
    task_name: str,
    benchmark_card_path: pathlib.Path,
) -> str | None:
    """Resolve a task type from the benchmark's documented task registry."""
    if not benchmark_card_path.exists():
        return None

    card: dict[str, object] = json.loads(
        benchmark_card_path.read_text(encoding="utf-8")
    )
    for scope_name in ("properties", "real_case"):
        scope = card.get(scope_name)
        if not isinstance(scope, dict):
            continue
        task = scope.get(task_name)
        if not isinstance(task, dict):
            continue
        task_type = task.get("task_type")
        if task_type not in {"classification", "regression"}:
            raise ValueError(
                f"Task '{task_name}' has invalid task_type={task_type!r} "
                f"in {benchmark_card_path}."
            )
        return task_type
    return None


def resolve(
    task_name: str,
    benchmark_card_path: pathlib.Path,
    hparams: dict[str, object] | None = None,
) -> str | None:
    """Resolve from the benchmark card, falling back to run metadata."""
    card_task_type = from_benchmark_card(task_name, benchmark_card_path)
    if card_task_type is not None:
        return card_task_type
    return from_hparams(hparams or {})


def main() -> None:
    """Print the resolved type for one benchmark task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_card", type=pathlib.Path)
    parser.add_argument("task_name")
    args = parser.parse_args()

    task_type = from_benchmark_card(args.task_name, args.benchmark_card)
    if task_type is None:
        raise ValueError(f"Task '{args.task_name}' is not in {args.benchmark_card}.")
    print(task_type)


if __name__ == "__main__":
    main()
