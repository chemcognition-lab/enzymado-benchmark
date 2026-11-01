#!/usr/bin/env python3
"""Subset the canonical sequence index to the sequences used by one task.

Embedding the full index is a multi-day, multi-hundred-GB job. A single-task
experiment only needs the sequences that task actually references, which is a
small fraction of it. This writes a sequence index in the same schema so that
`make_fasta_from_unique_sequences.py` and the embedding scripts consume it
unchanged.
"""

import argparse
import csv
from pathlib import Path
import sys

from enzymado import paths_and_constants
from enzymado.data_loading.data_registry import TASK_REGISTRY
from enzymado.data_loading.stores import resolve_sequence_index_path
from enzymado.data_loading.task_csv import clean_text, normalize_sequence

_P = paths_and_constants.Paths
REPO_ROOT = _P.project_root


def collect_task_sequences(task_name: str) -> set[str]:
    """Return every normalized sequence referenced by a task, across its splits."""
    config = TASK_REGISTRY[task_name]
    sequences: set[str] = set()

    for dataset_path in config.dataset_paths.values():
        if not dataset_path.exists():
            continue
        with dataset_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sequence = normalize_sequence(row.get(config.sequence_column, ""))
                if sequence:
                    sequences.add(sequence)
                if config.wildtype_sequence_column:
                    wildtype = normalize_sequence(
                        row.get(config.wildtype_sequence_column, "")
                    )
                    if wildtype:
                        sequences.add(wildtype)

    return sequences


def write_subset(sequences: set[str], index_path: Path, output_csv: Path) -> int:
    """Write the index rows whose sequence is in the requested set."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched: set[str] = set()
    written = 0

    with index_path.open(newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        with output_csv.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(
                out_handle, fieldnames=["sequence_id", "protein_sequence"]
            )
            writer.writeheader()
            for row in reader:
                sequence = normalize_sequence(row.get("protein_sequence", ""))
                sequence_id = clean_text(row.get("sequence_id", ""))
                if not sequence or not sequence_id:
                    continue
                if sequence not in sequences or sequence in matched:
                    continue
                writer.writerow(
                    {"sequence_id": sequence_id, "protein_sequence": sequence}
                )
                matched.add(sequence)
                written += 1

    missing = len(sequences) - len(matched)
    if missing:
        print(
            f"WARNING: {missing} task sequences are absent from the sequence index.",
            file=sys.stderr,
        )
    return written


def create_parser() -> argparse.ArgumentParser:
    """Build the task-subset command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a sequence index containing only the sequences one registered "
            "task references, for cheap single-task embedding runs."
        )
    )
    parser.add_argument(
        "-t",
        "--task_name",
        required=True,
        choices=sorted(TASK_REGISTRY),
        help="Registered task whose sequences should be kept.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output CSV path, in sequence_id/protein_sequence schema.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Sequence index to subset. Defaults to the resolved project index.",
    )
    return parser


def main() -> None:
    """Subset the canonical sequence index down to one task's sequences."""
    args = create_parser().parse_args()

    index_path = args.input or resolve_sequence_index_path()
    if index_path is None or not index_path.exists():
        raise SystemExit(f"Sequence index not found: {index_path}")

    sequences = collect_task_sequences(args.task_name)
    if not sequences:
        raise SystemExit(f"Task '{args.task_name}' referenced no sequences.")

    written = write_subset(sequences, index_path, args.output)
    print(
        f"Task '{args.task_name}': {len(sequences)} unique sequences, "
        f"wrote {written} to {args.output}"
    )


if __name__ == "__main__":
    main()
