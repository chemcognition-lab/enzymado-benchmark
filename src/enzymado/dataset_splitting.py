"""Generate random or similarity-aware dataset split CSV files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from enzymado.splitting.artifacts import (
    build_metadata_path,
    build_output_csv_path,
    resolve_csv_inputs,
    write_split_dataset,
    write_split_metadata,
)
from enzymado.splitting.core import (
    detect_id_column,
    detect_sequence_column,
    read_csv_rows,
    validate_split_fractions,
)
from enzymado.splitting.random_splits import (
    generate_point_stratified_split_rows,
    generate_random_split_rows,
)
from enzymado.splitting.similarity import (
    build_mmseqs_cluster_command,
    generate_similarity_split_from_clusters,
    generate_similarity_split_rows,
)
from enzymado.splitting.standardization import (
    build_standardized_dataset_rows,
    resolve_standardization_columns,
)

type CsvRow = dict[str, str]
type SplitSummary = dict[str, dict[str, int]]


@dataclass(frozen=True)
class _SplitRequest:
    """Resolved options needed to assign rows to dataset splits."""

    mode: str
    train_fraction: float
    val_fraction: float
    test_fraction: float
    sequence_col: str
    group_col: str | None
    point_id_col: str | None
    random_split_unit: str
    stratify_col: str | None
    stratify_bins: int
    identity_threshold: float
    coverage_threshold: float | None
    coverage_mode: int | None
    cluster_mode: int | None
    similarity_type: int | None
    sensitivity: float | None
    e_value: float | None
    alignment_mode: int | None
    seq_id_mode: int | None
    max_seqs: int | None
    single_step_clustering: bool
    cluster_reassign: bool
    seed: int


@dataclass(frozen=True)
class _GeneratedSplit:
    """Rows, columns, and counts produced by one split strategy."""

    fieldnames: list[str]
    rows: list[CsvRow]
    summary: SplitSummary


@dataclass(frozen=True)
class _StandardizedSplit:
    """Generated split plus its target-standardization metadata."""

    fieldnames: list[str]
    rows: list[CsvRow]
    metadata: dict[str, object]


def _generate_random_split(
    rows: list[CsvRow],
    fieldnames: list[str],
    request: _SplitRequest,
) -> _GeneratedSplit:
    """Generate a protein-grouped or point-stratified random split."""
    if request.random_split_unit == "protein":
        output = generate_random_split_rows(
            rows,
            fieldnames,
            request.train_fraction,
            request.val_fraction,
            request.test_fraction,
            sequence_col=request.sequence_col,
            group_col=request.group_col,
            seed=request.seed,
        )
    elif request.random_split_unit == "point":
        if request.group_col is not None:
            raise ValueError(
                "group_col cannot be combined with random_split_unit='point'."
            )
        if request.point_id_col is None:
            raise ValueError(
                "Point-level splitting requires --point-id-col or a "
                "detectable ID column."
            )
        if request.stratify_col is None:
            raise ValueError(
                "Point-level splitting requires a continuous --stratify-col."
            )
        output = generate_point_stratified_split_rows(
            rows,
            fieldnames,
            request.train_fraction,
            request.val_fraction,
            request.test_fraction,
            sequence_col=request.sequence_col,
            point_id_col=request.point_id_col,
            stratify_col=request.stratify_col,
            seed=request.seed,
            stratify_bins=request.stratify_bins,
        )
    else:
        raise ValueError(f"Unsupported random split unit: {request.random_split_unit}.")
    return _GeneratedSplit(*output)


def _generate_split(
    rows: list[CsvRow],
    fieldnames: list[str],
    request: _SplitRequest,
) -> _GeneratedSplit:
    """Dispatch a resolved request to its random or similarity strategy."""
    if request.mode == "random":
        return _generate_random_split(rows, fieldnames, request)
    if request.mode != "similarity":
        raise ValueError(f"Unsupported split mode: {request.mode}.")
    output = generate_similarity_split_rows(
        rows,
        fieldnames,
        request.train_fraction,
        request.val_fraction,
        request.test_fraction,
        sequence_col=request.sequence_col,
        group_col=request.group_col,
        identity_threshold=request.identity_threshold,
        coverage_threshold=request.coverage_threshold,
        coverage_mode=request.coverage_mode,
        cluster_mode=request.cluster_mode,
        similarity_type=request.similarity_type,
        sensitivity=request.sensitivity,
        e_value=request.e_value,
        alignment_mode=request.alignment_mode,
        seq_id_mode=request.seq_id_mode,
        max_seqs=request.max_seqs,
        single_step_clustering=request.single_step_clustering,
        cluster_reassign=request.cluster_reassign,
        seed=request.seed,
    )
    return _GeneratedSplit(*output)


def _standardize_generated_split(
    generated: _GeneratedSplit,
    *,
    standardize: bool,
    standardization_cols: tuple[str, ...] | None,
) -> _StandardizedSplit:
    """Append standardized targets and describe how columns were selected."""
    if not standardize:
        return _StandardizedSplit(
            generated.fieldnames,
            generated.rows,
            {
                "enabled": False,
                "skipped_reason": "Standardization was disabled for this run.",
            },
        )

    resolved_cols, skipped_cols, source = resolve_standardization_columns(
        generated.fieldnames,
        generated.rows,
        standardization_cols=standardization_cols,
    )
    metadata: dict[str, object] = {
        "enabled": bool(resolved_cols),
        "source": source,
        "target_columns": resolved_cols,
        "skipped_columns": skipped_cols,
    }
    if not resolved_cols:
        metadata["skipped_reason"] = (
            "No scalar numeric target columns were inferred. Use --standardize-col "
            "to opt in explicitly."
        )
        return _StandardizedSplit(
            generated.fieldnames,
            generated.rows,
            metadata,
        )

    fieldnames, rows, fitted_metadata = build_standardized_dataset_rows(
        generated.rows,
        generated.fieldnames,
        resolved_cols,
    )
    metadata.update(fitted_metadata)
    return _StandardizedSplit(fieldnames, rows, metadata)


def _build_split_metadata(
    input_csv_path: Path,
    output_csv_path: Path,
    request: _SplitRequest,
    generated: _GeneratedSplit,
    standardized: _StandardizedSplit,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build stable shared and mode-specific metadata payloads."""
    shared_fields: dict[str, object] = {
        "input_csv_path": str(input_csv_path),
        "sequence_col": request.sequence_col,
        "group_col": request.group_col,
        "id_col": request.point_id_col,
        "train_fraction": request.train_fraction,
        "val_fraction": request.val_fraction,
        "test_fraction": request.test_fraction,
        "seed": request.seed,
    }
    mode_payload: dict[str, object] = {
        "output_csv_path": str(output_csv_path),
        "summary": generated.summary,
        "standardization": standardized.metadata,
    }
    if request.mode == "random":
        is_point_split = request.random_split_unit == "point"
        mode_payload["split_strategy"] = {
            "unit": request.random_split_unit,
            "point_id_col": request.point_id_col if is_point_split else None,
            "stratify_col": request.stratify_col if is_point_split else None,
            "stratify_bins": request.stratify_bins if is_point_split else None,
            "within_sequence": is_point_split,
        }
    else:
        mode_payload["mmseqs"] = {
            "min_seq_id": request.identity_threshold,
            "coverage_threshold": request.coverage_threshold,
            "coverage_mode": request.coverage_mode,
            "cluster_mode": request.cluster_mode,
            "similarity_type": request.similarity_type,
            "sensitivity": request.sensitivity,
            "e_value": request.e_value,
            "alignment_mode": request.alignment_mode,
            "seq_id_mode": request.seq_id_mode,
            "max_seqs": request.max_seqs,
            "single_step_clustering": request.single_step_clustering,
            "cluster_reassign": request.cluster_reassign,
        }
    return shared_fields, mode_payload


def split_dataset_csv(
    input_csv_path: Path,
    output_csv_path: Path,
    *,
    mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    sequence_col: str | None = None,
    group_col: str | None = None,
    random_split_unit: str = "protein",
    point_id_col: str | None = None,
    stratify_col: str | None = None,
    stratify_bins: int = 10,
    identity_threshold: float = 0.3,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
    seed: int = 42,
    standardize: bool = True,
    standardization_cols: tuple[str, ...] | None = None,
    metadata_path: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Generate one split CSV and merge its metadata into the shared JSON."""
    fieldnames, rows = read_csv_rows(input_csv_path)
    if not rows:
        raise ValueError(f"{input_csv_path} does not contain any data rows.")

    resolved_sequence_col = detect_sequence_column(
        fieldnames, sequence_col=sequence_col
    )
    resolved_group_col = group_col
    resolved_id_col = detect_id_column(fieldnames)
    resolved_point_id_col = point_id_col or resolved_id_col
    request = _SplitRequest(
        mode=mode,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        sequence_col=resolved_sequence_col,
        group_col=resolved_group_col,
        point_id_col=resolved_point_id_col,
        random_split_unit=random_split_unit,
        stratify_col=stratify_col,
        stratify_bins=stratify_bins,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        coverage_mode=coverage_mode,
        cluster_mode=cluster_mode,
        similarity_type=similarity_type,
        sensitivity=sensitivity,
        e_value=e_value,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        max_seqs=max_seqs,
        single_step_clustering=single_step_clustering,
        cluster_reassign=cluster_reassign,
        seed=seed,
    )
    generated = _generate_split(rows, fieldnames, request)

    standardized = _standardize_generated_split(
        generated,
        standardize=standardize,
        standardization_cols=standardization_cols,
    )
    write_split_dataset(
        output_csv_path,
        standardized.rows,
        standardized.fieldnames,
    )

    if metadata_path is None:
        metadata_path = build_metadata_path(
            input_csv_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            output_root=output_csv_path.parent,
        )

    shared_fields, mode_payload = _build_split_metadata(
        input_csv_path,
        output_csv_path,
        request,
        generated,
        standardized,
    )
    write_split_metadata(
        metadata_path,
        mode=mode,
        shared_fields=shared_fields,
        mode_payload=mode_payload,
    )
    return generated.summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for split generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate train/val/test splits for one CSV or a directory of CSV "
            "datasets. Supports random grouped splits and similarity-aware splits "
            "using MMseqs2."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file or a directory containing CSV datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional directory where generated split files will be written. "
            "If omitted, each split CSV is written next to the source dataset."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["random", "similarity"],
        required=True,
        help="Split strategy to use.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction for the train split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        required=True,
        help="Fraction for the validation split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        required=True,
        help="Fraction for the test split.",
    )
    parser.add_argument(
        "--sequence-col",
        type=str,
        default=None,
        help="Sequence column. If omitted, it will be autodetected.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=None,
        help=(
            "Optional grouping column so all rows for the same protein stay together. "
            "If omitted, grouping is done by the protein sequence."
        ),
    )
    parser.add_argument(
        "--random-split-unit",
        choices=["protein", "point"],
        default="protein",
        help=(
            "Unit used by random splits. 'protein' keeps all rows for a protein "
            "together; 'point' stratifies measurement points by a continuous target."
        ),
    )
    parser.add_argument(
        "--point-id-col",
        type=str,
        default=None,
        help="Point identifier column used by point-level random splits.",
    )
    parser.add_argument(
        "--stratify-col",
        type=str,
        default=None,
        help="Continuous target column used for point-level stratification.",
    )
    parser.add_argument(
        "--stratify-bins",
        type=int,
        default=10,
        help="Maximum number of rank-quantile bins for point-level stratification.",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=0.3,
        help="Minimum sequence identity used to cluster similar proteins.",
    )
    parser.add_argument(
        "--mmseqs-coverage-threshold",
        type=float,
        default=None,
        help="Optional MMseqs2 alignment coverage threshold (-c).",
    )
    parser.add_argument(
        "--mmseqs-coverage-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 coverage mode (--cov-mode).",
    )
    parser.add_argument(
        "--mmseqs-cluster-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 clustering strategy (--cluster-mode).",
    )
    parser.add_argument(
        "--mmseqs-similarity-type",
        type=int,
        default=None,
        help="Optional MMseqs2 similarity metric (--similarity-type).",
    )
    parser.add_argument(
        "--mmseqs-sensitivity",
        type=float,
        default=None,
        help="Optional MMseqs2 sensitivity (-s).",
    )
    parser.add_argument(
        "--mmseqs-evalue",
        type=float,
        default=None,
        help="Optional MMseqs2 E-value threshold (-e).",
    )
    parser.add_argument(
        "--mmseqs-alignment-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 alignment mode (--alignment-mode).",
    )
    parser.add_argument(
        "--mmseqs-seq-id-mode",
        type=int,
        default=None,
        help="Optional MMseqs2 sequence identity mode (--seq-id-mode).",
    )
    parser.add_argument(
        "--mmseqs-max-seqs",
        type=int,
        default=None,
        help="Optional MMseqs2 prefilter cap (--max-seqs).",
    )
    parser.add_argument(
        "--mmseqs-single-step-clustering",
        action="store_true",
        help="Enable MMseqs2 single-step clustering (--single-step-clustering 1).",
    )
    parser.add_argument(
        "--mmseqs-cluster-reassign",
        action="store_true",
        help="Enable MMseqs2 cluster reassignment (--cluster-reassign 1).",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help=(
            "Optional filename tag so exploratory split variants do not "
            "overwrite each other."
        ),
    )
    parser.add_argument(
        "--standardize-col",
        action="append",
        default=None,
        help=(
            "Numeric target column to z-score after splitting. Repeat for multiple "
            "columns. If omitted, common scalar regression target columns are "
            "detected automatically."
        ),
    )
    parser.add_argument(
        "--no-standardization",
        action="store_true",
        help=(
            "Disable writing the standardized full-data sidecar CSV and scaler "
            "metadata."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the split assignment.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and generate split files for one or more datasets."""
    args = build_argument_parser().parse_args()

    if args.mode == "similarity" and shutil.which("mmseqs") is None:
        raise SystemExit(
            "mmseqs is not available in PATH. Activate the conda environment where "
            "you installed MMseqs2 and retry."
        )

    csv_inputs = resolve_csv_inputs(args.input)
    if not csv_inputs:
        raise SystemExit(f"No CSV files found in {args.input}")

    processed = 0
    skipped: list[tuple[str, str]] = []
    for csv_input in csv_inputs:
        output_csv_path = build_output_csv_path(
            csv_input,
            mode=args.mode,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            output_root=args.output_root,
            output_tag=args.output_tag,
        )
        metadata_path = build_metadata_path(
            csv_input,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            output_root=args.output_root,
            output_tag=args.output_tag,
        )
        try:
            summary = split_dataset_csv(
                csv_input,
                output_csv_path,
                mode=args.mode,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                sequence_col=args.sequence_col,
                group_col=args.group_col,
                random_split_unit=args.random_split_unit,
                point_id_col=args.point_id_col,
                stratify_col=args.stratify_col,
                stratify_bins=args.stratify_bins,
                identity_threshold=args.identity_threshold,
                coverage_threshold=args.mmseqs_coverage_threshold,
                coverage_mode=args.mmseqs_coverage_mode,
                cluster_mode=args.mmseqs_cluster_mode,
                similarity_type=args.mmseqs_similarity_type,
                sensitivity=args.mmseqs_sensitivity,
                e_value=args.mmseqs_evalue,
                alignment_mode=args.mmseqs_alignment_mode,
                seq_id_mode=args.mmseqs_seq_id_mode,
                max_seqs=args.mmseqs_max_seqs,
                single_step_clustering=args.mmseqs_single_step_clustering,
                cluster_reassign=args.mmseqs_cluster_reassign,
                seed=args.seed,
                standardize=not args.no_standardization,
                standardization_cols=(
                    tuple(args.standardize_col)
                    if args.standardize_col is not None
                    else None
                ),
                metadata_path=metadata_path,
            )
        except Exception as exc:  # noqa: BLE001 - directory runs are best-effort.
            skipped.append((str(csv_input), str(exc)))
            print(f"SKIPPED {csv_input}: {exc}")
            continue

        processed += 1
        print(f"{csv_input} -> {output_csv_path}")
        print(json.dumps(summary, indent=2))

    print(f"Processed datasets: {processed}")
    if skipped:
        print("Skipped datasets:")
        for path, reason in skipped:
            print(f"- {path}: {reason}")


__all__ = [
    "build_argument_parser",
    "build_metadata_path",
    "build_mmseqs_cluster_command",
    "build_output_csv_path",
    "build_standardized_dataset_rows",
    "generate_point_stratified_split_rows",
    "generate_random_split_rows",
    "generate_similarity_split_from_clusters",
    "generate_similarity_split_rows",
    "main",
    "resolve_standardization_columns",
    "split_dataset_csv",
    "validate_split_fractions",
    "write_split_dataset",
    "write_split_metadata",
]


if __name__ == "__main__":
    main()
