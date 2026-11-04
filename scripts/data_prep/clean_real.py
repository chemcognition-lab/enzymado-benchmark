#!/usr/bin/env python3
"""Normalize the repository's real-case datasets into benchmark-ready CSV files.

Real case = case studies (the code says real_case, the docs case studies).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.data_prep.real_cases import (
    align_tournament,
    cleveland_galox,
    erickson_petase,
    utils,
)

from enzymado import paths_and_constants

_P = paths_and_constants.Paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Build argument parser for cleaning Erickson, Cleeveland, or ALIGN datasets."""
    paths = _P()
    parser = argparse.ArgumentParser(
        description=(
            "Clean the Erickson, Cleeveland, or ALIGN datasets into interim and "
            "processed outputs."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["erickson", "cleeveland", "align"],
        default="erickson",
        help="Dataset cleaner to run. Default: erickson",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=paths.data_raw_dir / "erickson" / "sequences.csv",
        help="Path to the Erickson sequences file.",
    )
    parser.add_argument(
        "--values",
        type=Path,
        default=paths.data_raw_dir / "erickson" / "values.csv",
        help="Path to the Erickson values file.",
    )
    parser.add_argument(
        "--dataset-table",
        type=Path,
        default=paths.data_raw_dir / "cleeveland" / "og_dataset_clev.csv",
        help="Path to the Cleeveland activity table.",
    )
    parser.add_argument(
        "--align-family",
        choices=(*align_tournament.ALIGN_FAMILIES, "all"),
        default="all",
        help="ALIGN family to clean when --dataset align is used. Default: all",
    )
    parser.add_argument(
        "--align-root",
        type=Path,
        default=paths.data_raw_dir / "align",
        help="Root directory containing ALIGN datasets.",
    )
    parser.add_argument(
        "--proteins",
        type=Path,
        default=paths.data_raw_dir / "cleeveland" / "proteins.csv",
        help="Path to the Cleeveland proteins file.",
    )
    parser.add_argument(
        "--substrates",
        type=Path,
        default=paths.data_raw_dir / "cleeveland" / "substrates.csv",
        help="Path to the Cleeveland substrates file.",
    )
    parser.add_argument(
        "--out-interim",
        type=Path,
        default=paths.data_interim_dir / "erickson" / "erickson_long.csv",
        help="Output path for the interim dataset.",
    )
    parser.add_argument(
        "--out-processed",
        type=Path,
        default=paths.data_processed_properties / "aromatic_products" / "erickson.csv",
        help=(
            "Output path for the processed dataset, or a directory root for ALIGN "
            "outputs."
        ),
    )
    parser.add_argument(
        "--benchmark-card",
        type=Path,
        default=paths.project_root / "benchmark_card.json",
        help="Benchmark card to synchronize after cleaning ALIGN datasets.",
    )
    parser.add_argument(
        "--skip-benchmark-card",
        action="store_true",
        help="Do not update benchmark_card.json after cleaning ALIGN datasets.",
    )
    return parser


def _write_align_family_outputs(
    family: str,
    args: argparse.Namespace,
    paths: _P,
    default_erickson_interim: Path,
    processed_dir: Path,
) -> tuple[int, int]:
    """Clean one ALIGN family and write its interim and per-task processed CSVs."""
    family_slug = utils.slugify(family)
    out_interim = align_tournament.resolve_align_interim_path(
        args.out_interim,
        default_erickson_interim,
        paths,
        family_slug,
    )
    interim_fieldnames, interim_rows, processed_outputs = (
        align_tournament.build_align_rows(
            family,
            args.align_root,
        )
    )
    utils.write_csv(out_interim, interim_fieldnames, interim_rows)

    processed_count = 0
    for _col, task_name, fieldnames, rows in processed_outputs:
        task_dir = align_tournament.resolve_align_task_processed_dir(
            processed_dir, family, task_name
        )
        utils.write_csv(task_dir / f"{family_slug}__{task_name}.csv", fieldnames, rows)
        processed_count += len(rows)

    return len(interim_rows), processed_count


def _run_align_cleaner(
    args: argparse.Namespace,
    paths: _P,
    default_erickson_interim: Path,
    default_erickson_processed: Path,
) -> None:
    """Execute the ALIGN cleaner across selected enzyme families."""
    align_families = (
        align_tournament.ALIGN_FAMILIES
        if args.align_family == "all"
        else (args.align_family,)
    )
    align_out_interim_root = (
        paths.data_interim_dir / "align"
        if args.out_interim == default_erickson_interim
        else args.out_interim
    )
    processed_dir = align_tournament.resolve_align_processed_dir(
        args.out_processed,
        default_erickson_processed,
        paths,
    )

    total_interim_rows = 0
    total_processed_rows = 0
    for family in align_families:
        n_interim, n_proc = _write_align_family_outputs(
            family,
            args,
            paths,
            default_erickson_interim,
            processed_dir,
        )
        total_interim_rows += n_interim
        total_processed_rows += n_proc

    if not args.skip_benchmark_card:
        updated_sections = align_tournament.sync_align_benchmark_card(
            args.benchmark_card,
            processed_dir,
            paths.project_root,
        )
        print(
            f"Synchronized {len(updated_sections)} ALIGN sections in "
            f"{args.benchmark_card}."
        )

    print(
        f"Wrote {total_interim_rows} interim rows under {align_out_interim_root} "
        f"and {total_processed_rows} task rows under {processed_dir}."
    )


def main() -> None:
    """Run the selected real-case (real_case = case studies) dataset cleaner."""
    paths = _P()
    args = build_argument_parser().parse_args()
    default_erickson_interim = paths.data_interim_dir / "erickson" / "erickson_long.csv"
    default_erickson_processed = (
        paths.data_processed_properties / "aromatic_products" / "erickson.csv"
    )

    match args.dataset:
        case "erickson":
            interim_rows, processed_rows = erickson_petase.build_erickson_rows(
                args.sequences, args.values
            )
            utils.write_csv(
                args.out_interim, erickson_petase.ERICKSON_INTERIM_COLUMNS, interim_rows
            )
            utils.write_csv(
                args.out_processed,
                erickson_petase.ERICKSON_PROCESSED_COLUMNS,
                processed_rows,
            )
        case "cleeveland":
            default_cleveland_interim = (
                paths.data_interim_dir / "cleeveland" / "cleeveland_long.csv"
            )
            default_cleveland_processed = (
                paths.data_processed_properties / "specific_activity" / "cleeveland.csv"
            )
            out_interim = (
                default_cleveland_interim
                if args.out_interim == default_erickson_interim
                else args.out_interim
            )
            out_processed = (
                default_cleveland_processed
                if args.out_processed == default_erickson_processed
                else args.out_processed
            )
            interim_rows, processed_rows = cleveland_galox.build_cleeveland_rows(
                args.dataset_table,
                args.proteins,
                args.substrates,
            )
            utils.write_csv(
                out_interim, cleveland_galox.CLEVELAND_INTERIM_COLUMNS, interim_rows
            )
            utils.write_csv(
                out_processed,
                cleveland_galox.CLEVELAND_PROCESSED_COLUMNS,
                processed_rows,
            )
        case "align":
            _run_align_cleaner(
                args, paths, default_erickson_interim, default_erickson_processed
            )
            return
        case _:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(
        f"Wrote {len(interim_rows)} interim rows to {args.out_interim} and "
        f"{len(processed_rows)} processed rows to {args.out_processed}."
    )


if __name__ == "__main__":
    main()
