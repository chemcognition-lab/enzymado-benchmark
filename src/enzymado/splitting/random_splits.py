"""Grouped random and point-stratified dataset split algorithms."""

from __future__ import annotations

from collections import defaultdict
import math

from . import core, standardization


def generate_random_split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    group_col: str | None = None,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Generate random but group-aware split assignments."""
    fractions = core.validate_split_fractions(
        train_fraction, val_fraction, test_fraction
    )
    group_to_indices, _ = core.build_group_index(
        rows, sequence_col, group_col=group_col
    )
    group_weights = {
        group_key: len(indices) for group_key, indices in group_to_indices.items()
    }
    assignments = core.assign_items_to_splits(group_weights, fractions, seed=seed)
    output_fieldnames, split_rows = core.apply_group_assignments(
        rows,
        fieldnames,
        group_to_indices,
        assignments,
    )
    return (
        output_fieldnames,
        split_rows,
        core.summarize_split_rows(split_rows, sequence_col),
    )


def build_point_index(
    rows: list[dict[str, str]],
    *,
    sequence_col: str,
    point_id_col: str,
    stratify_col: str,
) -> tuple[dict[str, list[int]], dict[str, str], dict[str, float]]:
    """Index measurement points and validate their sequence and target values."""
    point_to_indices: dict[str, list[int]] = defaultdict(list)
    point_to_sequence: dict[str, str] = {}
    target_values: dict[str, list[float]] = defaultdict(list)

    for row_index, row in enumerate(rows):
        point_id = str(row.get(point_id_col, "")).strip()
        if not point_id:
            raise ValueError(
                f"Row {row_index} is missing a point identifier in column "
                f"'{point_id_col}'."
            )

        sequence = core.clean_sequence(row.get(sequence_col))
        if not sequence:
            raise ValueError(
                f"Row {row_index} is missing a valid sequence in column "
                f"'{sequence_col}'."
            )

        target = standardization.parse_finite_float(row.get(stratify_col))
        if target is None:
            raise ValueError(
                f"Row {row_index} has a missing or non-finite stratification target "
                f"in column '{stratify_col}'."
            )

        existing_sequence = point_to_sequence.get(point_id)
        if existing_sequence is not None and existing_sequence != sequence:
            raise ValueError(
                f"Point '{point_id}' maps to multiple protein sequences. "
                "Each point identifier must describe one protein."
            )

        point_to_indices[point_id].append(row_index)
        point_to_sequence[point_id] = sequence
        target_values[point_id].append(target)

    point_to_target = {
        point_id: sum(values) / len(values)
        for point_id, values in target_values.items()
    }
    return dict(point_to_indices), point_to_sequence, point_to_target


def assign_continuous_stratified_points(
    point_ids: list[str],
    *,
    point_weights: dict[str, int],
    point_targets: dict[str, float],
    fractions: dict[str, float],
    seed: int,
    stratify_bins: int,
) -> dict[str, str]:
    """Assign continuous-target points through deterministic rank-quantile bins."""
    positive_splits = [name for name in core.SPLIT_NAMES if fractions[name] > 0]
    if len(point_ids) < len(positive_splits):
        raise ValueError(
            "Not enough distinct points to populate every requested split. "
            f"Points={len(point_ids)}, positive splits={len(positive_splits)}."
        )
    if stratify_bins < 1:
        raise ValueError("stratify_bins must be at least 1.")

    smallest_fraction = min(fractions[name] for name in positive_splits)
    minimum_bin_size = max(
        len(positive_splits),
        math.ceil(1.0 / smallest_fraction),
    )
    bin_count = max(1, min(stratify_bins, len(point_ids) // minimum_bin_size))
    ordered_points = sorted(point_ids, key=lambda item: (point_targets[item], item))
    bins: list[list[str]] = [[] for _ in range(bin_count)]
    for rank, point_id in enumerate(ordered_points):
        bin_index = min(bin_count - 1, rank * bin_count // len(ordered_points))
        bins[bin_index].append(point_id)

    assignments: dict[str, str] = {}
    for bin_index, bin_points in enumerate(bins):
        assignments.update(
            core.assign_items_to_splits(
                {point_id: point_weights[point_id] for point_id in bin_points},
                fractions,
                seed=seed + bin_index,
            )
        )
    return assignments


def generate_point_stratified_split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    point_id_col: str,
    stratify_col: str,
    seed: int = 42,
    stratify_bins: int = 10,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Split points by target, within protein whenever enough points exist."""
    fractions = core.validate_split_fractions(
        train_fraction, val_fraction, test_fraction
    )
    point_to_indices, point_to_sequence, point_targets = build_point_index(
        rows,
        sequence_col=sequence_col,
        point_id_col=point_id_col,
        stratify_col=stratify_col,
    )
    point_weights = {
        point_id: len(indices) for point_id, indices in point_to_indices.items()
    }
    sequence_to_points: dict[str, list[str]] = defaultdict(list)
    for point_id, sequence in point_to_sequence.items():
        sequence_to_points[sequence].append(point_id)

    positive_split_count = sum(fractions[name] > 0 for name in core.SPLIT_NAMES)
    assignments: dict[str, str] = {}
    pooled_points: list[str] = []
    for sequence_index, sequence in enumerate(sorted(sequence_to_points)):
        sequence_points = sequence_to_points[sequence]
        if len(sequence_points) < positive_split_count:
            pooled_points.extend(sequence_points)
            continue
        assignments.update(
            assign_continuous_stratified_points(
                sequence_points,
                point_weights=point_weights,
                point_targets=point_targets,
                fractions=fractions,
                seed=seed + (sequence_index + 1) * 10_000,
                stratify_bins=stratify_bins,
            )
        )

    if pooled_points:
        if len(pooled_points) < positive_split_count:
            current_weights = {name: 0 for name in core.SPLIT_NAMES}
            for point_id, split_name in assignments.items():
                current_weights[split_name] += point_weights[point_id]
            target_weights = {
                name: len(rows) * fractions[name] for name in core.SPLIT_NAMES
            }
            for point_id in sorted(pooled_points):
                split_name = min(
                    (name for name in core.SPLIT_NAMES if fractions[name] > 0),
                    key=lambda name: (
                        abs(
                            current_weights[name]
                            + point_weights[point_id]
                            - target_weights[name]
                        ),
                        name,
                    ),
                )
                assignments[point_id] = split_name
                current_weights[split_name] += point_weights[point_id]
        else:
            assignments.update(
                assign_continuous_stratified_points(
                    pooled_points,
                    point_weights=point_weights,
                    point_targets=point_targets,
                    fractions=fractions,
                    seed=seed,
                    stratify_bins=stratify_bins,
                )
            )

    output_fieldnames, split_rows = core.apply_group_assignments(
        rows,
        fieldnames,
        point_to_indices,
        assignments,
    )
    return (
        output_fieldnames,
        split_rows,
        core.summarize_split_rows(split_rows, sequence_col),
    )
