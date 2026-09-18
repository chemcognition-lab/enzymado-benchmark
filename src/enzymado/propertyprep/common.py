"""Shared cleaning-statistics helpers for property preparation."""

def start_stats(total: int) -> dict[str, object]:
    """Create the legacy cleaning-statistics record in stable key order."""
    return {
        "datapoints_before_cleaning": total,
        "removed_by_reason": {
            "missing_values": 0,
            "duplicates": 0,
            "invalid_sequence": 0,
            "was_not_enzyme": 0,
            "other": 0,
        },
    }


def finish_stats(stats: dict[str, object], remaining: int) -> dict[str, object]:
    """Complete and wrap a legacy cleaning-statistics record."""
    stats["datapoints_after_cleaning"] = remaining
    removed = int(stats["datapoints_before_cleaning"]) - remaining
    stats["datapoints_removed_total"] = removed
    before = int(stats["datapoints_before_cleaning"])
    stats["removed_percentage"] = removed / before * 100 if before else 0
    return {"amount_of_removed_data": stats}


def merge_residue_annotations_by_sequence(
    frame,
    *,
    sequence_column: str,
    position_column: str,
    id_column: str,
    stats: dict[str, object] | None = None,
):
    """Collapse rows sharing a protein sequence into one union-annotated row.

    UniProt ships one entry per accession, so a sequence carried by several
    accessions arrives as several rows. Their residue annotations are usually
    identical, but sometimes one accession is annotated more completely than
    another. Keeping both rows hands the model the same input with two
    different targets, and every residue missing from the shorter annotation is
    scored as a negative, penalising a model that predicts it correctly.
    Collapsing to the union of positions keeps every annotated residue and
    leaves exactly one row per distinct model input.

    Returns the collapsed frame plus the number of rows removed and the number
    of sequences whose annotations actually disagreed.
    """
    if frame.empty:
        return frame, 0, 0

    ordered = frame.sort_values(id_column, kind="stable")
    merged_rows = []
    conflicting_sequences = 0

    for _, group in ordered.groupby(sequence_column, sort=False):
        row = group.iloc[0].copy()
        if len(group) > 1:
            position_sets = {tuple(sorted(set(v))) for v in group[position_column]}
            if len(position_sets) > 1:
                conflicting_sequences += 1
            union: set[int] = set()
            for positions in group[position_column]:
                union.update(int(p) for p in positions)
            row[position_column] = sorted(union)
        merged_rows.append(row)

    collapsed = type(frame)(merged_rows).reset_index(drop=True)
    removed = len(frame) - len(collapsed)
    if stats is not None:
        reasons = stats["removed_by_reason"]
        reasons["duplicates"] = int(reasons["duplicates"]) + removed
    return collapsed, removed, conflicting_sequences
