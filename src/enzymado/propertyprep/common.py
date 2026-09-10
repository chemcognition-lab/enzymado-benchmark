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
