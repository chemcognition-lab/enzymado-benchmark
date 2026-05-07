"""Shared task-row loading helpers for training pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from training.train_common import filter_rows_to_allowed_ids, log
from training.train_data import collect_needed_lookup_keys, get_task_rows


@dataclass
class LoadedTaskRows:
    """CSV rows plus the lookup keys needed to load matching embeddings."""

    train_rows: list[dict]
    val_rows: list[dict]
    test_rows: list[dict]
    needed_ids: set[str]
    needed_sequences: set[str]


def load_task_rows_bundle(
    config,
    split_variant: str,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    *,
    log_loading: bool = True,
) -> LoadedTaskRows:
    """Load task rows, optionally filter by allowed IDs, and collect lookup keys."""
    if log_loading:
        log("    Loading CSV rows...")

    train_rows = get_task_rows(config, "train", split_variant)
    val_rows = get_task_rows(config, "val", split_variant)
    test_rows = get_task_rows(config, "test", split_variant)

    if allowed_ids is not None:
        orig_train_rows = len(train_rows)
        orig_val_rows = len(val_rows)
        orig_test_rows = len(test_rows)
        train_rows = filter_rows_to_allowed_ids(train_rows, config, allowed_ids, sequence_to_id)
        val_rows = filter_rows_to_allowed_ids(val_rows, config, allowed_ids, sequence_to_id)
        test_rows = filter_rows_to_allowed_ids(test_rows, config, allowed_ids, sequence_to_id)
        log(
            "    Applied allowed-id filter:"
            f" train={len(train_rows):,}/{orig_train_rows:,}"
            f" val={len(val_rows):,}/{orig_val_rows:,}"
            f" test={len(test_rows):,}/{orig_test_rows:,}"
        )

    needed_ids, needed_sequences = collect_needed_lookup_keys(train_rows + val_rows + test_rows, config)
    log(
        f"    Rows loaded: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} "
        f"| need {len(needed_ids)} ids, {len(needed_sequences)} seqs"
    )
    return LoadedTaskRows(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        needed_ids=needed_ids,
        needed_sequences=needed_sequences,
    )


def log_loaded_data_summary(task_rows: LoadedTaskRows, n_train: int, n_val: int, n_test: int) -> None:
    """Log how many rows survived dataset construction."""
    orig_train = len(task_rows.train_rows)
    orig_val = len(task_rows.val_rows)
    orig_test = len(task_rows.test_rows)
    orig_total = orig_train + orig_val + orig_test
    kept_total = n_train + n_val + n_test
    dropped = orig_total - kept_total
    dropped_pct = (100 * dropped / orig_total) if orig_total > 0 else 0.0

    log(
        f"    Data summary:"
        f"\n      original : train={orig_train:,}  val={orig_val:,}  test={orig_test:,}  total={orig_total:,}"
        f"\n      kept     : train={n_train:,}  val={n_val:,}  test={n_test:,}  total={kept_total:,}"
        f"\n      dropped  : {dropped:,} ({dropped_pct:.1f}% of original)"
        + (f"  ← missing embeddings/ligand features" if dropped > 0 else "")
    )


__all__ = [
    "LoadedTaskRows",
    "load_task_rows_bundle",
    "log_loaded_data_summary",
]
