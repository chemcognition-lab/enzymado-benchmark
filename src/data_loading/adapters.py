from __future__ import annotations
"""Dataset adapters that normalize raw CSV rows before dataset construction."""

from dataclasses import dataclass
from pathlib import Path

from .splits import (
    clean_text,
    filter_rows_by_split,
    load_split_assignments,
    normalize_sequence,
    read_csv_rows,
)


@dataclass
class DatasetAdapter:
    """Load a task CSV and optionally align it with an external split file."""
    dataset_path: Path
    sample_id_candidates: tuple[str, ...]
    split_name: str | None = None
    split_column: str | None = None
    split_path: Path | None = None
    split_id_candidates: tuple[str, ...] = ("dataset_id", "uniprot_id", "protein_sequence")

    def load_rows(self) -> list[dict[str, str]]:
        """Read, normalize, filter, and tag rows with a stable `sample_id`."""
        _, rows = read_csv_rows(self.dataset_path)
        normalized_rows = [self._normalize_row(row) for row in rows]

        split_assignments = None
        if self.split_path is not None and self.split_name is not None:
            _, split_assignments = load_split_assignments(self.split_path)

        filtered_rows = filter_rows_by_split(
            normalized_rows,
            split_name=self.split_name,
            split_column=self.split_column,
            split_assignments=split_assignments,
            split_id_candidates=self.split_id_candidates,
        )
        return [self._attach_sample_id(row) for row in filtered_rows]

    def _normalize_row(self, row: dict[str, str]) -> dict[str, str]:
        """Normalize text fields and uppercase sequence columns."""
        normalized = {key: clean_text(value) for key, value in row.items()}
        for sequence_column in ("protein_sequence", "wildtype_sequence"):
            if sequence_column in normalized:
                normalized[sequence_column] = normalize_sequence(normalized[sequence_column])
        return normalized

    def _attach_sample_id(self, row: dict[str, str]) -> dict[str, str]:
        """Attach the resolved sample identifier used throughout training."""
        sample_id = self._resolve_sample_id(row)
        normalized = dict(row)
        normalized["sample_id"] = sample_id
        return normalized

    def _resolve_sample_id(self, row: dict[str, str]) -> str:
        """Pick the first populated identifier column configured for the task."""
        for candidate in self.sample_id_candidates:
            value = clean_text(row.get(candidate))
            if value:
                return value
        raise ValueError(
            f"Unable to resolve sample_id using candidates {self.sample_id_candidates}. "
            f"Available columns: {sorted(row)}"
        )


class ProteinGlobalAdapter(DatasetAdapter):
    """Adapter for global protein-property tasks."""
    pass


class ProteinLigandAdapter(DatasetAdapter):
    """Adapter for joint protein-ligand prediction tasks."""
    pass


class MutationDeltaAdapter(DatasetAdapter):
    """Adapter for mutation-effect tasks that compare mutant and wildtype rows."""
    pass


class ProteinResidueAdapter(DatasetAdapter):
    """Adapter for residue-level labeling tasks."""
    pass
