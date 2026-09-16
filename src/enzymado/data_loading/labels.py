"""Label builders for scalar, binary, multilabel, and residue-level tasks."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np

from .task_csv import clean_text, normalize_sequence


@dataclass
class RegressionLabelBuilder:
    """Parse numeric regression targets from dataset rows."""

    target_column: str

    def build(self, row: dict[str, str]) -> np.float32:
        """Build a single float32 regression target."""
        value = clean_text(row.get(self.target_column))
        if not value:
            raise ValueError(
                f"Missing regression target in column '{self.target_column}' "
                "for sample "
                f"'{row.get('sample_id', '')}'."
            )
        return np.float32(float(value))

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Validate that a regression target can be parsed."""
        try:
            self.build(row)
        except ValueError as exc:
            return str(exc)
        return None


@dataclass
class BinaryLabelBuilder:
    """Parse binary targets into `0.0` or `1.0` float labels."""

    target_column: str
    true_values: tuple[str, ...] = ("1", "1.0", "true", "True")
    false_values: tuple[str, ...] = ("0", "0.0", "false", "False")

    def build(self, row: dict[str, str]) -> np.float32:
        """Build a float32 binary label from string-like values."""
        value = clean_text(row.get(self.target_column))
        if value in self.true_values:
            return np.float32(1.0)
        if value in self.false_values:
            return np.float32(0.0)
        raise ValueError(
            f"Invalid binary target '{value}' in column '{self.target_column}' "
            "for sample "
            f"'{row.get('sample_id', '')}'."
        )

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Validate that a row contains a supported binary label value."""
        try:
            self.build(row)
        except ValueError as exc:
            return str(exc)
        return None


@dataclass
class HierarchicalECLabelBuilder:
    """Build non-cumulative hierarchical multi-label EC targets with masks."""

    target_column: str = "ec_number"
    separator: str = ";"
    vocabulary_by_level: dict[str, tuple[str, ...]] | None = None
    _index_by_level: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    @staticmethod
    def level_names() -> tuple[str, ...]:
        return ("level_1", "level_2", "level_3", "level_4")

    def parse_labels(self, row: dict[str, str]) -> list[str]:
        """Split semicolon-separated EC labels."""
        raw_value = clean_text(row.get(self.target_column))
        if not raw_value:
            return []
        return [
            label.strip() for label in raw_value.split(self.separator) if label.strip()
        ]

    def _parse_single_ec(self, ec_label: str) -> tuple[str, str, str, str]:
        """Parse one EC label into four levels."""
        parts = tuple(part.strip() for part in ec_label.split("."))
        if len(parts) != 4:
            raise ValueError(f"Invalid EC label '{ec_label}'. Expected 4 levels.")

        seen_missing = False
        for part in parts:
            if part == "-":
                seen_missing = True
            elif seen_missing:
                raise ValueError(
                    f"Invalid EC label '{ec_label}'. Missing levels must be trailing."
                )

        return parts  # type: ignore[return-value]

    def _positive_labels_and_unknown_prefixes(
        self,
        row: dict[str, str],
    ) -> tuple[dict[str, set[str]], list[tuple[int, str]]]:
        """Extract known positive EC labels and incomplete branches to mask."""
        positives = {level: set() for level in self.level_names()}
        unknown_prefixes: list[tuple[int, str]] = []

        for ec_label in self.parse_labels(row):
            parts = self._parse_single_ec(ec_label)
            known_parts = []

            for depth, part in enumerate(parts, start=1):
                if part == "-":
                    break

                known_parts.append(part)
                positives[f"level_{depth}"].add(".".join(known_parts))

            if "-" in parts:
                first_missing_depth = parts.index("-") + 1
                prefix_depth = first_missing_depth - 1
                prefix = ".".join(parts[:prefix_depth])
                unknown_prefixes.append((prefix_depth, prefix))

        return positives, unknown_prefixes

    def fit(self, rows: list[dict[str, str]]) -> "HierarchicalECLabelBuilder":
        """Build EC vocabulary from training rows only."""
        labels_by_level = {level: set() for level in self.level_names()}

        for row in rows:
            positives, _ = self._positive_labels_and_unknown_prefixes(row)

            for level_name in self.level_names():
                labels_by_level[level_name].update(positives[level_name])

        self.vocabulary_by_level = {
            level_name: tuple(sorted(labels))
            for level_name, labels in labels_by_level.items()
        }

        self._ensure_index_by_level()

        return self

    def _ensure_index_by_level(self) -> None:
        """Build label-to-column maps when vocabulary is supplied directly."""
        if self.vocabulary_by_level is None:
            raise ValueError("Builder must be fit before calling build().")
        if set(self._index_by_level) == set(self.vocabulary_by_level):
            return

        self._index_by_level = {
            level_name: {label: idx for idx, label in enumerate(vocab)}
            for level_name, vocab in self.vocabulary_by_level.items()
        }

    def build(self, row: dict[str, str]) -> dict[str, dict[str, np.ndarray]]:
        """Build target and mask vectors for one protein."""
        if self.vocabulary_by_level is None:
            raise ValueError("Builder must be fit before calling build().")
        self._ensure_index_by_level()

        positives, unknown_prefixes = self._positive_labels_and_unknown_prefixes(row)

        targets: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}

        has_any_ec = bool(self.parse_labels(row))

        for depth, level_name in enumerate(self.level_names(), start=1):
            vocab = self.vocabulary_by_level[level_name]
            index_by_label = self._index_by_level[level_name]

            target = np.zeros(len(vocab), dtype=np.float32)
            mask = (
                np.ones(len(vocab), dtype=bool)
                if has_any_ec
                else np.zeros(len(vocab), dtype=bool)
            )

            # Mask unknown descendants from incomplete EC labels.
            for prefix_depth, prefix in unknown_prefixes:
                if depth <= prefix_depth:
                    continue

                for idx, label in enumerate(vocab):
                    if not prefix or label.startswith(prefix + "."):
                        mask[idx] = False

            # Mark known positives; positives override masks.
            for label in positives[level_name]:
                if label in index_by_label:
                    idx = index_by_label[label]
                    target[idx] = 1.0
                    mask[idx] = True

            targets[level_name] = target
            masks[level_name] = mask

        return {
            "target": targets,
            "target_mask": masks,
        }

    def validate_row(self, row: dict[str, str]) -> str | None:
        """Validate EC format."""
        try:
            for ec_label in self.parse_labels(row):
                self._parse_single_ec(ec_label)
        except ValueError as exc:
            return str(exc)

        return None


@dataclass
class FlatECLabelBuilder(HierarchicalECLabelBuilder):
    """Flatten hierarchical EC targets into one concatenated multilabel vector."""

    flat_vocabulary: tuple[str, ...] | None = None
    _offset_by_level: dict[str, int] = field(default_factory=dict, init=False)

    def fit(self, rows: list[dict[str, str]]) -> "FlatECLabelBuilder":
        """Fit hierarchy vocabularies and derive the flattened index."""
        super().fit(rows)
        self._ensure_flat_index()
        return self

    def _ensure_flat_index(self) -> None:
        """Build the flattened vocabulary and level offsets on first use."""
        if self.vocabulary_by_level is None:
            raise ValueError("FlatECLabelBuilder must be fit before calling build().")
        if not self._index_by_level:
            self._index_by_level = {
                level_name: {label: idx for idx, label in enumerate(vocabulary)}
                for level_name, vocabulary in self.vocabulary_by_level.items()
            }
        if self.flat_vocabulary is not None and self._offset_by_level:
            return

        flat_labels: list[str] = []
        offset = 0
        for level_name in self.level_names():
            self._offset_by_level[level_name] = offset
            vocabulary = tuple(self.vocabulary_by_level[level_name])
            flat_labels.extend(vocabulary)
            offset += len(vocabulary)
        self.flat_vocabulary = tuple(flat_labels)

    def flat_target_width(self) -> int:
        """Width of one flattened target, built on demand.

        Callers that must size an empty batch need this before any row is built,
        because ``build`` is what would otherwise trigger the lazy index.
        """
        self._ensure_flat_index()
        return len(self.flat_vocabulary or ())

    def build(self, row: dict[str, str]) -> dict[str, np.ndarray]:
        """Concatenate hierarchical levels into one flat EC target."""
        self._ensure_flat_index()
        payload = super().build(row)
        level_names = self.level_names()

        return {
            "target": np.concatenate(
                [
                    np.asarray(payload["target"][level_name], dtype=np.float32)
                    for level_name in level_names
                ]
            ),
            "target_mask": np.concatenate(
                [
                    np.asarray(payload["target_mask"][level_name], dtype=bool)
                    for level_name in level_names
                ]
            ),
        }


def _parse_position_list(raw_value: str) -> list[int]:
    """Parse residue position payloads stored as Python literals."""
    cleaned = clean_text(raw_value)
    if not cleaned:
        return []

    parsed = ast.literal_eval(cleaned)
    if isinstance(parsed, int):
        positions = [parsed]
    elif isinstance(parsed, (list, tuple, set)):
        positions = [int(value) for value in parsed]
    else:
        raise ValueError(f"Unsupported residue position payload: {parsed!r}")
    return positions


@dataclass
class ResidueBinaryVectorLabelBuilder:
    """Build residue-wise binary vectors for active- and binding-site tasks."""

    positions_column: str
    sequence_column: str = "protein_sequence"

    def build(self, row: dict[str, str]) -> np.ndarray:
        """Build residue targets for one row."""
        sequence = normalize_sequence(row.get(self.sequence_column))
        if not sequence:
            raise ValueError(
                f"Missing sequence in column '{self.sequence_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        positions = _parse_position_list(clean_text(row.get(self.positions_column)))
        target = np.zeros(len(sequence), dtype=np.float32)

        for position in positions:
            index = position - 1
            if index < 0 or index >= len(sequence):
                raise ValueError(
                    f"Residue position {position} out of bounds for sample "
                    f"'{row.get('sample_id', '')}' with length {len(sequence)}."
                )
            target[index] = 1.0
        return target

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.build(row)
        except (SyntaxError, ValueError) as exc:
            return str(exc)
        return None
