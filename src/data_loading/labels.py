from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np

from .splits import clean_text, normalize_sequence


@dataclass
class StandardScaler1D:
    mean_: float | None = None
    scale_: float | None = None

    def fit(self, values: list[float] | np.ndarray) -> "StandardScaler1D":
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ValueError("Cannot fit StandardScaler1D on an empty array.")
        self.mean_ = float(array.mean())
        scale = float(array.std())
        self.scale_ = 1.0 if scale == 0.0 else scale
        return self

    def transform(self, values: list[float] | np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler1D must be fit before calling transform().")
        array = np.asarray(values, dtype=np.float32)
        return (array - self.mean_) / self.scale_

    def inverse_transform(self, values: list[float] | np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler1D must be fit before calling inverse_transform().")
        array = np.asarray(values, dtype=np.float32)
        return (array * self.scale_) + self.mean_

    @classmethod
    def from_rows(cls, rows: list[dict[str, str]], target_column: str) -> "StandardScaler1D":
        values = [float(clean_text(row.get(target_column))) for row in rows]
        return cls().fit(values)


@dataclass
class ScalarRegressionLabelBuilder:
    target_column: str

    def build(self, row: dict[str, str]) -> np.float32:
        value = clean_text(row.get(self.target_column))
        if not value:
            raise ValueError(
                f"Missing regression target in column '{self.target_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        return np.float32(float(value))

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.build(row)
        except ValueError as exc:
            return str(exc)
        return None


@dataclass
class BinaryLabelBuilder:
    target_column: str
    true_values: tuple[str, ...] = ("1", "1.0", "true", "True")
    false_values: tuple[str, ...] = ("0", "0.0", "false", "False")

    def build(self, row: dict[str, str]) -> np.float32:
        value = clean_text(row.get(self.target_column))
        if value in self.true_values:
            return np.float32(1.0)
        if value in self.false_values:
            return np.float32(0.0)
        raise ValueError(
            f"Invalid binary target '{value}' in column '{self.target_column}' for sample "
            f"'{row.get('sample_id', '')}'."
        )

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.build(row)
        except ValueError as exc:
            return str(exc)
        return None


@dataclass
class HierarchicalECLabelBuilder:
    target_column: str = "ec_number"
    separator: str = ";"
    vocabulary_by_level: dict[str, tuple[str, ...]] | None = None
    _index_by_level: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    def fit(self, rows: list[dict[str, str]]) -> "HierarchicalECLabelBuilder":
        labels_by_level = {
            level_name: sorted(
                {
                    label
                    for row in rows
                    for label in self.extract_level_labels(row)[level_name]
                }
            )
            for level_name in self.level_names()
        }
        self.vocabulary_by_level = {
            level_name: tuple(labels)
            for level_name, labels in labels_by_level.items()
        }
        self._index_by_level = {
            level_name: {label: idx for idx, label in enumerate(vocabulary)}
            for level_name, vocabulary in self.vocabulary_by_level.items()
        }
        return self

    def parse_labels(self, row: dict[str, str]) -> list[str]:
        raw_value = clean_text(row.get(self.target_column))
        if not raw_value:
            return []
        return [label.strip() for label in raw_value.split(self.separator) if label.strip()]

    @staticmethod
    def level_names() -> tuple[str, ...]:
        return ("level_1", "level_2", "level_3", "level_4")

    def _parse_single_ec(self, ec_label: str) -> tuple[str, str, str, str]:
        parts = [part.strip() for part in ec_label.split(".")]
        if len(parts) != 4:
            raise ValueError(f"Invalid EC label '{ec_label}'. Expected four levels.")
        return tuple(parts)  # type: ignore[return-value]

    def extract_level_labels(self, row: dict[str, str]) -> dict[str, set[str]]:
        labels_by_level = {level_name: set() for level_name in self.level_names()}
        for ec_label in self.parse_labels(row):
            level_1, level_2, level_3, level_4 = self._parse_single_ec(ec_label)
            if level_1 != "-":
                labels_by_level["level_1"].add(level_1)
            if level_1 != "-" and level_2 != "-":
                labels_by_level["level_2"].add(f"{level_1}.{level_2}")
            if level_1 != "-" and level_2 != "-" and level_3 != "-":
                labels_by_level["level_3"].add(f"{level_1}.{level_2}.{level_3}")
            if level_1 != "-" and level_2 != "-" and level_3 != "-" and level_4 != "-":
                labels_by_level["level_4"].add(f"{level_1}.{level_2}.{level_3}.{level_4}")
        return labels_by_level

    def build(self, row: dict[str, str]) -> dict[str, dict[str, np.ndarray]]:
        if self.vocabulary_by_level is None:
            raise ValueError("HierarchicalECLabelBuilder must be fit before calling build().")
        if not self._index_by_level:
            self._index_by_level = {
                level_name: {label: idx for idx, label in enumerate(vocabulary)}
                for level_name, vocabulary in self.vocabulary_by_level.items()
            }

        row_level_labels = self.extract_level_labels(row)
        targets: dict[str, np.ndarray] = {}
        target_masks: dict[str, np.ndarray] = {}
        for level_name in self.level_names():
            vocabulary = self.vocabulary_by_level[level_name]
            index_by_label = self._index_by_level[level_name]
            target = np.zeros(len(vocabulary), dtype=np.float32)
            known_active_labels = [
                label
                for label in row_level_labels[level_name]
                if label in index_by_label
            ]
            for label in known_active_labels:
                target[index_by_label[label]] = 1.0
            targets[level_name] = target
            target_masks[level_name] = np.asarray(bool(known_active_labels), dtype=bool)
        return {
            "target": targets,
            "target_mask": target_masks,
        }

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.extract_level_labels(row)
        except ValueError as exc:
            return str(exc)
        return None


class MultiLabelECBuilder(HierarchicalECLabelBuilder):
    pass


def _parse_position_list(raw_value: str) -> list[int]:
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
    positions_column: str
    sequence_column: str = "protein_sequence"

    def build(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        sequence = normalize_sequence(row.get(self.sequence_column))
        if not sequence:
            raise ValueError(
                f"Missing sequence in column '{self.sequence_column}' for sample "
                f"'{row.get('sample_id', '')}'."
            )
        positions = _parse_position_list(clean_text(row.get(self.positions_column)))
        target = np.zeros(len(sequence), dtype=np.float32)
        target_mask = np.ones(len(sequence), dtype=bool)

        for position in positions:
            index = position - 1
            if index < 0 or index >= len(sequence):
                raise ValueError(
                    f"Residue position {position} out of bounds for sample "
                    f"'{row.get('sample_id', '')}' with length {len(sequence)}."
                )
            target[index] = 1.0
        return target, target_mask

    def validate_row(self, row: dict[str, str]) -> str | None:
        try:
            self.build(row)
        except (SyntaxError, ValueError) as exc:
            return str(exc)
        return None
