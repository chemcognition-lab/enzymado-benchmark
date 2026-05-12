from __future__ import annotations
"""Composable dataset builders that connect adapters, features, and labels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:
    class TorchDataset:  # type: ignore[no-redef]
        pass

from .adapters import DatasetAdapter
from .collators import (
    MutationDeltaGlobalCollator,
    ProteinGlobalCollator,
    ProteinLigandGlobalCollator,
    ProteinResidueCollator,
)
from .features import (
    MutationDeltaEmbeddingBuilder,
    ProteinResidueEmbeddingBuilder,
    RDKitLigandFeatureBuilder,
    ResidueEmbeddingStore,
)
from .labels import (
    BinaryLabelBuilder,
    HierarchicalECLabelBuilder,
    ResidueBinaryVectorLabelBuilder,
    RegressionLabelBuilder,
)
from .registry import TaskConfig, get_task_config


class BaseComposableDataset(TorchDataset):
    """Base dataset that validates rows before exposing training samples."""
    def __init__(
        self,
        rows: list[dict[str, str]],
        label_builder: Any,
        missing_policy: str = "raise",
    ) -> None:
        self.label_builder = label_builder
        self.missing_policy = missing_policy
        self.skipped_rows = 0
        self.rows = self._apply_missing_policy(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def _feature_builders(self) -> tuple[Any, ...]:
        """Return builders that contribute feature validation for each row."""
        raise NotImplementedError

    def _build_sample(self, row: dict[str, str]) -> dict[str, Any]:
        """Convert one normalized row into the sample payload used by a collator."""
        raise NotImplementedError

    def _validate_row(self, row: dict[str, str]) -> list[str]:
        """Run each builder's validator and collect all reported issues."""
        errors: list[str] = []
        for builder in (*self._feature_builders(), self.label_builder):
            validator = getattr(builder, "validate_row", None)
            if validator is None:
                continue
            error = validator(row)
            if error:
                errors.append(error)
        return errors

    def _apply_missing_policy(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Either keep, skip, or reject rows with missing features or labels."""
        if self.missing_policy not in {"raise", "skip"}:
            raise ValueError("missing_policy must be either 'raise' or 'skip'.")

        kept_rows: list[dict[str, str]] = []
        validation_errors: list[str] = []
        for row in rows:
            errors = self._validate_row(row)
            if not errors:
                kept_rows.append(row)
                continue
            if self.missing_policy == "skip":
                self.skipped_rows += 1
                continue
            validation_errors.append(
                f"sample_id={row.get('sample_id', '')}: " + "; ".join(errors)
            )

        if validation_errors:
            preview = "\n".join(validation_errors[:5])
            raise ValueError(f"Found invalid rows while building dataset:\n{preview}")
        return kept_rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Build one sample on demand."""
        return self._build_sample(self.rows[index])

    @staticmethod
    def _merge_label_payload(label_payload: Any) -> dict[str, Any]:
        """Normalize label payloads to the dict shape expected by collators."""
        if isinstance(label_payload, dict) and "target" in label_payload:
            merged_payload = dict(label_payload)
            return merged_payload
        return {"target": label_payload}


class ProteinGlobalDataset(BaseComposableDataset):
    """Dataset for global protein-property prediction tasks."""
    def __init__(
        self,
        rows: list[dict[str, str]],
        protein_builder: ProteinResidueEmbeddingBuilder,
        label_builder: Any,
        missing_policy: str = "raise",
    ) -> None:
        self.protein_builder = protein_builder
        super().__init__(rows, label_builder=label_builder, missing_policy=missing_policy)

    def _feature_builders(self) -> tuple[Any, ...]:
        return (self.protein_builder,)

    def _build_sample(self, row: dict[str, str]) -> dict[str, Any]:
        """Attach the protein embedding and scalar or multilabel target."""
        sample = {
            "sample_id": row["sample_id"],
            "protein_emb": self.protein_builder.build(row),
        }
        sample.update(self._merge_label_payload(self.label_builder.build(row)))
        return sample


class ProteinLigandGlobalDataset(BaseComposableDataset):
    """Dataset for tasks that combine protein embeddings with ligand features."""
    def __init__(
        self,
        rows: list[dict[str, str]],
        protein_builder: ProteinResidueEmbeddingBuilder,
        ligand_builder: RDKitLigandFeatureBuilder,
        label_builder: Any,
        missing_policy: str = "raise",
    ) -> None:
        self.protein_builder = protein_builder
        self.ligand_builder = ligand_builder
        super().__init__(rows, label_builder=label_builder, missing_policy=missing_policy)

    def _feature_builders(self) -> tuple[Any, ...]:
        return (self.protein_builder, self.ligand_builder)

    def _build_sample(self, row: dict[str, str]) -> dict[str, Any]:
        """Attach protein, ligand, and target payloads for one sample."""
        sample = {
            "sample_id": row["sample_id"],
            "protein_emb": self.protein_builder.build(row),
            "ligand_feat": self.ligand_builder.build(row),
        }
        sample.update(self._merge_label_payload(self.label_builder.build(row)))
        return sample


class MutationDeltaGlobalDataset(BaseComposableDataset):
    """Dataset for mutation-effect tasks using mutant-wildtype deltas."""
    def __init__(
        self,
        rows: list[dict[str, str]],
        mutation_builder: MutationDeltaEmbeddingBuilder,
        label_builder: Any,
        missing_policy: str = "raise",
    ) -> None:
        self.mutation_builder = mutation_builder
        super().__init__(rows, label_builder=label_builder, missing_policy=missing_policy)

    def _feature_builders(self) -> tuple[Any, ...]:
        return (self.mutation_builder,)

    def _build_sample(self, row: dict[str, str]) -> dict[str, Any]:
        """Attach the precomputed mutation delta embedding and target."""
        sample = {
            "sample_id": row["sample_id"],
            "protein_delta_emb": self.mutation_builder.build(row),
        }
        sample.update(self._merge_label_payload(self.label_builder.build(row)))
        return sample


class ProteinResidueDataset(BaseComposableDataset):
    """Dataset for residue-wise supervision such as active-site prediction."""
    def __init__(
        self,
        rows: list[dict[str, str]],
        protein_builder: ProteinResidueEmbeddingBuilder,
        label_builder: ResidueBinaryVectorLabelBuilder,
        missing_policy: str = "raise",
    ) -> None:
        self.protein_builder = protein_builder
        super().__init__(rows, label_builder=label_builder, missing_policy=missing_policy)

    def _feature_builders(self) -> tuple[Any, ...]:
        return (self.protein_builder,)

    def _validate_row(self, row: dict[str, str]) -> list[str]:
        """Validate row-level payloads and residue-length compatibility."""
        errors = super()._validate_row(row)
        if errors:
            return errors

        try:
            target_per_residue = self.label_builder.build(row)
            protein_length = self.protein_builder.lookup_length(row)
        except (KeyError, ValueError) as exc:
            return [str(exc)]

        if protein_length != target_per_residue.shape[0]:
            return [
                "Protein embedding and residue target length mismatch for sample "
                f"'{row.get('sample_id', '')}': embedding={protein_length}, "
                f"target={target_per_residue.shape[0]}."
            ]
        return []

    def _build_sample(self, row: dict[str, str]) -> dict[str, Any]:
        """Build a residue-level sample and enforce length alignment."""
        target_per_residue = self.label_builder.build(row)
        protein_emb = self.protein_builder.build(row)
        if protein_emb.shape[0] != target_per_residue.shape[0]:
            raise ValueError(
                "Protein embedding and residue target length mismatch for sample "
                f"'{row.get('sample_id', '')}': embedding={protein_emb.shape[0]}, "
                f"target={target_per_residue.shape[0]}."
            )
        return {
            "sample_id": row["sample_id"],
            "protein_emb": protein_emb,
            "target_per_residue": target_per_residue,
        }


@dataclass
class TaskDatasetBundle:
    """Bundle the dataset, collator, labels, and config for one task split."""
    config: TaskConfig
    rows: list[dict[str, str]]
    dataset: BaseComposableDataset
    collator: Any
    label_builder: Any


def _build_adapter(
    config: TaskConfig,
    split_name: str | None = None,
    split_variant: str = "random",
) -> DatasetAdapter:
    """Instantiate the adapter configured for the task's batch type."""
    if split_variant not in config.dataset_paths:
        available = ", ".join(sorted(config.dataset_paths)) or "<none>"
        raise KeyError(
            f"Unknown split variant '{split_variant}' for task '{config.name}'. "
            f"Available variants: {available}"
        )

    if config.batch_kind not in {
        "protein_global",
        "protein_ligand_global",
        "mutation_delta_global",
        "protein_residue",
    }:
        raise ValueError(f"Unsupported batch_kind '{config.batch_kind}'.")

    return DatasetAdapter(
        dataset_path=config.dataset_paths[split_variant],
        sample_id_candidates=config.sample_id_candidates,
        split_name=split_name,
        split_column=config.split_column,
    )


def _build_label_builder(
    config: TaskConfig,
    rows: list[dict[str, str]],
    ec_vocabulary: dict[str, tuple[str, ...]] | None = None,
) -> Any:
    """Create the label builder required by the task configuration."""
    if config.label_kind == "scalar_regression":
        return RegressionLabelBuilder(target_column=str(config.target_column))
    if config.label_kind == "binary":
        return BinaryLabelBuilder(target_column=str(config.target_column))
    if config.label_kind == "multilabel_ec":
        builder = HierarchicalECLabelBuilder(
            target_column=str(config.target_column),
            separator=config.ec_separator,
            vocabulary_by_level=ec_vocabulary,
        )
        if ec_vocabulary is None:
            builder.fit(rows)
        return builder
    if config.label_kind == "residue_binary_vector":
        return ResidueBinaryVectorLabelBuilder(
            positions_column=str(config.residue_positions_column),
            sequence_column=config.sequence_column,
        )
    raise ValueError(f"Unsupported label_kind '{config.label_kind}'.")


def build_dataset_from_config(
    config: TaskConfig,
    embedding_store: ResidueEmbeddingStore,
    split_name: str | None = None,
    split_variant: str = "random",
    missing_policy: str = "raise",
    ligand_featurizer: Any | None = None,
    ec_vocabulary: dict[str, tuple[str, ...]] | None = None,
) -> TaskDatasetBundle:
    """Build a split-specific dataset bundle from a registered task config."""
    adapter = _build_adapter(config, split_name=split_name, split_variant=split_variant)
    rows = adapter.load_rows()

    if config.label_kind == "multilabel_ec" and split_name not in (None, "train") and ec_vocabulary is None:
        raise ValueError(
            "Building non-train EC datasets requires a vocabulary fit on train rows. "
            "Pass ec_vocabulary explicitly to avoid leakage."
        )

    # Features and labels are instantiated after rows are loaded so train-derived
    # vocabularies or split-specific checks can depend on the actual payload.
    label_builder = _build_label_builder(config, rows=rows, ec_vocabulary=ec_vocabulary)
    protein_builder = ProteinResidueEmbeddingBuilder(
        embedding_store=embedding_store,
        sequence_column=config.sequence_column,
    )

    if config.batch_kind == "protein_global":
        dataset = ProteinGlobalDataset(
            rows=rows,
            protein_builder=protein_builder,
            label_builder=label_builder,
            missing_policy=missing_policy,
        )
        collator = ProteinGlobalCollator()
    elif config.batch_kind == "protein_ligand_global":
        ligand_builder = RDKitLigandFeatureBuilder(
            ligand_column=str(config.ligand_column),
            featurize_smiles=ligand_featurizer,
        )
        dataset = ProteinLigandGlobalDataset(
            rows=rows,
            protein_builder=protein_builder,
            ligand_builder=ligand_builder,
            label_builder=label_builder,
            missing_policy=missing_policy,
        )
        collator = ProteinLigandGlobalCollator()
    elif config.batch_kind == "mutation_delta_global":
        mutation_builder = MutationDeltaEmbeddingBuilder(
            embedding_store=embedding_store,
            mutation_sequence_column=config.sequence_column,
            wildtype_sequence_column=str(config.wildtype_sequence_column),
        )
        dataset = MutationDeltaGlobalDataset(
            rows=rows,
            mutation_builder=mutation_builder,
            label_builder=label_builder,
            missing_policy=missing_policy,
        )
        collator = MutationDeltaGlobalCollator()
    elif config.batch_kind == "protein_residue":
        dataset = ProteinResidueDataset(
            rows=rows,
            protein_builder=protein_builder,
            label_builder=label_builder,
            missing_policy=missing_policy,
        )
        collator = ProteinResidueCollator()
    else:
        raise ValueError(f"Unsupported batch_kind '{config.batch_kind}'.")

    return TaskDatasetBundle(
        config=config,
        rows=rows,
        dataset=dataset,
        collator=collator,
        label_builder=label_builder,
    )


def build_task_dataset(
    task_name: str,
    embedding_store: ResidueEmbeddingStore,
    split_name: str | None = None,
    split_variant: str = "random",
    missing_policy: str = "raise",
    ligand_featurizer: Any | None = None,
    ec_vocabulary: dict[str, tuple[str, ...]] | None = None,
) -> TaskDatasetBundle:
    """Convenience wrapper around `build_dataset_from_config` keyed by task name."""
    return build_dataset_from_config(
        get_task_config(task_name),
        embedding_store=embedding_store,
        split_name=split_name,
        split_variant=split_variant,
        missing_policy=missing_policy,
        ligand_featurizer=ligand_featurizer,
        ec_vocabulary=ec_vocabulary,
    )
