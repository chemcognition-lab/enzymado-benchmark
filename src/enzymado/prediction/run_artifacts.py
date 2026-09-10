"""Persist predictions and trained models for benchmark runs."""

from __future__ import annotations

import gc
from pathlib import Path
import typing

import enzymado.artifacts.results_layout
import enzymado.artifacts.sample_identity
import enzymado.artifacts.serialization
import enzymado.data_loading.datasets
import enzymado.data_loading.stores
import enzymado.data_loading.task_data
import enzymado.training.residue

from . import predict


def _bundle_sample_ids(
    bundle: enzymado.data_loading.datasets.TaskDatasetBundle,
) -> tuple[str, ...]:
    """Return identifiers in the exact order exposed by one dataset bundle."""
    rows = typing.cast(
        list[dict[str, str]] | None,
        getattr(bundle.dataset, "rows", None),
    )
    if rows is None:
        raise TypeError("Dataset does not expose validated rows for sample IDs.")
    identifiers = []
    for row in rows:
        identifier = row.get("prediction_id") or row.get("sample_id", "")
        identifiers.append(identifier)
    return enzymado.artifacts.sample_identity.normalize(identifiers)


def _sample_ids(
    prepared: enzymado.data_loading.task_data.PooledTaskData
    | enzymado.data_loading.task_data.LoaderTaskData,
) -> dict[str, tuple[str, ...]]:
    """Collect ordered identifiers for every prepared prediction split."""
    if isinstance(prepared, enzymado.data_loading.task_data.PooledTaskData):
        return {
            "train": prepared.sample_ids_train,
            "val": prepared.sample_ids_val,
            "test": prepared.sample_ids_test,
        }
    return {
        "train": _bundle_sample_ids(prepared.train_bundle),
        "val": (
            _bundle_sample_ids(prepared.val_bundle)
            if prepared.val_bundle is not None
            else ()
        ),
        "test": _bundle_sample_ids(prepared.test_bundle),
    }


def build_sample_manifest(
    prepared: enzymado.data_loading.task_data.PooledTaskData
    | enzymado.data_loading.task_data.LoaderTaskData,
) -> dict[str, dict[str, int | str]]:
    """Build compact ordered-ID provenance for hparams.json."""
    manifest = {}
    for split_name, sample_ids in _sample_ids(prepared).items():
        manifest[split_name] = enzymado.artifacts.sample_identity.manifest(sample_ids)
    return manifest


def save_split_predictions(
    outputs: predict.ModelRunOutputs,
    *,
    prepared: enzymado.data_loading.task_data.PooledTaskData
    | enzymado.data_loading.task_data.LoaderTaskData,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
    use_validation: bool,
) -> dict[str, Path]:
    """Save raw predictions for each enabled split."""
    sample_ids_by_split = _sample_ids(prepared)
    split_payloads: list[tuple[str, object, object, object]] = [
        (
            "train",
            outputs.y_train,
            outputs.pred_train,
            outputs.target_mask_train,
        )
    ]
    if use_validation:
        split_payloads.append(
            ("val", outputs.y_val, outputs.pred_val, outputs.target_mask_val)
        )
    split_payloads.append(
        ("test", outputs.y_test, outputs.pred_test, outputs.target_mask_test)
    )

    saved_paths: dict[str, Path] = {}
    for split_name, y_true, pred, target_mask in split_payloads:
        saved_paths[split_name] = enzymado.artifacts.serialization.save_raw_predictions(
            y_true,
            pred,
            target_mask,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name=split_name,
            sample_ids=sample_ids_by_split[split_name],
        )
    return saved_paths


def save_residue_predictions(
    trained_payload: enzymado.training.residue.ResidueTrainingPayload,
    *,
    prepared: enzymado.data_loading.task_data.LoaderTaskData,
    use_validation: bool,
    batch_kind: str,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
) -> dict[str, Path]:
    """Generate and save residue predictions one split at a time."""
    split_jobs = [
        ("train", prepared.train_bundle, trained_payload.loaders.train_predict_loader),
        ("test", prepared.test_bundle, trained_payload.loaders.test_loader),
    ]
    if use_validation:
        split_jobs.insert(
            1,
            ("val", prepared.val_bundle, trained_payload.loaders.val_loader),
        )

    saved_paths: dict[str, Path] = {}
    for split_name, bundle, loader in split_jobs:
        enzymado.data_loading.stores.log(
            f"      Predicting/saving split='{split_name}'..."
        )
        y_true, target_mask = enzymado.data_loading.stores.get_target_payload(bundle)
        pred = trained_payload.trained.predict_loader(
            loader,
            batch_kind=batch_kind,
        )
        saved_paths[split_name] = enzymado.artifacts.serialization.save_raw_predictions(
            y_true,
            pred,
            target_mask,
            run_dir=run_dir,
            task_name=task_name,
            emb_name=emb_name,
            model_name=model_name,
            split_variant=split_variant,
            split_name=split_name,
            sample_ids=_bundle_sample_ids(bundle),
        )
        del y_true, target_mask, pred
        gc.collect()
    return saved_paths


def save_trained_model(
    trained: object,
    *,
    route_name: str,
    run_dir: Path,
    task_name: str,
    emb_name: str,
    model_name: str,
    split_variant: str,
) -> None:
    """Save one trained estimator using the existing route-specific formats."""
    canonical_model_path = enzymado.artifacts.results_layout.build_model_artifact_path(
        run_dir,
        task_name=task_name,
        emb_name=emb_name,
        model_name=model_name,
        split_variant=split_variant,
        extension=".safetensors" if route_name == "attention" else ".pt",
    )
    if route_name == "attention":
        enzymado.artifacts.serialization.save_model_safetensors(
            trained,
            run_dir,
            model_path=canonical_model_path,
        )
        return

    enzymado.artifacts.serialization.save_model(trained, run_dir)
    safetensors_path = enzymado.artifacts.results_layout.build_model_artifact_path(
        run_dir,
        task_name=task_name,
        emb_name=emb_name,
        model_name=model_name,
        split_variant=split_variant,
        extension=".safetensors",
    )
    enzymado.artifacts.serialization.save_model_safetensors(
        trained,
        run_dir,
        model_path=safetensors_path,
    )
