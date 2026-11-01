"""Execute and persist one selected benchmark model run."""

from __future__ import annotations

import collections.abc
import dataclasses
import json
from pathlib import Path
import shutil
import typing

from enzymado import types
from enzymado.artifacts import reproducibility, results_layout, serialization
from enzymado.data_loading import data_registry, stores, task_data
from enzymado.models import common
from enzymado.prediction import predict, restore, run_artifacts
from enzymado.training import residue, train

from . import run_config, run_data


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Task-level state shared by each selected model run."""

    output_dir: Path
    task_scope: str
    task_name: str
    emb_name: str
    split_variant: str
    config: data_registry.TaskConfig
    prepared: task_data.PooledTaskData | task_data.LoaderTaskData
    route: train.TrainingRouteSpec
    sizes: run_data.TaskSizes
    overwrite_results: bool
    random_seed: int
    dataset_path: Path
    embedding_path: Path
    deterministic_algorithms: bool = False
    attention_overrides: collections.abc.Mapping[str, object] | None = None
    attach_probabilities: bool = False


def _save_attention_predictions(
    context: RunContext,
    model_info: types.ModelInfo,
    trained_payload: object,
    save_kwargs: dict[str, object],
) -> predict.SavedModelRun:
    """Generate and persist predictions from one attention model."""
    prepared = typing.cast(task_data.LoaderTaskData, context.prepared)
    stores.log("      Predicting with attention model...")
    batch_size = int(model_info.get("loader_batch_size", 16))
    num_workers = int(model_info.get("loader_num_workers", 4))
    train_loader = stores.get_dataloader(
        prepared.train_bundle,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = (
        stores.get_dataloader(
            prepared.val_bundle,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if context.sizes.use_validation and prepared.val_bundle is not None
        else None
    )
    test_loader = stores.get_dataloader(
        prepared.test_bundle,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    outputs = predict.predict_from_loaders(
        trained_payload,
        train_loader,
        val_loader,
        test_loader,
        prepared,
        context.sizes.use_validation,
        context.config.batch_kind,
    )
    prediction_paths = run_artifacts.save_split_predictions(
        outputs,
        prepared=prepared,
        **save_kwargs,
    )
    return predict.SavedModelRun(
        trained=outputs.trained,
        prediction_paths=prediction_paths,
    )


def _train_and_predict(
    context: RunContext,
    model_name: str,
    model_info: types.ModelInfo,
    run_dir: Path,
) -> predict.SavedModelRun:
    """Train one selected model and persist its raw split predictions."""
    trained_payload = context.route.train(
        model_name=model_name,
        emb_name=context.emb_name,
        model_info=model_info,
        data=context.prepared,
        config=context.config,
        use_validation=context.sizes.use_validation,
        run_dir=run_dir,
    )
    save_kwargs = {
        "run_dir": run_dir,
        "task_name": context.task_name,
        "emb_name": context.emb_name,
        "model_name": model_name,
        "split_variant": context.split_variant,
        "use_validation": context.sizes.use_validation,
    }

    if context.route.name == "residue":
        residue_payload = typing.cast(residue.ResidueTrainingPayload, trained_payload)
        prepared = typing.cast(task_data.LoaderTaskData, context.prepared)
        stores.log("      Predicting and saving residue outputs split-by-split...")
        prediction_paths = run_artifacts.save_residue_predictions(
            residue_payload,
            prepared=prepared,
            batch_kind=context.config.batch_kind,
            **save_kwargs,
        )
        trained = residue_payload.trained
    elif context.route.name == "attention":
        return _save_attention_predictions(
            context,
            model_info,
            trained_payload,
            save_kwargs,
        )
    else:
        prepared = typing.cast(task_data.PooledTaskData, context.prepared)
        stores.log("      Predicting with pooled model...")
        arrays = train._select_global_arrays(model_info, prepared)
        outputs = predict.predict_from_arrays(
            trained_payload,
            arrays.x_train,
            arrays.y_train,
            arrays.x_val,
            arrays.y_val,
            arrays.x_test,
            arrays.y_test,
            context.sizes.use_validation,
            arrays.mask_train,
            arrays.mask_val,
            arrays.mask_test,
        )
        prediction_paths = run_artifacts.save_split_predictions(
            outputs,
            prepared=prepared,
            **save_kwargs,
        )
        trained = outputs.trained

    return predict.SavedModelRun(trained=trained, prediction_paths=prediction_paths)


def _attach_run_probabilities(
    context: RunContext,
    model_name: str,
    model_info: types.ModelInfo,
    run_dir: Path,
) -> None:
    """Add test-split probabilities to one published classification run.

    Restores the fitted wrapper the run already stored, so nothing is retrained
    and the published hard labels are copied through untouched.
    """
    artifacts = sorted(run_dir.glob("*_test_predictions.safetensors.xz"))
    if not artifacts:
        stores.log(f"    [{model_name}] No test predictions — skipping.")
        return
    wrapper = restore.restore_fitted_wrapper(run_dir, device=common.DEVICE)
    if not hasattr(wrapper, "predict_proba_loader") and not hasattr(
        wrapper, "predict_proba"
    ):
        stores.log(
            f"    [{model_name}] {type(wrapper).__name__} exposes no probability "
            "API — skipping."
        )
        return

    if context.route.name in {"residue", "attention"}:
        prepared = typing.cast(task_data.LoaderTaskData, context.prepared)
        default_batch = 16 if context.route.name == "attention" else 64
        loader = stores.get_dataloader(
            prepared.test_bundle,
            batch_size=int(model_info.get("loader_batch_size", default_batch)),
            num_workers=int(model_info.get("loader_num_workers", 4)),
        )
        proba = restore.predict_probabilities(
            wrapper,
            dataloader=loader,
            batch_kind=context.config.batch_kind,
        )
    else:
        prepared = typing.cast(task_data.PooledTaskData, context.prepared)
        arrays = train._select_global_arrays(model_info, prepared)
        proba = restore.predict_probabilities(wrapper, features=arrays.x_test)

    restore.attach_probabilities(artifacts[0], proba)
    stores.log(f"    [{model_name}] Probabilities attached to {artifacts[0].name}")


def run(
    context: RunContext,
    model_name: str,
    model_info: types.ModelInfo,
) -> None:
    """Execute and persist one model while preserving benchmark contracts."""
    run_dir = results_layout.build_results_run_dir(
        context.output_dir,
        context.split_variant,
        context.task_name,
        context.emb_name,
        model_name,
        task_scope=context.task_scope,
    )
    if context.attach_probabilities:
        _attach_run_probabilities(context, model_name, model_info, run_dir)
        return
    if not context.overwrite_results and results_layout.is_run_complete(run_dir):
        stores.log(f"    [{model_name}] Already trained — skipping.")
        return
    if context.overwrite_results and run_dir.exists():
        shutil.rmtree(run_dir)

    stores.log(
        f"    [{model_name}] n_train={context.sizes.n_train}, "
        f"n_test={context.sizes.n_test} | route={context.route.name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = results_layout.build_artifact_stem(
        context.task_name,
        context.emb_name,
        model_name,
        context.split_variant,
    )
    seeded_info = run_config.apply_random_seed(model_info, context.random_seed)
    if context.route is train.ATTENTION_ROUTE:
        seeded_info = run_config.apply_attention_overrides(
            seeded_info,
            context.attention_overrides,
        )
    common.seed_random_generators(
        context.random_seed,
        deterministic_algorithms=context.deterministic_algorithms,
    )

    sample_manifest = run_artifacts.build_sample_manifest(context.prepared)
    hparams_path = serialization.write_hparams_file(
        run_dir,
        task_name=context.task_name,
        emb_name=context.emb_name,
        model_name=model_name,
        split_variant=context.split_variant,
        model_info=seeded_info,
        task_config=context.config,
        sample_identity_manifest=sample_manifest,
        random_seed=context.random_seed,
        deterministic_algorithms=context.deterministic_algorithms,
    )
    reproducibility.write_run_manifest(
        run_dir,
        dataset_path=context.dataset_path,
        embedding_path=context.embedding_path,
        hparams_path=hparams_path,
        split_manifest=sample_manifest,
        deterministic_algorithms=context.deterministic_algorithms,
    )
    failure_path = run_dir / "training_failure.json"
    if failure_path.exists():
        raise ValueError(
            f"Previous training failed: {failure_path}. Review it before retrying."
        )
    try:
        saved_run = _train_and_predict(context, model_name, seeded_info, run_dir)
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        failure_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    trained = saved_run.trained
    serialization.write_training_log_file(trained, run_dir)
    run_artifacts.save_trained_model(
        trained,
        route_name=context.route.name,
        run_dir=run_dir,
        task_name=context.task_name,
        emb_name=context.emb_name,
        model_name=model_name,
        split_variant=context.split_variant,
    )

    # AUROC and AUPRC are areas under a curve swept over a score, so a single
    # confusion matrix cannot express them. Writing probabilities here keeps the
    # ranking metrics on their real definition instead of a one-threshold lower
    # bound; wrappers without a probability method are skipped inside.
    _attach_run_probabilities(context, model_name, model_info, run_dir)

    if getattr(trained, "best_epoch_", None) is not None:
        stores.log(
            f"      Early stopping: best_epoch={trained.best_epoch_},"
            f" best_validation_objective={trained.best_val_loss_:.6f}"
        )
    relative_dir = run_dir.relative_to(context.output_dir)
    stores.log(f"      Saved → {relative_dir} [{artifact_stem}]")
