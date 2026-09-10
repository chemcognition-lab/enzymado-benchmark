"""Run model, task, and embedding combinations for the benchmark.

Example:
    python scripts/run_benchmark.py --help
"""

from __future__ import annotations

import argparse
import collections.abc
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "predictions"
sys.path.insert(0, str(REPO_ROOT / "src"))

import enzymado.data_loading.data_registry  # noqa: E402
import enzymado.data_loading.stores  # noqa: E402
import enzymado.data_loading.task_data  # noqa: E402
import enzymado.evaluation.task_types  # noqa: E402
import enzymado.pipeline.run_data  # noqa: E402
import enzymado.pipeline.run_execution  # noqa: E402
import enzymado.training.model_selection  # noqa: E402
import enzymado.training.train  # noqa: E402
import enzymado.types  # noqa: E402

_DEFAULT_RANDOM_SEED = 42

_ModelMap = enzymado.types.ModelRegistry
_ModelMapFn = collections.abc.Callable[[str, str, str | None], _ModelMap]


def _run_prepared_task(
    *,
    config: enzymado.data_loading.data_registry.TaskConfig,
    task_name: str,
    task_scope: str,
    emb_name: str,
    emb_info: enzymado.types.EmbeddingInfo,
    output_dir: Path,
    split_variant: str,
    model_map: _ModelMap,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    residue_store_mode: str,
    overwrite_results: bool,
    random_seed: int,
    deterministic_algorithms: bool = False,
    attention_overrides: dict[str, object] | None,
    training_route_override: enzymado.training.train.TrainingRouteSpec | None,
    attach_probabilities: bool = False,
) -> None:
    """Prepare one task, run its selected models, and release loaded data."""
    embedding_dim = int(emb_info["dim"]) if emb_info.get("dim") is not None else None
    prepared_task = enzymado.pipeline.run_data.prepare(
        config=config,
        task_name=task_name,
        emb_path=emb_info["path"],
        split_variant=split_variant,
        allowed_ids=allowed_ids,
        sequence_to_id=sequence_to_id,
        residue_store_mode=residue_store_mode,
        embedding_dim=embedding_dim,
        route_override=training_route_override,
    )
    prepared = prepared_task.data
    try:
        sizes = enzymado.pipeline.run_data.summarize(prepared)
        if sizes.n_train == 0 or sizes.n_test == 0:
            # Not a skip: every registered task has data, so an empty split
            # means the filters (allowlist, split variant, embedding) are
            # misconfigured. Silently returning leaves a hole in the results
            # tree that nothing downstream reports.
            raise ValueError(
                f"Task {task_name!r} has an empty dataset after filtering "
                f"(n_train={sizes.n_train}, n_test={sizes.n_test})."
            )

        if sizes.use_validation:
            enzymado.data_loading.stores.log(
                f"    Validation enabled: n_val={sizes.n_val:,}"
            )
        else:
            enzymado.data_loading.stores.log(
                "    Validation disabled: empty val split after filtering"
            )

        context = enzymado.pipeline.run_execution.RunContext(
            output_dir=output_dir,
            task_scope=task_scope,
            task_name=task_name,
            emb_name=emb_name,
            split_variant=split_variant,
            config=config,
            prepared=prepared,
            route=prepared_task.route,
            sizes=sizes,
            overwrite_results=overwrite_results,
            random_seed=random_seed,
            dataset_path=Path(config.dataset_paths[split_variant]),
            embedding_path=Path(emb_info["path"]),
            deterministic_algorithms=deterministic_algorithms,
            attention_overrides=attention_overrides,
            attach_probabilities=attach_probabilities,
        )
        for model_name, model_info in model_map.items():
            enzymado.pipeline.run_execution.run(context, model_name, model_info)
    finally:
        prepared.cleanup()


def _run_task(
    *,
    task_scope: str,
    task_name: str,
    emb_name: str,
    emb_info: enzymado.types.EmbeddingInfo,
    output_dir: Path,
    model_prefix_filter: str | None,
    allowed_ids: set[str] | None,
    sequence_to_id: dict[str, str],
    skip_label_kinds: set[str],
    residue_store_mode: str,
    split_variant_filter: str | None,
    embedding_mode_filters: set[str],
    overwrite_results: bool,
    random_seed: int,
    deterministic_algorithms: bool = False,
    attention_overrides: dict[str, object] | None = None,
    training_route_override: enzymado.training.train.TrainingRouteSpec | None = None,
    model_map_fn: _ModelMapFn | None = None,
    attach_probabilities: bool = False,
) -> bool:
    """Select and run models for one task and embedding pair."""
    config = enzymado.data_loading.data_registry.get_task_config(task_name)
    task_type = enzymado.evaluation.task_types.from_label_kind(str(config.label_kind))
    if task_type is None:
        raise ValueError(f"Unsupported label kind {str(config.label_kind)!r}.")
    if config.label_kind in skip_label_kinds:
        enzymado.data_loading.stores.log(
            f"  Skipping {task_name!r} "
            f"(label_kind={str(config.label_kind)!r} filtered by CLI)"
        )
        return True

    split_variant = split_variant_filter or "random"
    if split_variant not in config.dataset_paths:
        available = ", ".join(sorted(config.dataset_paths)) or "<none>"
        enzymado.data_loading.stores.log(
            f"  Skipping {task_name!r} (requested split {split_variant!r} is not "
            f"available; available: {available})"
        )
        return True

    enzymado.data_loading.stores.log(
        f"\n  Task: {task_name} | type: {task_type} | split: {split_variant}"
    )
    try:
        selector = model_map_fn or enzymado.training.model_selection.select_models
        model_map = selector(task_type, config.label_kind, model_prefix_filter)
        model_map = enzymado.training.model_selection.filter_by_embedding_modes(
            model_map,
            task_name,
            embedding_mode_filters,
        )
        if not model_map:
            enzymado.data_loading.stores.log(
                "    No models selected after filters — skipping task."
            )
            return True

        if attach_probabilities:
            # Completed runs are the input here, not something to skip: their
            # stored weights are what gets restored.
            enzymado.data_loading.stores.log(
                f"    Attach mode: adding probabilities to {len(model_map)} "
                "completed models."
            )
        elif overwrite_results:
            enzymado.data_loading.stores.log(
                "    Overwrite enabled: existing artifacts for selected models will "
                "be replaced."
            )
        else:
            completed = enzymado.training.model_selection.find_completed_models(
                model_map,
                output_dir,
                split_variant,
                task_name,
                emb_name,
                task_scope,
            )
            if len(completed) == len(model_map):
                enzymado.data_loading.stores.log(
                    f"    All {len(completed)} models already trained — skipping task."
                )
                return True
            if completed:
                enzymado.data_loading.stores.log(
                    f"    {len(completed)}/{len(model_map)} models already done: "
                    f"{completed}"
                )

        _run_prepared_task(
            config=config,
            task_name=task_name,
            task_scope=task_scope,
            emb_name=emb_name,
            emb_info=emb_info,
            output_dir=output_dir,
            split_variant=split_variant,
            model_map=model_map,
            allowed_ids=allowed_ids,
            sequence_to_id=sequence_to_id,
            residue_store_mode=residue_store_mode,
            overwrite_results=overwrite_results,
            random_seed=random_seed,
            deterministic_algorithms=deterministic_algorithms,
            attention_overrides=attention_overrides,
            training_route_override=training_route_override,
            attach_probabilities=attach_probabilities,
        )
        return True
    except Exception as task_exc:  # noqa: BLE001
        enzymado.data_loading.stores.log(
            f"  [ERROR] Task {task_name!r} failed: {task_exc}"
        )
        enzymado.data_loading.stores.log(traceback.format_exc())
        return False


def _build_attention_overrides(
    *,
    lr: float | None,
    gradient_clip_norm: float | None,
    pre_layer_norm: bool,
    batch_diagnostics: bool,
    diagnostic_loss_threshold: float,
    diagnostic_z_threshold: float,
    diagnostic_grad_norm_threshold: float,
) -> dict[str, object]:
    """Build opt-in attention settings from the existing CLI arguments."""
    overrides: dict[str, object] = {}
    if lr is not None:
        if lr <= 0:
            raise ValueError("attention_lr must be positive when provided.")
        overrides["lr"] = float(lr)
    if gradient_clip_norm is not None:
        if gradient_clip_norm <= 0:
            raise ValueError(
                "attention_gradient_clip_norm must be positive when provided."
            )
        overrides["gradient_clip_norm"] = float(gradient_clip_norm)
    if pre_layer_norm:
        overrides["pre_attention_layer_norm"] = True
    if batch_diagnostics:
        overrides.update(
            {
                "record_batch_diagnostics": True,
                "diagnostic_loss_threshold": float(diagnostic_loss_threshold),
                "diagnostic_z_threshold": float(diagnostic_z_threshold),
                "diagnostic_grad_norm_threshold": float(diagnostic_grad_norm_threshold),
            }
        )
    return overrides


@dataclass(frozen=True)
class _BenchmarkOptions:
    """Normalized options shared by regular and attention embedding runs."""

    output_dir: Path
    emb_name_filter: str | None
    model_prefix_filter: str | None
    task_name_filters: frozenset[str]
    skip_label_kinds: frozenset[str]
    allowed_ids: set[str] | None
    sequence_to_id: dict[str, str]
    residue_store_mode: str
    split_variant_filter: str | None
    embedding_mode_filters: frozenset[str]
    overwrite_results: bool
    random_seed: int
    deterministic_algorithms: bool
    attach_probabilities: bool


def _normalize_task_scopes(task_scope_filters: set[str] | None) -> set[str]:
    """Apply the default task scope and reject unsupported values."""
    scopes = task_scope_filters or {"properties"}
    invalid_scopes = scopes - {"properties", "real_case"}
    if invalid_scopes:
        raise ValueError(f"Unknown task scopes: {sorted(invalid_scopes)}")
    return scopes


def _load_allowed_id_context(
    allowed_ids_path: str | None,
) -> tuple[set[str] | None, dict[str, str]]:
    """Load an optional embedding allowlist and its sequence lookup index."""
    if allowed_ids_path is None:
        return None, {}
    allowed_ids = enzymado.data_loading.stores.load_allowed_ids(allowed_ids_path)
    sequence_to_id: dict[str, str] = {}
    sequence_index_path = enzymado.data_loading.stores.resolve_sequence_index_path()
    if sequence_index_path is not None:
        sequence_to_id = enzymado.data_loading.stores.load_sequence_to_id_index(
            sequence_index_path
        )
    enzymado.data_loading.stores.log(
        f"[startup] Restricting training rows to {len(allowed_ids):,} "
        f"embedding IDs from {allowed_ids_path}"
    )
    return allowed_ids, sequence_to_id


def _validate_embedding_filter(
    embeddings: collections.abc.Mapping[str, enzymado.types.EmbeddingInfo],
    emb_name_filter: str | None,
    *,
    kind: str,
) -> None:
    """Reject an embedding filter absent from the selected manifest."""
    if emb_name_filter is None or emb_name_filter in embeddings:
        return
    available = ", ".join(embeddings)
    raise ValueError(
        f"Unknown {kind}embedding {emb_name_filter!r}. Available: {available}"
    )


def _select_run_tasks(
    selected_tasks: list[
        tuple[str, enzymado.data_loading.data_registry.TaskConfig]
    ],
    task_name_filters: frozenset[str],
) -> list[tuple[str, enzymado.data_loading.data_registry.TaskConfig]]:
    """Apply the CLI task-name filter while logging excluded tasks."""
    run_tasks: list[
        tuple[str, enzymado.data_loading.data_registry.TaskConfig]
    ] = []
    for task_name, config in selected_tasks:
        if task_name_filters and task_name not in task_name_filters:
            enzymado.data_loading.stores.log(
                f"  Skipping {task_name!r} (not selected by CLI task filter)"
            )
            continue
        run_tasks.append((task_name, config))
    return run_tasks


def _log_startup_options(
    options: _BenchmarkOptions,
    task_scope_filters: set[str],
    attention_overrides: dict[str, object],
) -> None:
    """Log the normalized benchmark selection and reproducibility controls."""
    enzymado.data_loading.stores.log(
        f"[startup] Python started. emb_name={options.emb_name_filter}"
    )
    enzymado.data_loading.stores.log(
        f"[startup] Training random seed: {options.random_seed}"
    )
    if options.attach_probabilities:
        enzymado.data_loading.stores.log(
            "[startup] Attach-probabilities mode: restoring stored weights, no "
            "training, published hard labels preserved."
        )
    enzymado.data_loading.stores.log(
        "[startup] Deterministic PyTorch algorithms: "
        f"{options.deterministic_algorithms}"
    )
    if attention_overrides:
        enzymado.data_loading.stores.log(
            f"[startup] Attention experiment overrides: {attention_overrides}"
        )
    if options.model_prefix_filter is not None:
        enzymado.data_loading.stores.log(
            "[startup] Filtering model names by prefix: "
            f"{options.model_prefix_filter}"
        )
    if options.task_name_filters:
        enzymado.data_loading.stores.log(
            f"[startup] Restricting tasks to: {sorted(options.task_name_filters)}"
        )
    enzymado.data_loading.stores.log(
        f"[startup] Restricting task scopes to: {sorted(task_scope_filters)}"
    )
    if options.skip_label_kinds:
        enzymado.data_loading.stores.log(
            f"[startup] Skipping label kinds: {sorted(options.skip_label_kinds)}"
        )
    enzymado.data_loading.stores.log(
        f"[startup] Residue store mode: {options.residue_store_mode}"
    )
    if options.split_variant_filter is not None:
        enzymado.data_loading.stores.log(
            "[startup] Restricting split variant to: "
            f"{options.split_variant_filter}"
        )
    if options.embedding_mode_filters:
        enzymado.data_loading.stores.log(
            "[startup] Restricting embedding modes to: "
            f"{sorted(options.embedding_mode_filters)}"
        )
    enzymado.data_loading.stores.log(
        f"[startup] Overwrite results: {options.overwrite_results}"
    )


def _run_embedding_group(
    embeddings: collections.abc.Mapping[str, enzymado.types.EmbeddingInfo],
    run_tasks: list[tuple[str, enzymado.data_loading.data_registry.TaskConfig]],
    options: _BenchmarkOptions,
    *,
    heading: str,
    attention_overrides: dict[str, object] | None = None,
    training_route_override: enzymado.training.train.TrainingRouteSpec | None = None,
    model_map_fn: _ModelMapFn | None = None,
) -> list[str]:
    """Run one regular or attention manifest, returning the tasks that failed.

    Individual failures do not abort the batch — the remaining combinations are
    still worth training — but they are reported upwards so the process can
    exit non-zero. A job that silently exits 0 after losing a task makes SLURM,
    and every downstream check, report success.
    """
    failures: list[str] = []
    for emb_name, emb_info in embeddings.items():
        if (
            options.emb_name_filter is not None
            and emb_name != options.emb_name_filter
        ):
            continue
        enzymado.data_loading.stores.log(
            f"\n=== {heading}: {emb_name} (dim={emb_info['dim']}) ==="
        )
        for task_name, config in run_tasks:
            ok = _run_task(
                task_scope=config.task_scope,
                task_name=task_name,
                emb_name=emb_name,
                emb_info=emb_info,
                output_dir=options.output_dir,
                model_prefix_filter=options.model_prefix_filter,
                allowed_ids=options.allowed_ids,
                sequence_to_id=options.sequence_to_id,
                skip_label_kinds=set(options.skip_label_kinds),
                residue_store_mode=options.residue_store_mode,
                split_variant_filter=options.split_variant_filter,
                embedding_mode_filters=set(options.embedding_mode_filters),
                overwrite_results=options.overwrite_results,
                random_seed=options.random_seed,
                deterministic_algorithms=options.deterministic_algorithms,
                attention_overrides=attention_overrides,
                training_route_override=training_route_override,
                model_map_fn=model_map_fn,
                attach_probabilities=options.attach_probabilities,
            )
            if not ok:
                failures.append(f"{emb_name}/{task_name}")
    return failures


def main(
    embeddings_json: str,
    output_dir: str,
    emb_name_filter: str | None = None,
    model_prefix_filter: str | None = None,
    task_name_filters: set[str] | None = None,
    task_scope_filters: set[str] | None = None,
    skip_label_kinds: set[str] | None = None,
    allowed_ids_path: str | None = None,
    residue_store_mode: str = "auto",
    split_variant_filter: str | None = None,
    embedding_mode_filters: set[str] | None = None,
    overwrite_results: bool = False,  # noqa: FBT001, FBT002
    attention_embeddings_json: str | None = None,
    random_seed: int = _DEFAULT_RANDOM_SEED,
    deterministic: bool = False,  # noqa: FBT001, FBT002
    attention_lr: float | None = None,
    attention_gradient_clip_norm: float | None = None,
    attention_pre_layer_norm: bool = False,  # noqa: FBT001, FBT002
    attention_batch_diagnostics: bool = False,  # noqa: FBT001, FBT002
    diagnostic_loss_threshold: float = 100.0,
    diagnostic_z_threshold: float = 20.0,
    diagnostic_grad_norm_threshold: float = 1000.0,
    attach_probabilities: bool = False,  # noqa: FBT001, FBT002
) -> list[str]:
    """Train selected embedding, model, and task combinations."""
    task_scopes = _normalize_task_scopes(task_scope_filters)
    attention_overrides = _build_attention_overrides(
        lr=attention_lr,
        gradient_clip_norm=attention_gradient_clip_norm,
        pre_layer_norm=attention_pre_layer_norm,
        batch_diagnostics=attention_batch_diagnostics,
        diagnostic_loss_threshold=diagnostic_loss_threshold,
        diagnostic_z_threshold=diagnostic_z_threshold,
        diagnostic_grad_norm_threshold=diagnostic_grad_norm_threshold,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    allowed_ids, sequence_to_id = _load_allowed_id_context(allowed_ids_path)
    options = _BenchmarkOptions(
        output_dir=output_path,
        emb_name_filter=emb_name_filter,
        model_prefix_filter=model_prefix_filter,
        task_name_filters=frozenset(task_name_filters or ()),
        skip_label_kinds=frozenset(skip_label_kinds or ()),
        allowed_ids=allowed_ids,
        sequence_to_id=sequence_to_id,
        residue_store_mode=residue_store_mode,
        split_variant_filter=split_variant_filter,
        embedding_mode_filters=frozenset(embedding_mode_filters or ()),
        overwrite_results=overwrite_results,
        random_seed=random_seed,
        deterministic_algorithms=deterministic,
        attach_probabilities=attach_probabilities,
    )
    _log_startup_options(options, task_scopes, attention_overrides)

    embeddings = enzymado.data_loading.stores.load_embeddings(embeddings_json)
    _validate_embedding_filter(embeddings, emb_name_filter, kind="")
    selected_tasks = [
        (name, config)
        for name, config in enzymado.data_loading.data_registry.TASK_REGISTRY.items()
        if config.task_scope in task_scopes
    ]
    run_tasks = _select_run_tasks(selected_tasks, options.task_name_filters)
    enzymado.data_loading.stores.log(
        f"[startup] Loaded. {len(embeddings)} embedding models, "
        f"{len(selected_tasks)} tasks."
    )
    failures = _run_embedding_group(
        embeddings,
        run_tasks,
        options,
        heading="Embedding",
    )

    if attention_embeddings_json is not None:
        attention_embeddings = enzymado.data_loading.stores.load_embeddings(
            attention_embeddings_json
        )
        _validate_embedding_filter(
            attention_embeddings,
            emb_name_filter,
            kind="attention ",
        )
        enzymado.data_loading.stores.log(
            f"[attention] {len(attention_embeddings)} attention embedding models found."
        )
        failures += _run_embedding_group(
            attention_embeddings,
            run_tasks,
            options,
            heading="Attention Embedding",
            attention_overrides=attention_overrides,
            training_route_override=enzymado.training.train.ATTENTION_ROUTE,
            model_map_fn=(
                enzymado.training.model_selection.select_attention_models
            ),
        )

    if failures:
        enzymado.data_loading.stores.log(
            f"Done with {len(failures)} failed task(s): {', '.join(failures)}"
        )
    else:
        enzymado.data_loading.stores.log("Done.")
    return failures


def _add_core_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset, model, split, and output arguments."""
    parser.add_argument("embeddings_json", help="Path to embeddings_models_output.json")
    parser.add_argument(
        "--output_dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=(
            "Directory to write prediction and model files "
            f"(default: {_DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--emb_name",
        default=None,
        help="Train only this embedding (for SLURM arrays)",
    )
    parser.add_argument(
        "--model_prefix",
        default=None,
        help="Train only models whose name starts with this prefix, e.g. \x27mlp_\x27.",
    )
    parser.add_argument(
        "--task_name",
        action="append",
        default=[],
        help="Train only this benchmark task name. Repeatable.",
    )
    parser.add_argument(
        "--task_scope",
        action="append",
        choices=("properties", "real_case"),
        default=[],
        help=(
            "Train tasks from this benchmark-card section. Repeatable. "
            "Defaults to properties."
        ),
    )
    parser.add_argument(
        "--skip_label_kind",
        action="append",
        default=[],
        help="Skip tasks with this label_kind. Repeatable.",
    )
    parser.add_argument(
        "--allowed_ids",
        default=None,
        help=(
            "Optional path to common_ids.txt. Rows whose embeddings are not in "
            "this ID set are excluded."
        ),
    )
    parser.add_argument(
        "--residue_store_mode",
        choices=("auto", "materialized", "lazy"),
        default="auto",
        help=(
            "How protein_residue tasks should access residue embeddings. "
            "\x27auto\x27 switches to lazy mode when the estimated materialized "
            "cache is too large."
        ),
    )
    parser.add_argument(
        "--split_variant",
        choices=("random", "similarity"),
        default=None,
        help=(
            "Train only this split variant. Defaults to \x27random\x27. A task that "
            "registers no similarity split is skipped when \x27similarity\x27 is "
            "requested."
        ),
    )
    parser.add_argument(
        "--embedding_mode",
        action="append",
        choices=("attention", "no_attention"),
        default=[],
        help="Train only model/task combinations in this results embedding mode.",
    )
    parser.add_argument(
        "--attach_probabilities",
        action="store_true",
        help=(
            "Skip training: restore each published run's stored weights, predict "
            "test-split probabilities, and add them to the existing artifact "
            "without touching the stored hard labels."
        ),
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help=(
            "Replace existing artifacts for selected runs instead of skipping "
            "completed runs."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic PyTorch operations and record that mode.",
    )


def _add_attention_args(parser: argparse.ArgumentParser) -> None:
    """Add attention experiment and diagnostic arguments."""
    parser.add_argument(
        "--attention_embeddings_json",
        default=None,
        help=(
            "Path to a per-residue embeddings JSON. When provided, also trains "
            "attention models."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_RANDOM_SEED,
        help=(
            f"Training seed used for every model run (default: {_DEFAULT_RANDOM_SEED})."
        ),
    )
    parser.add_argument(
        "--attention_lr",
        type=float,
        default=None,
        help="Optional learning-rate override for attention models only.",
    )
    parser.add_argument(
        "--attention_gradient_clip_norm",
        type=float,
        default=None,
        help="Optional max gradient norm for attention models only.",
    )
    parser.add_argument(
        "--attention_pre_layer_norm",
        action="store_true",
        help="Enable the experimental LayerNorm immediately before pooled attention.",
    )
    parser.add_argument(
        "--attention_batch_diagnostics",
        action="store_true",
        help="Stream per-batch attention diagnostics into each run directory.",
    )
    parser.add_argument(
        "--diagnostic_loss_threshold",
        type=float,
        default=100.0,
        help="Batch loss threshold used to identify the first problematic batch.",
    )
    parser.add_argument(
        "--diagnostic_z_threshold",
        type=float,
        default=20.0,
        help="Normalized input magnitude threshold for a problematic batch.",
    )
    parser.add_argument(
        "--diagnostic_grad_norm_threshold",
        type=float,
        default=1000.0,
        help="Gradient norm threshold for a problematic batch.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the existing benchmark command-line interface."""
    parser = argparse.ArgumentParser()
    _add_core_args(parser)
    _add_attention_args(parser)
    return parser


def _run_cli() -> None:
    """Parse command-line arguments and invoke the benchmark runner."""
    args = _build_parser().parse_args()
    failures = main(
        args.embeddings_json,
        args.output_dir,
        args.emb_name,
        model_prefix_filter=args.model_prefix,
        task_name_filters=set(args.task_name),
        task_scope_filters=set(args.task_scope) or None,
        skip_label_kinds=set(args.skip_label_kind),
        allowed_ids_path=args.allowed_ids,
        residue_store_mode=args.residue_store_mode,
        split_variant_filter=args.split_variant,
        embedding_mode_filters=set(args.embedding_mode),
        overwrite_results=args.overwrite_results,
        attention_embeddings_json=args.attention_embeddings_json,
        random_seed=args.seed,
        deterministic=args.deterministic,
        attention_lr=args.attention_lr,
        attention_gradient_clip_norm=args.attention_gradient_clip_norm,
        attention_pre_layer_norm=args.attention_pre_layer_norm,
        attention_batch_diagnostics=args.attention_batch_diagnostics,
        diagnostic_loss_threshold=args.diagnostic_loss_threshold,
        diagnostic_z_threshold=args.diagnostic_z_threshold,
        diagnostic_grad_norm_threshold=args.diagnostic_grad_norm_threshold,
        attach_probabilities=args.attach_probabilities,
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} task(s) failed: {', '.join(failures)}"
        )


if __name__ == "__main__":
    _run_cli()
