"""Select registered models for individual benchmark training runs."""

from pathlib import Path

import enzymado.artifacts.results_layout
import enzymado.training.model_registry


def _filter_prefix(
    model_map: dict[str, dict[str, object]],
    model_prefix: str | None,
) -> dict[str, dict[str, object]]:
    """Select registry entries whose names start with the requested prefix."""
    if model_prefix is None:
        return model_map
    selected: dict[str, dict[str, object]] = {}
    for name, info in model_map.items():
        if name.startswith(model_prefix):
            selected[name] = info
    return selected


def select_models(
    task_type: str,
    label_kind: str,
    model_prefix: str | None,
) -> dict[str, dict[str, object]]:
    """Select standard models for one task and optional name prefix."""
    model_map = enzymado.training.model_registry.get_model_map(task_type, label_kind)
    return _filter_prefix(model_map, model_prefix)


def select_attention_models(
    task_type: str,
    label_kind: str,
    model_prefix: str | None,
) -> dict[str, dict[str, object]]:
    """Select attention models for one task and optional name prefix."""
    model_map = enzymado.training.model_registry.get_attention_model_map(
        task_type,
        label_kind,
    )
    return _filter_prefix(model_map, model_prefix)


def filter_by_embedding_modes(
    model_map: dict[str, dict[str, object]],
    task_name: str,
    embedding_modes: set[str],
) -> dict[str, dict[str, object]]:
    """Select models assigned to the requested embedding modes."""
    if not embedding_modes:
        return model_map
    selected: dict[str, dict[str, object]] = {}
    for name, info in model_map.items():
        mode = enzymado.artifacts.results_layout.canonical_embedding_mode(
            task_name, name
        )
        if mode in embedding_modes:
            selected[name] = info
    return selected


def find_completed_models(
    model_map: dict[str, dict[str, object]],
    output_dir: Path,
    split_variant: str,
    task_name: str,
    emb_name: str,
    task_scope: str,
) -> list[str]:
    """Return selected model names whose expected artifacts are complete."""
    completed = []
    for model_name in model_map:
        run_dir = enzymado.artifacts.results_layout.build_results_run_dir(
            output_dir,
            split_variant,
            task_name,
            emb_name,
            model_name,
            task_scope=task_scope,
        )
        if enzymado.artifacts.results_layout.is_run_complete(run_dir):
            completed.append(model_name)
    return completed
