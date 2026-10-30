"""Build bounded provenance manifests for benchmark runs."""

from __future__ import annotations

import collections.abc
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import typing

import torch

import enzymado.paths_and_constants

RUN_MANIFEST_SCHEMA_VERSION = 1
_FILE_SAMPLE_BYTES = 64 * 1024
_SOURCE_DIRS = ("src", "scripts", "embedding_scripts")
_SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "embeddings_models_output.json",
    "embeddings_models_output_mean_pooled.json",
)
_PACKAGE_DISTRIBUTIONS = {
    "joblib": "joblib",
    "numpy": "numpy",
    "pandas": "pandas",
    "safetensors": "safetensors",
    "scikit_learn": "scikit-learn",
    "scipy": "scipy",
    "torch": "torch",
    "xgboost": "xgboost",
}


def _canonical_json_sha256(payload: object) -> str:
    """Hash a JSON-compatible payload with stable key and separator settings."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_sha256(path: Path) -> str:
    """Hash a complete file by streaming bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_content_sha256(path: Path) -> tuple[str, list[int]]:
    """Hash bounded samples from a large file and return sampled offsets."""
    size_bytes = path.stat().st_size
    last_offset = max(0, size_bytes - _FILE_SAMPLE_BYTES)
    middle_offset = max(0, (size_bytes - _FILE_SAMPLE_BYTES) // 2)
    offsets = sorted({0, middle_offset, last_offset})
    digest = hashlib.sha256(str(size_bytes).encode("ascii"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(str(offset).encode("ascii"))
            digest.update(b"\0")
            digest.update(handle.read(_FILE_SAMPLE_BYTES))
    return digest.hexdigest(), offsets


def _format_manifest_path(path: Path) -> str:
    """Format a manifest path relative to the repository root when possible."""
    root = enzymado.paths_and_constants.Paths.project_root.resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@functools.lru_cache(maxsize=None)
def _fingerprint_file_cached(
    resolved_path: str,
    size_bytes: int,
    modified_ns: int,
    *,
    complete: bool,
) -> dict[str, object]:
    """Cache immutable input fingerprints within one benchmark process."""
    resolved = Path(resolved_path)
    if complete:
        kind = "content_sha256"
        sha256 = _content_sha256(resolved)
        offsets: list[int] | None = None
    else:
        kind = "sampled_content_sha256"
        sha256, offsets = _sampled_content_sha256(resolved)
    fingerprint: dict[str, object] = {
        "path": _format_manifest_path(resolved),
        "kind": kind,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "modified_ns": modified_ns,
    }
    if offsets is not None:
        fingerprint["sample_bytes"] = _FILE_SAMPLE_BYTES
        fingerprint["sample_offsets"] = offsets
    return fingerprint


def fingerprint_file(path: str | Path, *, complete: bool) -> dict[str, object]:
    """Fingerprint one required file with either complete or bounded I/O."""
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return dict(
        _fingerprint_file_cached(
            str(resolved),
            stat.st_size,
            stat.st_mtime_ns,
            complete=complete,
        )
    )


@functools.lru_cache(maxsize=None)
def _fingerprint_directory_cached(resolved_path: str) -> dict[str, object]:
    """Cache one immutable shard-tree inventory within a benchmark process."""
    resolved = Path(resolved_path)
    records = []
    total_bytes = 0
    for file_path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        stat = file_path.stat()
        total_bytes += stat.st_size
        records.append(
            {
                "relative_path": file_path.relative_to(resolved).as_posix(),
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )
    return {
        "path": _format_manifest_path(resolved),
        "kind": "tree_inventory_sha256",
        "sha256": _canonical_json_sha256(records),
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


def fingerprint_directory(path: str | Path) -> dict[str, object]:
    """Hash sorted directory metadata without reading large shard contents."""
    resolved = Path(path).resolve(strict=True)
    return dict(_fingerprint_directory_cached(str(resolved)))


def fingerprint_embedding(path: str | Path) -> dict[str, object]:
    """Use a bounded fingerprint appropriate to one embedding artifact layout."""
    embedding_path = Path(path)
    if embedding_path.is_dir():
        return fingerprint_directory(embedding_path)
    return fingerprint_file(embedding_path, complete=False)


def _source_tree_sha256(repository_root: Path) -> str:
    """Hash maintained source and configuration files, including local edits."""
    paths = []
    for directory_name in _SOURCE_DIRS:
        directory = repository_root / directory_name
        if directory.is_dir():
            paths.extend(directory.rglob("*.py"))
    paths.extend(
        path
        for filename in _SOURCE_FILES
        if (path := repository_root / filename).is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative_path = path.relative_to(repository_root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository_root: Path, *arguments: str) -> str:
    """Return one Git value for the benchmark checkout."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    """Return versions of scientific packages that may affect benchmark output."""
    versions = {}
    for name, distribution in _PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@functools.lru_cache(maxsize=1)
def _build_static_environment_manifest() -> dict[str, object]:
    """Build the process-stable, comparatively expensive environment fields."""
    repository_root = enzymado.paths_and_constants.Paths.project_root
    status = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_SOURCE_DIRS,
        *_SOURCE_FILES,
    )
    gpu_names = (
        [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_names": gpu_names,
        "git_commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_tree_sha256": _source_tree_sha256(repository_root),
    }


def build_environment_manifest() -> dict[str, object]:
    """Describe the code, runtime, accelerator, and scheduler environment."""
    return {
        **_build_static_environment_manifest(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def write_run_manifest(
    run_dir: Path,
    *,
    dataset_path: str | Path,
    embedding_path: str | Path,
    hparams_path: str | Path,
    split_manifest: collections.abc.Mapping[
        str,
        collections.abc.Mapping[str, int | str],
    ],
    deterministic_algorithms: bool,
) -> Path:
    """Persist versioned input and environment provenance for one model run."""
    environment = build_environment_manifest()
    payload: dict[str, typing.Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "deterministic_algorithms": deterministic_algorithms,
        "inputs": {
            "dataset": fingerprint_file(dataset_path, complete=True),
            "embedding": fingerprint_embedding(embedding_path),
            "hparams": fingerprint_file(hparams_path, complete=True),
        },
        "splits": split_manifest,
        "environment": environment,
        "environment_sha256": _canonical_json_sha256(environment),
    }
    path = run_dir / "run_manifest.json"
    if (run_dir / "training_checkpoint.bin").exists():
        if not path.exists():
            raise ValueError("Checkpoint has no provenance; use a new run directory.")
        previous = json.loads(path.read_text())
        keys = ("schema_version", "deterministic_algorithms", "inputs", "splits")
        environment_keys = (
            "packages",
            "source_tree_sha256",
            "cuda_runtime",
            "gpu_names",
        )
        compatible = all(previous.get(key) == payload[key] for key in keys)
        compatible &= all(
            previous["environment"].get(key) == environment[key]
            for key in environment_keys
        )
        if not compatible:
            raise ValueError(
                "Checkpoint data, splits or runtime changed; use a new run directory."
            )
        return path
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "build_environment_manifest",
    "fingerprint_directory",
    "fingerprint_embedding",
    "fingerprint_file",
    "write_run_manifest",
]
