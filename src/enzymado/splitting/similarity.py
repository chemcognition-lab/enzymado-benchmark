"""MMseqs2 clustering and similarity-aware dataset split logic."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import core


def build_sequence_dedup_index(
    group_to_sequence: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Deduplicate repeated sequences before MMseqs2 clustering."""
    sequence_to_sequence_id: dict[str, str] = {}
    group_to_sequence_id: dict[str, str] = {}

    for index, (group_key, sequence) in enumerate(group_to_sequence.items()):
        sequence_id = sequence_to_sequence_id.get(sequence)
        if sequence_id is None:
            sequence_id = f"seq_{index}"
            sequence_to_sequence_id[sequence] = sequence_id
        group_to_sequence_id[group_key] = sequence_id

    sequence_id_to_sequence = {
        sequence_id: sequence
        for sequence, sequence_id in sequence_to_sequence_id.items()
    }
    return group_to_sequence_id, sequence_id_to_sequence


def write_fasta(sequence_id_to_sequence: dict[str, str], fasta_path: Path) -> None:
    """Write a temporary FASTA file used for MMseqs2 clustering."""
    with fasta_path.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence in sequence_id_to_sequence.items():
            handle.write(f">{sequence_id}\n{sequence}\n")


def parse_mmseqs_cluster_tsv(
    cluster_tsv_path: Path,
    sequence_ids: set[str],
) -> dict[str, str]:
    """Parse MMseqs2 output into sequence-id to cluster-id assignments."""
    cluster_assignments = {sequence_id: sequence_id for sequence_id in sequence_ids}
    with cluster_tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cluster_id, member_id = parts[0], parts[1]
            cluster_assignments[member_id] = cluster_id
    return cluster_assignments


def build_mmseqs_cluster_command(
    db_path: Path,
    cluster_db_path: Path,
    tmp_work_dir: Path,
    *,
    identity_threshold: float,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
) -> list[str]:
    """Build the ``mmseqs cluster`` command with optional clustering controls."""
    command = [
        "mmseqs",
        "cluster",
        str(db_path),
        str(cluster_db_path),
        str(tmp_work_dir),
        "--min-seq-id",
        str(identity_threshold),
    ]
    if coverage_threshold is not None:
        command.extend(["-c", str(coverage_threshold)])
    if coverage_mode is not None:
        command.extend(["--cov-mode", str(coverage_mode)])
    if cluster_mode is not None:
        command.extend(["--cluster-mode", str(cluster_mode)])
    if similarity_type is not None:
        command.extend(["--similarity-type", str(similarity_type)])
    if sensitivity is not None:
        command.extend(["-s", str(sensitivity)])
    if e_value is not None:
        command.extend(["-e", str(e_value)])
    if alignment_mode is not None:
        command.extend(["--alignment-mode", str(alignment_mode)])
    if seq_id_mode is not None:
        command.extend(["--seq-id-mode", str(seq_id_mode)])
    if max_seqs is not None:
        command.extend(["--max-seqs", str(max_seqs)])
    if single_step_clustering:
        command.extend(["--single-step-clustering", "1"])
    if cluster_reassign:
        command.extend(["--cluster-reassign", "1"])
    return command


def cluster_sequences_with_mmseqs2(
    sequence_id_to_sequence: dict[str, str],
    identity_threshold: float,
    *,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
) -> dict[str, str]:
    """Cluster sequences with MMseqs2 and return a representative per sequence."""
    if shutil.which("mmseqs") is None:
        raise FileNotFoundError("mmseqs is not available in PATH.")

    with tempfile.TemporaryDirectory(prefix="dataset-split-mmseqs-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        fasta_path = tmp_dir / "sequences.fasta"
        db_path = tmp_dir / "db"
        cluster_db_path = tmp_dir / "cluster_db"
        cluster_tsv_path = tmp_dir / "clusters.tsv"
        tmp_work_dir = tmp_dir / "tmp"

        # Keep all MMseqs2 scratch files inside a temporary directory so split
        # generation does not leave intermediate databases in the repo tree.
        write_fasta(sequence_id_to_sequence, fasta_path)
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(db_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            build_mmseqs_cluster_command(
                db_path,
                cluster_db_path,
                tmp_work_dir,
                identity_threshold=identity_threshold,
                coverage_threshold=coverage_threshold,
                coverage_mode=coverage_mode,
                cluster_mode=cluster_mode,
                similarity_type=similarity_type,
                sensitivity=sensitivity,
                e_value=e_value,
                alignment_mode=alignment_mode,
                seq_id_mode=seq_id_mode,
                max_seqs=max_seqs,
                single_step_clustering=single_step_clustering,
                cluster_reassign=cluster_reassign,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "mmseqs",
                "createtsv",
                str(db_path),
                str(db_path),
                str(cluster_db_path),
                str(cluster_tsv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_mmseqs_cluster_tsv(cluster_tsv_path, set(sequence_id_to_sequence))


def build_group_clusters(
    group_to_sequence: dict[str, str],
    *,
    identity_threshold: float,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
) -> dict[str, str]:
    """Map each group key onto the cluster id of its representative sequence."""
    group_to_sequence_id, sequence_id_to_sequence = build_sequence_dedup_index(
        group_to_sequence
    )
    sequence_clusters = cluster_sequences_with_mmseqs2(
        sequence_id_to_sequence,
        identity_threshold,
        coverage_threshold=coverage_threshold,
        coverage_mode=coverage_mode,
        cluster_mode=cluster_mode,
        similarity_type=similarity_type,
        sensitivity=sensitivity,
        e_value=e_value,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        max_seqs=max_seqs,
        single_step_clustering=single_step_clustering,
        cluster_reassign=cluster_reassign,
    )

    return {
        group_key: sequence_clusters[sequence_id]
        for group_key, sequence_id in group_to_sequence_id.items()
    }


def generate_similarity_split_from_clusters(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    group_clusters: dict[str, str],
    group_col: str | None = None,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Assign whole sequence clusters to splits to reduce homology leakage."""
    fractions = core.validate_split_fractions(
        train_fraction, val_fraction, test_fraction
    )
    group_to_indices, _ = core.build_group_index(
        rows, sequence_col, group_col=group_col
    )

    missing_groups = sorted(set(group_to_indices) - set(group_clusters))
    if missing_groups:
        preview = ", ".join(missing_groups[:5])
        raise ValueError(f"Missing cluster assignments for groups: {preview}.")

    cluster_to_weight: dict[str, int] = defaultdict(int)
    cluster_to_groups: dict[str, list[str]] = defaultdict(list)
    for group_key, row_indices in group_to_indices.items():
        cluster_id = group_clusters[group_key]
        cluster_to_groups[cluster_id].append(group_key)
        cluster_to_weight[cluster_id] += len(row_indices)

    cluster_assignments = core.assign_items_to_splits(
        dict(cluster_to_weight),
        fractions,
        seed=seed,
    )
    group_assignments = {
        group_key: cluster_assignments[cluster_id]
        for cluster_id, group_keys in cluster_to_groups.items()
        for group_key in group_keys
    }
    output_fieldnames, split_rows = core.apply_group_assignments(
        rows,
        fieldnames,
        group_to_indices,
        group_assignments,
    )
    return (
        output_fieldnames,
        split_rows,
        core.summarize_split_rows(split_rows, sequence_col),
    )


def generate_similarity_split_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    *,
    sequence_col: str,
    group_col: str | None = None,
    identity_threshold: float = 0.3,
    coverage_threshold: float | None = None,
    coverage_mode: int | None = None,
    cluster_mode: int | None = None,
    similarity_type: int | None = None,
    sensitivity: float | None = None,
    e_value: float | None = None,
    alignment_mode: int | None = None,
    seq_id_mode: int | None = None,
    max_seqs: int | None = None,
    single_step_clustering: bool = False,
    cluster_reassign: bool = False,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, int]]]:
    """Cluster groups first, then derive split rows from those clusters."""
    group_to_indices, group_to_sequence = core.build_group_index(
        rows,
        sequence_col,
        group_col=group_col,
    )
    if len(group_to_indices) < 3:
        raise ValueError(
            "At least three proteins/groups are required to build "
            "train/val/test splits."
        )

    group_clusters = build_group_clusters(
        group_to_sequence,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        coverage_mode=coverage_mode,
        cluster_mode=cluster_mode,
        similarity_type=similarity_type,
        sensitivity=sensitivity,
        e_value=e_value,
        alignment_mode=alignment_mode,
        seq_id_mode=seq_id_mode,
        max_seqs=max_seqs,
        single_step_clustering=single_step_clustering,
        cluster_reassign=cluster_reassign,
    )
    return generate_similarity_split_from_clusters(
        rows,
        fieldnames,
        train_fraction,
        val_fraction,
        test_fraction,
        sequence_col=sequence_col,
        group_clusters=group_clusters,
        group_col=group_col,
        seed=seed,
    )
