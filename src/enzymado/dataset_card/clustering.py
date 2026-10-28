"""Compute and visualize whole-dataset sequence clusters for dataset cards."""

import collections
from pathlib import Path
import shutil
import subprocess
import tempfile
import typing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import statistics


def _sequence_map(base_df: pd.DataFrame, seq_col: str) -> dict[str, str]:
    """Map normalized unique sequences to their preferred identifiers."""
    id_col = next(
        (column for column in base_df.columns if column in ["uniprot_id"]),
        None,
    )
    columns = [seq_col] + ([id_col] if id_col else [])
    sequence_to_id = {}
    for _, row in base_df[columns].dropna(subset=[seq_col]).iterrows():
        sequence = "".join(str(row[seq_col]).strip().split()).upper()
        identifier = str(row[id_col]).strip() if id_col else sequence[:16]
        if sequence and sequence not in sequence_to_id:
            sequence_to_id[sequence] = identifier
    return sequence_to_id


def _write_fasta(sequence_to_id: dict[str, str], fasta_path: Path) -> None:
    """Write MMseqs-safe sequential identifiers for unique sequences."""
    with fasta_path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequence_to_id):
            handle.write(f">seq{index}\n{sequence}\n")


def _run_mmseqs(
    fasta_path: Path,
    work_dir: Path,
    seq_identity: float,
) -> Path | None:
    """Run the existing MMseqs2 command sequence and return its assignment TSV."""
    database_path = work_dir / "db"
    cluster_database_path = work_dir / "cluster_db"
    tsv_path = work_dir / "clusters.tsv"
    temporary_work = work_dir / "tmp"
    try:
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_path), str(database_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "mmseqs",
                "cluster",
                str(database_path),
                str(cluster_database_path),
                str(temporary_work),
                "--min-seq-id",
                str(seq_identity),
                "-c",
                "0.8",
                "--cov-mode",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "mmseqs",
                "createtsv",
                str(database_path),
                str(database_path),
                str(cluster_database_path),
                str(tsv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"  [Cluster] MMSeqs2 failed: {error.stderr[:300]}")
        return None
    return tsv_path


def _read_assignments(tsv_path: Path) -> dict[str, str]:
    """Read MMseqs2 representative/member pairs keyed by member identifier."""
    assignments = {}
    with tsv_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                assignments[parts[1]] = parts[0]
    return assignments


def run(
    base_df: pd.DataFrame,
    _prop_name: str,
    seq_identity: float,
) -> dict[str, typing.Any] | None:
    """Run MMseqs2 easy-cluster on the unique sequences in one dataset."""
    if shutil.which("mmseqs") is None:
        print("  [Cluster] mmseqs not found in PATH, skipping clustering.")
        print(
            "            Hint: run 'module load mmseqs2' before launching this script."
        )
        return None

    seq_col = next(
        (
            column
            for column in base_df.columns
            if column in ["protein_sequence", "wildtype_sequence"]
        ),
        None,
    )
    if seq_col is None:
        print("  [Cluster] No sequence column found, skipping clustering.")
        return None

    sequence_to_id = _sequence_map(base_df, seq_col)
    total_sequences = len(sequence_to_id)
    if total_sequences < 2:
        print(
            f"  [Cluster] Only {total_sequences} unique sequences, skipping clustering."
        )
        return None

    print(
        f"  [Cluster] Running MMSeqs2 on {total_sequences} unique sequences "
        f"(identity={seq_identity})..."
    )
    with tempfile.TemporaryDirectory(prefix="enzymado-cluster-") as temporary_dir:
        work_dir = Path(temporary_dir)
        fasta_path = work_dir / "seqs.fasta"
        _write_fasta(sequence_to_id, fasta_path)
        tsv_path = _run_mmseqs(fasta_path, work_dir, seq_identity)
        if tsv_path is None:
            return None
        assignments = _read_assignments(tsv_path)

    cluster_sizes = collections.Counter(assignments.values())
    sizes = sorted(cluster_sizes.values(), reverse=True)
    n_clusters = len(cluster_sizes)
    n_singletons = sum(1 for size in sizes if size == 1)
    largest = sizes[0] if sizes else 0
    print(
        f"  [Cluster] Done: {n_clusters} clusters from "
        f"{total_sequences} sequences "
        f"({n_singletons} singletons, largest={largest})."
    )

    return statistics.sanitize(
        {
            "cluster_method": "mmseqs2",
            "cluster_parameters": {
                "min_sequence_identity": seq_identity,
                "coverage_mode": 0,
                "coverage_threshold": 0.8,
            },
            "total_sequences": total_sequences,
            "number_of_clusters": n_clusters,
            "singleton_clusters": n_singletons,
            "largest_cluster_size": largest,
            "ratio_unique_variants": round(n_clusters / total_sequences, 4),
            "_cluster_sizes": sizes,
        }
    )


def _plot_rank_abundance(
    axis: typing.Any,
    sizes_array: np.ndarray,
) -> None:
    """Draw the cluster rank-abundance panel."""
    n_clusters = len(sizes_array)
    ranks = np.arange(1, n_clusters + 1)
    axis.plot(ranks, sizes_array, color="#2D6A9F", linewidth=1.5)
    axis.fill_between(ranks, sizes_array, alpha=0.15, color="#2D6A9F")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Cluster rank (log)", fontsize=10)
    axis.set_ylabel("Cluster size (log)", fontsize=10)
    axis.set_title("Rank–Abundance", fontsize=10)
    axis.axvline(
        n_clusters * 0.5,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label="50th percentile rank",
    )
    singleton_fraction = (sizes_array == 1).sum() / n_clusters
    axis.text(
        0.98,
        0.97,
        f"Singletons: {singleton_fraction:.1%}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555",
    )
    axis.legend(fontsize=7)
    axis.grid(visible=True, which="both", linestyle=":", alpha=0.5)


def _plot_rarefaction(
    axis: typing.Any,
    sizes_array: np.ndarray,
) -> None:
    """Draw the deterministic rarefaction panel."""
    n_clusters = len(sizes_array)
    total_sequences = int(sizes_array.sum())
    rng = np.random.default_rng(42)
    sequence_labels = np.repeat(np.arange(n_clusters), sizes_array)
    n_points = min(40, total_sequences)
    sample_sizes = np.unique(
        np.concatenate(
            [
                np.geomspace(1, total_sequences, n_points).astype(int),
                [total_sequences],
            ]
        )
    )
    means = []
    standard_deviations = []
    for sample_size in sample_sizes:
        counts = []
        for _ in range(5):
            sample = rng.choice(
                sequence_labels,
                size=sample_size,
                replace=False,
            )
            counts.append(len(np.unique(sample)))
        means.append(np.mean(counts))
        standard_deviations.append(np.std(counts))

    means_array = np.array(means)
    standard_deviations_array = np.array(standard_deviations)
    axis.plot(
        sample_sizes,
        means_array,
        color="#C0392B",
        linewidth=1.5,
        label="Mean (5 replicates)",
    )
    axis.fill_between(
        sample_sizes,
        means_array - standard_deviations_array,
        means_array + standard_deviations_array,
        alpha=0.2,
        color="#C0392B",
        label="±1 SD",
    )
    axis.set_xlabel("Sample size (sequences)", fontsize=10)
    axis.set_ylabel("Unique clusters discovered", fontsize=10)
    axis.set_title("Rarefaction Curve", fontsize=10)
    axis.plot(
        [total_sequences],
        [n_clusters],
        "k^",
        markersize=7,
        label=f"Full set: {n_clusters} clusters",
    )
    axis.legend(fontsize=7)
    axis.grid(visible=True, linestyle=":", alpha=0.5)


def plot_diversity(
    cluster_stats: dict[str, typing.Any],
    prop_name: str,
    output_dir: Path,
    seq_identity: float,
) -> str | None:
    """Generate the existing rank-abundance and rarefaction figure."""
    sizes = cluster_stats.get("_cluster_sizes")
    if not sizes:
        return None

    matplotlib.use("Agg")
    sizes_array = np.array(sizes, dtype=np.int64)
    total_sequences = int(sizes_array.sum())
    n_clusters = len(sizes_array)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    figure.suptitle(
        f"{prop_name}: Sequence Diversity  "
        f"(MMSeqs2 identity≥{seq_identity:.0%}, "
        f"{n_clusters} clusters / {total_sequences} sequences)",
        fontsize=11,
        fontweight="bold",
    )
    _plot_rank_abundance(axes[0], sizes_array)
    _plot_rarefaction(axes[1], sizes_array)

    plt.tight_layout()
    output_path = output_dir / f"{prop_name}_cluster_diversity.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  [ClusterDiversity] Saved cluster diversity panel → {output_path}")
    return str(output_path)
