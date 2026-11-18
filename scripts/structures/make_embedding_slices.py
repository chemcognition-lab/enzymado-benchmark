"""Split prepared structures into ID lists for parallel embedding jobs.

Reads the prepare manifests in --manifest_dir that match --pattern (all CSVs
by default), keeps the ok rows (only the IDs in --only_ids, if given), sorts them
by atom count and cuts the sorted list into --n_slices contiguous pieces of
about equal cost, with cost = n_atoms + --overhead_atoms per protein. Similar
sizes in one slice keep JAX recompilation low (few shape buckets per job) and
put the largest proteins, the ones that may run out of GPU memory, together.

Writes, under --output_dir:
    slice_<k>.txt   one sequence_id per line, ascending atom count
    slices.csv      slice, n_ids, min_atoms, max_atoms, cost

Example:
    python scripts/structures/make_embedding_slices.py \\
        --manifest_dir /scratch/.../structures_prod/prepare_manifests \\
        --n_slices 40 --output_dir /scratch/.../structures_prod/slices_40
"""

import argparse
import csv
from pathlib import Path


def load_ok_rows(
    manifest_dir: Path, pattern: str = "*.csv"
) -> list[tuple[str, int]]:
    """Return (sequence_id, n_atoms) for every ok row, newest row per ID.

    A later row for the same ID (a retried error) replaces an earlier one.
    Raises ValueError if two manifests both report the same ID as ok.
    """
    final: dict[str, tuple[str, dict]] = {}
    for path in sorted(manifest_dir.glob(pattern)):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seq_id = row["sequence_id"]
                prev = final.get(seq_id)
                if (
                    prev is not None
                    and prev[0] != path.name
                    and prev[1]["status"] == "ok"
                    and row["status"] == "ok"
                ):
                    raise ValueError(f"{seq_id} is ok in {prev[0]} and {path.name}")
                final[seq_id] = (path.name, row)
    return [
        (seq_id, int(row["n_atoms"]))
        for seq_id, (_, row) in final.items()
        if row["status"] == "ok"
    ]


def cut_slices(
    rows: list[tuple[str, int]], n_slices: int, overhead_atoms: int
) -> list[list[tuple[str, int]]]:
    """Cut rows, sorted by atoms, into n_slices contiguous pieces of near-equal cost."""
    ordered = sorted(rows, key=lambda r: (r[1], r[0]))
    total = sum(n + overhead_atoms for _, n in ordered)
    slices: list[list[tuple[str, int]]] = [[] for _ in range(n_slices)]
    cumulative = 0.0
    for seq_id, n_atoms in ordered:
        cost = n_atoms + overhead_atoms
        k = min(int((cumulative + cost / 2) / total * n_slices), n_slices - 1)
        slices[k].append((seq_id, n_atoms))
        cumulative += cost
    return slices


def run(
    manifest_dir: Path,
    n_slices: int,
    overhead_atoms: int,
    output_dir: Path,
    pattern: str = "*.csv",
    only_ids: Path | None = None,
) -> None:
    rows = load_ok_rows(manifest_dir, pattern)
    if only_ids is not None:
        wanted = set(only_ids.read_text().split())
        rows = [r for r in rows if r[0] in wanted]
        print(f"{len(rows)} of {len(wanted)} IDs in {only_ids.name} are ok")
    if not rows:
        raise ValueError(f"no ok rows in {manifest_dir}")
    slices = cut_slices(rows, n_slices, overhead_atoms)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "slices.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["slice", "n_ids", "min_atoms", "max_atoms", "cost"])
        for k, piece in enumerate(slices):
            ids_path = output_dir / f"slice_{k:04d}.txt"
            ids_path.write_text("".join(f"{seq_id}\n" for seq_id, _ in piece))
            atoms = [n for _, n in piece]
            writer.writerow([
                k,
                len(piece),
                min(atoms, default=0),
                max(atoms, default=0),
                sum(n + overhead_atoms for n in atoms),
            ])
    print(f"{len(rows)} ok structures in {n_slices} slices under {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split prepared structures into ID lists for embedding jobs."
    )
    parser.add_argument("--manifest_dir", required=True, type=str,
                        help="Directory of prepare manifest CSVs.")
    parser.add_argument("--pattern", type=str, default="*.csv",
                        help="Glob of manifests to read (default '*.csv').")
    parser.add_argument("--n_slices", required=True, type=int,
                        help="Number of slices (embedding jobs).")
    parser.add_argument("--overhead_atoms", type=int, default=2000,
                        help="Fixed per-protein cost in atom units (default 2000).")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory for slice_<k>.txt and slices.csv.")
    parser.add_argument("--only_ids", type=str, default=None,
                        help="Keep only the IDs listed in this file, one per line"
                        " (for example rows re-prepared after a fix).")
    return parser


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    run(
        Path(_args.manifest_dir),
        _args.n_slices,
        _args.overhead_atoms,
        Path(_args.output_dir),
        _args.pattern,
        Path(_args.only_ids) if _args.only_ids else None,
    )
