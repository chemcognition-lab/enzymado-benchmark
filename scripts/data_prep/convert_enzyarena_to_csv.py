#!/usr/bin/env python3
"""Convert EnzyArena campaign tables into Enzymado real-case task CSVs.

EnzyArena publishes one row per experimental campaign, with the individual
variants held in parallel Python-literal lists (``mutant_list``,
``value_list``, ``type_list``, ``mutant_sequence_list``). Enzymado expects one
row per measurement, so this script flattens them.

It also:

* drops the wild-type row of every campaign, because CatPred-DB (the source of
  the ``kcat`` and ``km`` property tasks) holds wild-type enzymes only, so the
  wild-type rows are the only ones that could leak into training;
* normalises units, since BRENDA reports KM in mM and SABIO-RK in M;
* takes log10 of the target, following the benchmark's convention for values
  spanning several orders of magnitude.

The raw value is not written out. It is ``10 ** log10_<param>`` exactly, and
leaving it out keeps a single numeric column in the CSV so that the splitter's
target auto-detection is unambiguous.

Usage:

    python convert_enzyarena_to_csv.py --input-dir . --output-dir .
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import math
from pathlib import Path

import pandas as pd

SOURCE_FILES = {
    "brenda": "Benchmarking_multiple_models_from_brenda.csv",
    "sabio": "Benchmarking_multiple_models_from_sabio-rk.csv",
}

# The kinetic parameter is not given in a column of its own; it is implied by
# the unit. The scale factor converts every value to one unit per parameter:
# 1/s for kcat, mM for KM, 1/(mM*s) for kcat/KM.
UNIT_MAP = {
    "1/s": ("kcat", 1.0),
    "s^(-1)": ("kcat", 1.0),
    "mM": ("km", 1.0),
    "M": ("km", 1000.0),
    "1/mMs-1": ("kcat_over_km", 1.0),
    "M^(-1)*s^(-1)": ("kcat_over_km", 1e-3),
}

WANTED_PARAMS = ("kcat", "km")


def wildtype_group(sequence: str) -> str:
    """Return a short stable id for a wild-type sequence, for grouped splits."""
    digest = hashlib.sha1(str(sequence).encode()).hexdigest()
    return f"wt{digest[:10]}"


def parse_campaign(row, source, campaign_index, stats):
    """Yield one record per mutant measurement in a single campaign row."""
    parameter, scale = UNIT_MAP[row["unit"]]
    if parameter not in WANTED_PARAMS:
        return

    mutations = ast.literal_eval(row["mutant_list"])
    values = ast.literal_eval(row["value_list"])
    kinds = ast.literal_eval(row["type_list"])
    sequences = {
        item[0]: item[1] for item in ast.literal_eval(row["mutant_sequence_list"])
    }

    if not len(mutations) == len(values) == len(kinds):
        stats["campaigns_with_ragged_lists"] += 1
        return

    smiles = row.get("Substrate_smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        stats["campaigns_without_smiles"] += 1
        return

    group = wildtype_group(row["wildtype_sequence"])
    campaign_id = f"{source}{campaign_index:04d}"

    for mutation, value, kind in zip(mutations, values, kinds):
        if kind != "mutant":
            stats["wildtype_rows_dropped"] += 1
            continue

        sequence = sequences.get(mutation)
        if not isinstance(sequence, str) or not sequence.strip():
            stats["rows_without_sequence"] += 1
            continue

        try:
            scaled = float(value) * scale
        except (TypeError, ValueError):
            stats["rows_with_unparseable_value"] += 1
            continue

        if not math.isfinite(scaled) or scaled <= 0:
            stats["rows_non_positive_cannot_log"] += 1
            continue

        yield {
            "parameter": parameter,
            "dataset_id": f"ea_{campaign_id}_{mutation}",
            "protein_sequence": sequence.strip().upper(),
            "smiles": smiles.strip(),
            "temperature": row.get("temperature"),
            "ph": row.get("pH"),
            f"log10_{parameter}": round(math.log10(scaled), 6),
            "mutation": mutation,
            "uniprot_id": row.get("uniprot_id"),
            "wt_group": group,
            "campaign_id": campaign_id,
            "source_db": source,
        }


def deduplicate_ids(frame, stats):
    """Append a counter to any dataset_id that occurs more than once."""
    counts = collections.Counter()
    unique_ids = []
    for value in frame["dataset_id"]:
        counts[value] += 1
        unique_ids.append(value if counts[value] == 1 else f"{value}_{counts[value]}")
    stats["dataset_ids_disambiguated"] += sum(n - 1 for n in counts.values() if n > 1)
    frame["dataset_id"] = unique_ids
    return frame


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="directory holding the two EnzyArena CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory to write enzyarena_<param>.csv into",
    )
    args = parser.parse_args()

    stats = collections.Counter()
    records = []

    for source, filename in SOURCE_FILES.items():
        path = args.input_dir / filename
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")
        frame = pd.read_csv(path)
        print(f"read {filename}: {len(frame)} campaigns")
        for index, row in frame.reset_index(drop=True).iterrows():
            records.extend(parse_campaign(row, source, index, stats))

    everything = pd.DataFrame.from_records(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for parameter in WANTED_PARAMS:
        target = f"log10_{parameter}"
        subset = everything[everything["parameter"] == parameter].drop(
            columns=["parameter"]
        )
        subset = subset.dropna(axis=1, how="all")
        subset = subset[
            [c for c in subset.columns if not c.startswith("log10_") or c == target]
        ]
        subset = deduplicate_ids(subset.reset_index(drop=True), stats)

        out = args.output_dir / f"enzyarena_{parameter}.csv"
        subset.to_csv(out, index=False)

        print(f"\n== enzyarena_{parameter}.csv")
        print(f"   rows                 {len(subset)}")
        print(f"   unique sequences     {subset.protein_sequence.nunique()}")
        print(f"   unique substrates    {subset.smiles.nunique()}")
        print(f"   wild-type groups     {subset.wt_group.nunique()}")
        print(f"   campaigns            {subset.campaign_id.nunique()}")
        print(
            f"   {target} range       "
            f"{subset[target].min():.3f} .. {subset[target].max():.3f}"
        )
        print(f"   temperature missing  {int(subset.temperature.isna().sum())}")
        print(f"   ph missing           {int(subset.ph.isna().sum())}")
        print(f"   written to           {out}")

    print("\n== filtering summary")
    for key in sorted(stats):
        print(f"   {key:34s} {stats[key]}")


if __name__ == "__main__":
    main()
