"""Prepare legacy BioLiP, FireProtDB, and BRENDA intermediate datasets."""

import hashlib
import json
import re

import numpy as np
import pandas as pd
from rdkit import Chem

from enzymado.paths_and_constants import Paths

EC_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[1-7]\."
    r"(?:\d+|-)\."
    r"(?:\d+|-)\."
    r"(?:\d+|-|n\d+)"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def has_explicit_ec_number(ec_values: pd.Series) -> pd.Series:
    """Return whether each value contains a complete, partial, or provisional EC."""
    return ec_values.astype("string").str.contains(EC_NUMBER_PATTERN, na=False)


def is_ion_smiles(smiles: str) -> bool:
    """
    Returns true if SMILES corresponds to a monoatomic ion.
    """
    ion_symbols = {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "MG",
        "CA",
        "SR",
        "BA",
        "ZN",
        "FE",
        "CU",
        "MN",
        "CO",
        "NI",
        "CD",
        "HG",
        "AL",
        "CR",
        "V",
        "MO",
        "W",
        "CL",
        "BR",
        "I",
        "F",
        "AG",
        "AU",
        "PT",
        "PB",
        "SN",
    }

    if pd.isna(smiles) or not smiles:
        return False

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")

    m = Chem.MolFromSmiles(str(smiles))

    RDLogger.EnableLog("rdApp.error")

    if m is None:
        return False

    # An ion is a single atom with no bonds.
    if m.GetNumAtoms() != 1:
        return False

    atom = m.GetAtomWithIdx(0)
    return atom.GetSymbol().upper() in ion_symbols


def add_negatives(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    seq_col: str = "protein_sequence",
    lig_col: str = "smiles",
) -> pd.DataFrame:
    """Append the frozen BioLiP negative assignments in manifest order."""
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = 1

    sequence_by_hash = {
        hashlib.sha256(str(sequence).encode()).hexdigest(): sequence
        for sequence in df[seq_col].drop_duplicates()
    }
    neg_df = pd.DataFrame(
        {
            seq_col: assignments["protein_sequence_sha256"].map(
                sequence_by_hash.__getitem__
            ),
            lig_col: assignments["smiles"],
        }
    )
    neg_df["label"] = 0

    for col in df.columns:
        if col not in neg_df.columns:
            neg_df[col] = np.nan
    neg_df = neg_df[df.columns]

    return pd.concat([df, neg_df], ignore_index=True)


def prep_biolip(p: Paths):
    """
    Dataprep for Biolip dataset.
    """
    biol_path = p.data_interim_dir / "biolip" / "BioLiP_nr.csv"
    if not biol_path.exists():
        return None

    biol = pd.read_csv(biol_path, low_memory=False)

    stats = {
        "datapoints_before_cleaning": len(biol),
        "removed_by_reason": {
            "missing_values": 0,
            "duplicates": 0,
            "invalid_sequence": 0,
            "was_not_enzyme": 0,
            "other": 0,
        },
    }

    len_before = len(biol)
    biol = biol[has_explicit_ec_number(biol["ec_number"])].copy()
    stats["removed_by_reason"]["was_not_enzyme"] += len_before - len(biol)

    len_before = len(biol)
    biol = biol[biol["uniprot_id"].notna()]
    stats["removed_by_reason"]["missing_values"] += len_before - len(biol)

    len_before = len(biol)
    biol = biol[biol["ligand_id"].notna()]
    stats["removed_by_reason"]["missing_values"] += len_before - len(biol)

    ccd_path = p.data_interim_dir / "biolip" / "Components-smiles-cactvs.smi"
    ccd = pd.read_csv(
        ccd_path,
        sep="\t",
        header=None,
        names=["smiles", "ccd_id", "ligand_name"],
        dtype=str,
    )
    ccd["ccd_id"] = ccd["ccd_id"].str.strip()
    ccd["smiles"] = ccd["smiles"].str.strip()
    ccd["ligand_name"] = ccd["ligand_name"].str.strip()

    # Lookups (diccs) to map ccd_id -> (smiles, name)
    ccd_exact = ccd.drop_duplicates("ccd_id").set_index("ccd_id")[
        ["smiles", "ligand_name"]
    ]
    ccd_upper = (
        ccd.assign(ccd_id_upper=ccd["ccd_id"].str.upper())
        .drop_duplicates("ccd_id_upper")
        .set_index("ccd_id_upper")[["smiles", "ligand_name"]]
    )

    # Prepare BioLiP IDs
    biol2 = biol.copy()
    biol2["ligand_id_raw"] = biol2["ligand_id"].astype(str).str.strip()

    # Define polymers that should NOT be mapped as small molecules
    polymer_tokens = {"dna", "rna", "pep", "peptide", "protein", "polypeptide"}
    is_polymer = biol2["ligand_id_raw"].str.lower().isin(polymer_tokens)

    # Initialize empty columns with object dtype to avoid
    # LossySetitemError when inserting strings
    biol2["smiles"] = pd.Series(np.nan, index=biol2.index, dtype=object)
    biol2["ligand_name"] = pd.Series(np.nan, index=biol2.index, dtype=object)
    biol2["map_status"] = "unmapped"

    # Mark polymers so they bypass the mapping completely
    biol2.loc[is_polymer, "map_status"] = "polymer_token_blocked"

    # 1. Map (case-sensitive) ONLY for non-polymers
    ok_to_map = ~is_polymer
    exact_matches = (
        biol2.loc[ok_to_map, "ligand_id_raw"].map(ccd_exact["smiles"]).notna()
    )
    idx_exact = biol2.index[ok_to_map & exact_matches]

    biol2.loc[idx_exact, "smiles"] = biol2.loc[idx_exact, "ligand_id_raw"].map(
        ccd_exact["smiles"]
    )
    biol2.loc[idx_exact, "ligand_name"] = biol2.loc[idx_exact, "ligand_id_raw"].map(
        ccd_exact["ligand_name"]
    )
    biol2.loc[idx_exact, "map_status"] = "exact"

    # 2. Fallback to UPPER (only if not mapped yet and not a polymer token)
    need = biol2["smiles"].isna()
    idx_upper = biol2.index[need & ok_to_map]

    upper_matches = (
        biol2.loc[idx_upper, "ligand_id_raw"]
        .str.upper()
        .map(ccd_upper["smiles"])
        .notna()
    )
    idx_upper_success = idx_upper[upper_matches]

    biol2.loc[idx_upper_success, "smiles"] = (
        biol2.loc[idx_upper_success, "ligand_id_raw"]
        .str.upper()
        .map(ccd_upper["smiles"])
    )
    biol2.loc[idx_upper_success, "ligand_name"] = (
        biol2.loc[idx_upper_success, "ligand_id_raw"]
        .str.upper()
        .map(ccd_upper["ligand_name"])
    )
    biol2.loc[idx_upper_success, "map_status"] = "rescued_upper"

    biol_smiles = biol2

    len_before = len(biol_smiles)
    biol_smiles.dropna(subset=["smiles"], inplace=True)
    stats["removed_by_reason"]["missing_values"] += len_before - len(biol_smiles)

    len_before = len(biol_smiles)
    is_ion = biol_smiles["smiles"].apply(is_ion_smiles)
    biol_filtered = biol_smiles[~is_ion].copy()
    stats["removed_by_reason"]["other"] += len_before - len(biol_filtered)

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    len_before = len(biol_filtered)
    parseable_mask = biol_filtered["smiles"].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    biol_filtered = biol_filtered[parseable_mask].copy()
    stats["removed_by_reason"]["other"] += len_before - len(biol_filtered)
    stats.setdefault("smiles_invalid_removed", 0)
    stats["smiles_invalid_removed"] += len_before - len(biol_filtered)
    RDLogger.EnableLog("rdApp.error")

    len_before = len(biol_filtered)
    biol_filtered = biol_filtered.drop_duplicates(
        subset=["protein_sequence", "smiles"]
    ).reset_index(drop=True)
    stats["removed_by_reason"]["duplicates"] += len_before - len(biol_filtered)

    assignment_path = p.data_interim_dir / "biolip" / "biolip_negative_assignments.csv"
    assignments = pd.read_csv(assignment_path, dtype=str).copy()
    neg_df = add_negatives(
        biol_filtered,
        assignments,
        seq_col="protein_sequence",
        lig_col="smiles",
    )

    out_dir = p.data_processed_datasets / "biolip_binding"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "biolip_cleaned_with_negatives.csv"

    neg_df.to_csv(out_path, index=False)
    print(f"Saved biolip cleaned dataset to {out_path}")

    stats["datapoints_after_cleaning"] = len(
        biol_filtered
    )  # use positive points for stats
    stats["datapoints_removed_total"] = (
        stats["datapoints_before_cleaning"] - stats["datapoints_after_cleaning"]
    )
    stats["removed_percentage"] = (
        (stats["datapoints_removed_total"] / stats["datapoints_before_cleaning"] * 100)
        if stats["datapoints_before_cleaning"]
        else 0
    )

    return {"amount_of_removed_data": stats}


def prep_fireprot(p: Paths) -> dict[str, object] | None:
    """Prepare DDG and DTM intermediate datasets from FireProtDB."""
    raw_path = p.data_raw_fireprotdb_dir / "fireprotdb_ec.tsv"

    if not raw_path.exists():
        return None

    fireprot = pd.read_csv(raw_path, sep="\t")

    stats = {
        "datapoints_before_cleaning": len(fireprot) * 2,
        "removed_by_reason": {
            "missing_values": 0,
            "duplicates": 0,
            "invalid_sequence": 0,
            "was_not_enzyme": 0,
            "other": 0,
        },
    }

    fireprot["DDG"] = pd.to_numeric(fireprot["DDG"], errors="coerce")
    fireprot["DTM"] = pd.to_numeric(fireprot["DTM"], errors="coerce")

    ddg_df = fireprot[fireprot["DDG"].notna()].copy()
    dtm_df = fireprot[fireprot["DTM"].notna()].copy()

    stats["removed_by_reason"]["missing_values"] += len(fireprot) - len(ddg_df)
    stats["removed_by_reason"]["missing_values"] += len(fireprot) - len(dtm_df)

    interim_dir = p.data_interim_dir / "fireprotdb"
    interim_dir.mkdir(parents=True, exist_ok=True)

    ddg_df.to_csv(interim_dir / "fireprotdb_ddg.csv", index=False)
    dtm_df.to_csv(interim_dir / "fireprotdb_dtm.csv", index=False)
    print(f"Saved DDG and DTM datasets to {interim_dir}")

    stats["datapoints_after_cleaning"] = len(ddg_df) + len(dtm_df)
    stats["datapoints_removed_total"] = (
        stats["datapoints_before_cleaning"] - stats["datapoints_after_cleaning"]
    )
    stats["removed_percentage"] = (
        stats["datapoints_removed_total"] / stats["datapoints_before_cleaning"] * 100
        if stats["datapoints_before_cleaning"]
        else 0
    )

    return {"amount_of_removed_data": stats}


def parse_ph(x: object) -> float | None:
    """Parse a pH range or single value into a float."""
    if pd.isna(x):
        return None

    x = str(x)
    if "-" in x:
        parts = x.split("-")
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2
        except ValueError:
            return None

    try:
        return float(x)
    except ValueError:
        return None


def prep_brenda(p: Paths):
    """
    Dataprep for phOpt (BRENDA).
    """
    brenda_path = p.data_dir / "raw" / "brenda" / "brenda_2025_1.json"

    if not brenda_path.exists():
        return None

    with open(brenda_path, "r") as f:
        data = json.load(f)

    brenda_data = data.get("data", {})
    results = []

    for ec, ec_data in brenda_data.items():
        if not isinstance(ec_data, dict) or "ph_optimum" not in ec_data:
            continue

        proteins = ec_data.get("protein", {})

        for entry in ec_data["ph_optimum"]:
            ph = entry.get("value")
            protein_ids = entry.get("proteins", [])

            for pid in protein_ids:
                p_info = proteins.get(str(pid), {})
                org = p_info.get("organism")
                accs = p_info.get("accessions", [])
                uniprot = accs[0] if accs else None

                results.append(
                    {
                        "ec": ec,
                        "protein_id": str(pid),
                        "organism": org,
                        "phopt": ph,
                        "uniprot_from_brenda": uniprot,
                    }
                )

    df_raw = pd.DataFrame(results)

    if df_raw.empty:
        return None

    stats = {
        "datapoints_before_cleaning": len(df_raw),
        "removed_by_reason": {
            "missing_values": 0,
            "duplicates": 0,
            "invalid_sequence": 0,
            "was_not_enzyme": 0,
            "other": 0,
        },
    }

    len_before = len(df_raw)
    df_raw["phopt"] = df_raw["phopt"].apply(parse_ph)
    df_raw = df_raw.dropna(subset=["phopt"])
    stats["removed_by_reason"]["missing_values"] += len_before - len(df_raw)

    len_before = len(df_raw)
    df_one = df_raw.groupby(
        ["ec", "protein_id", "organism", "uniprot_from_brenda"], as_index=False
    ).agg(
        phopt_median=("phopt", "median"),
        n_measurements=("phopt", "count"),
        phopt_std=("phopt", "std"),
    )
    stats["removed_by_reason"]["duplicates"] += len_before - len(df_one)

    interim_dir = p.data_interim_dir / "phopt"
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "brenda_phopt.csv"

    df_one.to_csv(out_path, index=False)
    print(f"Saved BRENDA phopt dataset to {out_path}")

    stats["datapoints_after_cleaning"] = len(df_one)
    stats["datapoints_removed_total"] = (
        stats["datapoints_before_cleaning"] - stats["datapoints_after_cleaning"]
    )
    stats["removed_percentage"] = (
        (stats["datapoints_removed_total"] / stats["datapoints_before_cleaning"] * 100)
        if stats["datapoints_before_cleaning"]
        else 0
    )

    return {"amount_of_removed_data": stats}


def main():
    p = Paths()

    stats = {}

    stats["biolip"] = prep_biolip(p)
    stats["fireprotdb"] = prep_fireprot(p)
    stats["brenda"] = prep_brenda(p)

    results_dir = p.results_dir / "dataprep_datacards"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "dataprep_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
