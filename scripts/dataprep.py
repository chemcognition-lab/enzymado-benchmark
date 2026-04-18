import json

import numpy as np
import pandas as pd
from rdkit import Chem

from src.paths_and_constants import Paths


def is_ion_smiles(smiles: str) -> bool:
    """
    Returns true if SMILES corresponds to a monoatomic ion.
    """
    ion_symbols = {
        "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA",
        "ZN", "FE", "CU", "MN", "CO", "NI", "CD", "HG", "AL",
        "CR", "V", "MO", "W", "CL", "BR", "I", "F", "AG",
        "AU", "PT", "PB", "SN",
    }
    
    if pd.isna(smiles) or not smiles:
        return False

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')

    m = Chem.MolFromSmiles(str(smiles))
    
    RDLogger.EnableLog('rdApp.error')

    if m is None:
        return False

    # Ion = un solo átomo, sin enlaces
    if m.GetNumAtoms() != 1:
        return False

    atom = m.GetAtomWithIdx(0)
    return atom.GetSymbol().upper() in ion_symbols


def add_negatives(
    df: pd.DataFrame,
    seq_col: str = "protein_sequence",
    lig_col: str = "smiles",
    neg_per_pos: int = 1,
    seed: int = 42,
    max_tries: int = 1000,
) -> pd.DataFrame:
    """
    Add negative examples generated randomly.
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    if "label" not in df.columns:
        df["label"] = 1

    ligands_filtered = set(df[lig_col].dropna().unique())

    negatives = set()

    for p, grp in df.groupby(seq_col):
        pos_ligands_p = set(grp[lig_col].dropna().unique())
        if not pos_ligands_p:
            continue

        candidate_ligands = list(ligands_filtered - pos_ligands_p)
        n_neg = min(neg_per_pos * len(pos_ligands_p), len(candidate_ligands))
        
        if not candidate_ligands:
            continue

        chosen = rng.choice(candidate_ligands, size=n_neg, replace=False)

        for l_neg in chosen:
            negatives.add((p, l_neg))

    neg_df = pd.DataFrame(list(negatives), columns=[seq_col, lig_col])
    neg_df["label"] = 0

    # columns alignment
    for col in df.columns:
        if col not in neg_df.columns:
            neg_df[col] = np.nan
    neg_df = neg_df[df.columns]

    out_df = pd.concat([df, neg_df], ignore_index=True)
    return out_df


def prep_biolip(p: Paths):
    """
    Dataprep for Biolip dataset.
    """
    biol_path = p.data_interim_dir / "biolip" / "BioLiP_nr.csv"
    if not biol_path.exists():
        return
        
    biol = pd.read_csv(biol_path, low_memory=False)
    biol = biol[biol['ec_number'].notna()]
    biol = biol[biol['uniprot_id'].notna()]
    biol = biol[biol['ligand_id'].notna()]

    ccd_path = p.data_interim_dir / "biolip" / "Components-smiles-cactvs.smi"
    ccd = pd.read_csv(
        ccd_path,
        sep="\t",
        header=None,
        names=["smiles", "ccd_id", "ligand_name"],
        dtype=str
    )
    ccd["ccd_id"] = ccd["ccd_id"].str.strip()
    ccd["smiles"] = ccd["smiles"].str.strip()
    ccd["ligand_name"] = ccd["ligand_name"].str.strip()

    # Lookups (diccs) to map ccd_id -> (smiles, name)
    ccd_exact = ccd.drop_duplicates("ccd_id").set_index("ccd_id")[["smiles", "ligand_name"]]
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

    # Initialize empty columns with object dtype to avoid LossySetitemError when inserting strings
    biol2["smiles"] = pd.Series(np.nan, index=biol2.index, dtype=object)
    biol2["ligand_name"] = pd.Series(np.nan, index=biol2.index, dtype=object)
    biol2["map_status"] = "unmapped"

    # Mark polymers so they bypass the mapping completely
    biol2.loc[is_polymer, "map_status"] = "polymer_token_blocked"

    # 1. Map (case-sensitive) ONLY for non-polymers
    ok_to_map = ~is_polymer
    exact_matches = biol2.loc[ok_to_map, "ligand_id_raw"].map(ccd_exact["smiles"]).notna()
    idx_exact = biol2.index[ok_to_map & exact_matches]

    biol2.loc[idx_exact, "smiles"] = biol2.loc[idx_exact, "ligand_id_raw"].map(ccd_exact["smiles"])
    biol2.loc[idx_exact, "ligand_name"] = biol2.loc[idx_exact, "ligand_id_raw"].map(ccd_exact["ligand_name"])
    biol2.loc[idx_exact, "map_status"] = "exact"

    # 2. Fallback to UPPER (only if not mapped yet and not a polymer token)
    need = biol2["smiles"].isna()
    idx_upper = biol2.index[need & ok_to_map]

    upper_matches = biol2.loc[idx_upper, "ligand_id_raw"].str.upper().map(ccd_upper["smiles"]).notna()
    idx_upper_success = idx_upper[upper_matches]

    biol2.loc[idx_upper_success, "smiles"] = biol2.loc[idx_upper_success, "ligand_id_raw"].str.upper().map(ccd_upper["smiles"])
    biol2.loc[idx_upper_success, "ligand_name"] = biol2.loc[idx_upper_success, "ligand_id_raw"].str.upper().map(ccd_upper["ligand_name"])
    biol2.loc[idx_upper_success, "map_status"] = "rescued_upper"

    biol_smiles = biol2
    biol_smiles.dropna(subset=["smiles"], inplace=True)

    is_ion = biol_smiles["smiles"].apply(is_ion_smiles)
    biol_filtered = biol_smiles[~is_ion].copy()

    neg_df = add_negatives(
        biol_filtered,
        seq_col="protein_sequence",
        lig_col="smiles",
        neg_per_pos=1,
        seed=42,
        max_tries=1000,
    )
    
    out_dir = p.data_processed_datasets
    out_path = out_dir / "biolip_cleaned_with_negatives.csv"
    
    neg_df.to_csv(out_path, index=False)
    print(f"Saved biolip cleaned dataset to {out_path}")


def prep_fireprot(p: Paths):
    """
    Dataprep for DDG and ThermoStab (FireProtDB).
    """
    raw_path = p.data_raw_fireprotdb_dir / "fireprotdb_ec.tsv"
    
    if not raw_path.exists():
        return

    fireprot = pd.read_csv(raw_path, sep="\t")

    fireprot["DDG"] = pd.to_numeric(fireprot["DDG"], errors="coerce")
    fireprot["DTM"] = pd.to_numeric(fireprot["DTM"], errors="coerce")

    ddg_df = fireprot[fireprot["DDG"].notna()].copy()
    dtm_df = fireprot[fireprot["DTM"].notna()].copy()

    interim_dir = p.data_interim_dir / "fireprotdb"
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    ddg_df.to_csv(interim_dir / "fireprotdb_ddg.csv", index=False)
    dtm_df.to_csv(interim_dir / "fireprotdb_dtm.csv", index=False)
    print(f"Saved DDG and DTM datasets to {interim_dir}")


def parse_ph(x):
    """
    Parse pH string ranges or single values into floats.
    """
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
        return

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

                results.append({
                    "ec": ec,
                    "protein_id": str(pid),
                    "organism": org,
                    "phopt": ph,
                    "uniprot_from_brenda": uniprot,
                })

    df_raw = pd.DataFrame(results)
    
    if df_raw.empty:
        return

    df_raw["phopt"] = df_raw["phopt"].apply(parse_ph)
    df_raw = df_raw.dropna(subset=["phopt"])

    df_one = (
        df_raw.groupby(["ec", "protein_id", "organism", "uniprot_from_brenda"], as_index=False)
          .agg(
              phopt_median=("phopt", "median"),
              n_measurements=("phopt", "count"),
              phopt_std=("phopt", "std")
          )
    )
    
    interim_dir = p.data_interim_dir / "phopt"
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "brenda_phopt.csv"
    
    df_one.to_csv(out_path, index=False)
    print(f"Saved BRENDA phopt dataset to {out_path}")


def main():
    p = Paths()

    prep_biolip(p)
    prep_fireprot(p)
    prep_brenda(p)


if __name__ == "__main__":
    main()
