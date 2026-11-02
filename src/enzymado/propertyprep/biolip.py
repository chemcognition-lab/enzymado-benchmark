"""BioLiP binding dataset preparation."""

import pandas as pd

from . import chemistry, common, contracts


def prep_biolip_binding(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object] | None:
    """Prepare the final BioLiP binding dataset without processing other tasks."""
    print("\nProcessing BioLip Binding...")
    biol_path = paths.data_interim_dir / "biolip" / "biolip_cleaned_with_negatives.csv"
    if not biol_path.exists():
        return None

    biol = pd.read_csv(biol_path, low_memory=False)
    len_before = len(biol)
    st = common.start_stats(len_before)

    biol["protein_sequence"] = (
        biol["protein_sequence"].astype(str).str.strip().str.upper()
    )
    biol["uniprot_id"] = biol["uniprot_id"].replace("", pd.NA)

    seq_to_uniprot = (
        biol.dropna(subset=["uniprot_id"])
        .drop_duplicates(subset=["protein_sequence"])
        .set_index("protein_sequence")["uniprot_id"]
    )
    biol["uniprot_id"] = biol["uniprot_id"].fillna(
        biol["protein_sequence"].map(seq_to_uniprot)
    )

    biol["smiles"] = biol["smiles"].astype(str).str.strip()
    biol["ligand_id"] = biol["ligand_id"].replace("", pd.NA)
    biol["ligand_id_raw"] = biol["ligand_id_raw"].replace("", pd.NA)
    biol["ligand_name"] = biol["ligand_name"].replace("", pd.NA)

    smiles_to_id = (
        biol.dropna(subset=["ligand_id_raw"])
        .drop_duplicates(subset=["smiles"])
        .set_index("smiles")["ligand_id_raw"]
    )
    smiles_to_name = (
        biol.dropna(subset=["ligand_name"])
        .drop_duplicates(subset=["smiles"])
        .set_index("smiles")["ligand_name"]
    )

    biol["ligand_id_raw"] = biol["ligand_id_raw"].fillna(
        biol["smiles"].map(smiles_to_id)
    )
    biol["ligand_id"] = biol["ligand_id"].fillna(biol["ligand_id_raw"])
    biol["ligand_name"] = biol["ligand_name"].fillna(biol["smiles"].map(smiles_to_name))

    final_cols = [
        "uniprot_id",
        "protein_sequence",
        "ligand_id",
        "ligand_name",
        "smiles",
        "label",
    ]
    if not all(c in biol.columns for c in final_cols):
        return None

    biol_final = (
        biol[final_cols].rename(columns={"label": "binding"}).reset_index(drop=True)
    )
    _canonical_smiles, valid_mask = chemistry.canonicalize_smiles_column(
        biol_final["smiles"]
    )
    n_invalid = int((~valid_mask).sum())
    biol_final = biol_final[valid_mask].copy()
    canonical_mapping = pd.read_csv(
        paths.data_interim_dir / "biolip" / "biolip_smiles_mapping.csv",
        dtype=str,
    ).copy()
    canonical_by_raw = dict(
        zip(
            canonical_mapping["raw_smiles"],
            canonical_mapping["canonical_smiles"],
            strict=True,
        )
    )
    biol_final["smiles"] = biol_final["smiles"].map(canonical_by_raw.__getitem__)
    st["smiles_invalid_removed"] = n_invalid
    st["removed_by_reason"]["other"] += n_invalid

    out_dir = paths.data_processed_properties / "binding"
    out_dir.mkdir(parents=True, exist_ok=True)
    biol_final.to_csv(out_dir / "biolip_binding.csv", index=False)
    print("BioLip Binding saved.")
    return common.finish_stats(st, len(biol_final))
