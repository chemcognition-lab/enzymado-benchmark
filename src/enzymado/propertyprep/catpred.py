"""CatPred kcat and KM dataset preparation."""

import pandas as pd

from . import chemistry, common, contracts


def _prep_kinetic_property(
    paths: contracts.Paths,
    constants: contracts.Constants,
    *,
    display_name: str,
    source_file: str,
    source_smiles: str,
    source_target: str,
    target: str,
) -> dict[str, object] | None:
    print(f"\nProcessing CatPred {display_name}...")
    source_path = paths.data_raw_catpred_dir / source_file
    if not source_path.exists():
        return None

    raw = pd.read_csv(source_path, index_col=0)
    stats = common.start_stats(len(raw))
    source_columns = [
        "uniprot",
        "sequence",
        source_smiles,
        "temperature",
        "ph",
        source_target,
    ]
    if not all(column in raw.columns for column in source_columns):
        return None

    data = raw[source_columns].rename(
        columns={
            "uniprot": constants.uniprot_id,
            "sequence": constants.protein_sequence,
            source_smiles: constants.smiles,
            "temperature": "temperature",
            "ph": "ph",
            source_target: target,
        }
    )
    data[constants.protein_sequence] = (
        data[constants.protein_sequence].astype(str).str.strip().str.upper()
    )
    rows_before = len(data)
    data = data.dropna(subset=[constants.protein_sequence, constants.smiles, target])
    stats["removed_by_reason"]["missing_values"] += rows_before - len(data)
    canonical_smiles, valid_mask = chemistry.canonicalize_smiles_column(
        data[constants.smiles]
    )
    invalid_count = int((~valid_mask).sum())
    data = data[valid_mask].copy()
    data[constants.smiles] = canonical_smiles[valid_mask].values
    stats["smiles_invalid_removed"] = invalid_count
    stats["removed_by_reason"]["other"] += invalid_count

    destination = paths.data_processed_properties / target
    destination.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination / f"catpred_{target}.csv", index=False)
    print(f"CatPred {display_name} saved.")
    return common.finish_stats(stats, len(data))


def prep_kcat(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object] | None:
    """Prepare the CatPred kcat property."""
    return _prep_kinetic_property(
        paths,
        constants,
        display_name="kcat",
        source_file="CatPred_kcat.csv",
        source_smiles="reactant_smiles",
        source_target="log10kcat_max",
        target="kcat",
    )


def prep_kcat_expanded(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object]:
    """Prepare CatPred kcat keeping products, reactions and assay conditions.

    Unlike `prep_kcat`, rows missing pH, temperature or products are dropped so
    every row carries a complete reaction measured under known conditions. The
    target is that single measurement, not the per-pair maximum `prep_kcat` uses.
    """
    raw = pd.read_csv(paths.data_raw_catpred_dir / "CatPred_kcat.csv", index_col=0)
    stats = common.start_stats(len(raw))
    columns = {
        "uniprot": constants.uniprot_id,
        "sequence": constants.protein_sequence,
        "reactant_smiles": constants.smiles,
        "product_smiles": "product_smiles",
        "temperature": "temperature",
        "ph": "ph",
        "log10_value": "kcat",
    }
    data = raw[list(columns)].rename(columns=columns).copy()
    seqs = data[constants.protein_sequence].astype(str).str.strip().str.upper()
    data[constants.protein_sequence] = seqs
    required = [c for c in data.columns if c != constants.uniprot_id]
    rows_before = len(data)
    data = data.dropna(subset=required)
    stats["removed_by_reason"]["missing_values"] += rows_before - len(data)

    invalid_count = 0
    for column in (constants.smiles, "product_smiles"):
        canonical, valid = chemistry.canonicalize_smiles_column(data[column])
        invalid_count += int((~valid).sum())
        data = data[valid].copy()
        data[column] = canonical[valid].values
    stats["smiles_invalid_removed"] = invalid_count
    stats["removed_by_reason"]["other"] += invalid_count

    reaction = data[constants.smiles] + ">>" + data["product_smiles"]
    data.insert(4, "reaction_smiles", reaction)  # raw is exactly reactants>>products

    destination = paths.data_processed_properties / "kcat"
    destination.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination / "kcat_expanded.csv", index=False)
    return common.finish_stats(stats, len(data))


def prep_km(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object] | None:
    """Prepare the CatPred KM property."""
    return _prep_kinetic_property(
        paths,
        constants,
        display_name="KM",
        source_file="CatPred_km.csv",
        source_smiles="substrate_smiles",
        source_target="log10km_mean",
        target="km",
    )
