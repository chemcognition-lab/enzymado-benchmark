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
