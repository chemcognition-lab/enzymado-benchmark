"""FireProtDB DDG and melting-temperature preparation."""

import pandas as pd

from . import common, contracts, parsing, uniprot


def _prep_mutation_property(
    paths: contracts.Paths,
    constants: contracts.Constants,
    *,
    source_target: str,
    target: str,
    output_dir: str,
    drop_missing_target: bool,
) -> dict[str, object] | None:
    source_path = paths.data_interim_dir / "fireprotdb" / f"fireprotdb_{target}.csv"
    if not source_path.exists():
        return None

    data = pd.read_csv(source_path, index_col=0).reset_index()
    stats = common.start_stats(len(data))
    source_columns = ["EXPERIMENT_ID", "UNIPROTKB", "SUBSTITUTION", source_target]
    data = data[source_columns]
    uniprot_ids = data["UNIPROTKB"].unique()

    sequences, _failed = uniprot.sequence_getter(
        uniprot_ids,
        checkpoint_path=str(
            paths.data_interim_dir / "sequence_checkpoint" / "uniprot_checkpoint.json"
        ),
        save_every=10,
    )
    data["sequence"] = data["UNIPROTKB"].map(sequences)
    data = data.rename(
        columns={
            "EXPERIMENT_ID": constants.dataset_id,
            "UNIPROTKB": constants.uniprot_id,
            "sequence": constants.protein_sequence,
            "SUBSTITUTION": "mutation",
            source_target: target,
        }
    )
    rows_before = len(data)
    data = data.dropna(subset=[constants.protein_sequence, "mutation"])
    stats["removed_by_reason"]["missing_values"] += rows_before - len(data)
    data["mut_protein_sequence"] = data.apply(
        lambda row: parsing.apply_substitutions(
            row["protein_sequence"], row["mutation"]
        ),
        axis=1,
    )
    data.rename(
        columns={
            "protein_sequence": "wildtype_sequence",
            "mut_protein_sequence": "protein_sequence",
        },
        inplace=True,
    )
    data = data[
        [
            "dataset_id",
            "uniprot_id",
            "wildtype_sequence",
            "mutation",
            "protein_sequence",
            target,
        ]
    ]
    if drop_missing_target:
        data = data.dropna(subset=[constants.protein_sequence, target])
    data = data.drop_duplicates(subset=[constants.protein_sequence, target])

    destination = paths.data_processed_properties / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination / f"fireprotdb_{target}.csv", index=False)
    print(f"FireProtDB {source_target} saved.")
    return common.finish_stats(stats, len(data))


def prep_ddg(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object] | None:
    """Prepare the FireProtDB DDG property."""
    print("\nProcessing FireProtDB DDG...")
    return _prep_mutation_property(
        paths,
        constants,
        source_target="DDG",
        target="ddg",
        output_dir="ddg_mut",
        drop_missing_target=True,
    )


def prep_dtm(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object] | None:
    """Prepare the FireProtDB melting-temperature property."""
    print("\nProcessing FireProtDB DTM...")
    return _prep_mutation_property(
        paths,
        constants,
        source_target="DTM",
        target="dtm",
        output_dir="dtm",
        drop_missing_target=False,
    )
