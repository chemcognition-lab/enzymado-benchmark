"""ProteinEA solubility dataset preparation."""

import pandas as pd

from . import common, contracts, uniprot


def prep_solubility(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object]:
    stats = {}
    print("\nProcessing ProteinEA Solubility...")
    proteinea_configs = [
        {
            "property_name": "solubility",
            "target_col": "solubility",
            "input_path": paths.data_processed_properties
            / "solubility"
            / "proteinea_solubility.csv",
            "output_path": paths.data_processed_properties
            / "solubility"
            / "proteinea_solubility.csv",
        },
    ]

    proteinea_remote_email = None
    checkpoint_dir = paths.data_interim_dir / "sequence_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for cfg in proteinea_configs:
        if not cfg["input_path"].exists():
            continue
        df = pd.read_csv(cfg["input_path"])
        st = common.start_stats(len(df))
        df[constants.protein_sequence] = (
            df[constants.protein_sequence].astype(str).str.strip().str.upper()
        )
        df["split"] = df["split"].astype(str).str.strip()
        df = df.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        len_b = len(df)
        df = df.dropna(
            subset=[constants.protein_sequence, cfg["target_col"], "split"]
        ).copy()
        st["removed_by_reason"]["missing_values"] += len_b - len(df)
        len_b = len(df)
        df = df.drop_duplicates(
            subset=[constants.protein_sequence, cfg["target_col"], "split"]
        ).copy()
        st["removed_by_reason"]["duplicates"] += len_b - len(df)

        unique_sequences = df[constants.protein_sequence].dropna().astype(str).unique()
        done, _failed = uniprot.sequence_to_enzyme_metadata(
            unique_sequences,
            checkpoint_path=str(
                checkpoint_dir
                / f"proteinea_{cfg['property_name']}_to_enzyme_checkpoint.json"
            ),
            swissprot_tsv_path=paths.data_raw_swissprot_dir / "swissprot_ec.tsv",
            save_every=25,
            use_remote_fallback=bool(proteinea_remote_email),
            email=proteinea_remote_email,
        )

        metadata_by_sequence = {
            seq: meta for seq, meta in done.items() if isinstance(meta, dict)
        }
        df[constants.uniprot_id] = df[constants.protein_sequence].map(
            lambda seq: (metadata_by_sequence.get(seq) or {}).get("accession")
        )
        df["ec_number"] = df[constants.protein_sequence].map(
            lambda seq: (metadata_by_sequence.get(seq) or {}).get("ec_number")
        )
        df["is_enzyme"] = df[constants.protein_sequence].map(
            lambda seq: bool((metadata_by_sequence.get(seq) or {}).get("is_enzyme"))
        )

        len_b = len(df)
        df = df[df["is_enzyme"]].copy()
        st["removed_by_reason"]["was_not_enzyme"] += len_b - len(df)
        cols_to_keep = [
            "dataset_id",
            constants.uniprot_id,
            constants.protein_sequence,
            "ec_number",
            cfg["target_col"],
            "split",
        ]
        if all(c in df.columns for c in cols_to_keep):
            df = df[cols_to_keep].reset_index(drop=True)
            cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cfg["output_path"], index=False)
            print(f"ProteinEA {cfg['property_name']} saved.")
            stats[f"proteinea_{cfg['property_name']}"] = common.finish_stats(
                st, len(df)
            )
    return stats
