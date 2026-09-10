"""BRENDA optimum-pH dataset preparation."""

import pandas as pd

from . import common, contracts, uniprot


def prep_phopt(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object] | None:
    print("\nProcessing BRENDA phopt...")
    brenda_path = paths.data_interim_dir / "phopt" / "brenda_phopt.csv"
    if brenda_path.exists():
        brenda_raw = pd.read_csv(brenda_path, index_col=0)
        st = common.start_stats(len(brenda_raw))
        unique_brenda = brenda_raw["uniprot_from_brenda"].unique()

        done, _failed = uniprot.sequence_getter(
            unique_brenda,
            checkpoint_path=str(
                paths.data_interim_dir
                / "sequence_checkpoint"
                / "uniprot_checkpoint.json"
            ),
            save_every=10,
        )
        cols = ["uniprot_from_brenda", "phopt_median", "n_measurements", "phopt_std"]
        brenda_opt = brenda_raw[cols]
        brenda_opt["sequence"] = brenda_opt["uniprot_from_brenda"].map(done)
        brenda_opt = brenda_opt.rename(
            columns={
                "uniprot_from_brenda": constants.uniprot_id,
                "sequence": constants.protein_sequence,
                "phopt": "phopt",
            }
        )
        brenda_opt = brenda_opt[
            [
                constants.uniprot_id,
                constants.protein_sequence,
                "phopt_median",
                "n_measurements",
                "phopt_std",
            ]
        ]
        len_b = len(brenda_opt)
        brenda_opt = brenda_opt.dropna(subset=[constants.protein_sequence])
        st["removed_by_reason"]["missing_values"] += len_b - len(brenda_opt)
        len_b = len(brenda_opt)
        brenda_opt = brenda_opt.drop_duplicates(
            subset=[
                constants.protein_sequence,
                "phopt_median",
                "n_measurements",
                "phopt_std",
            ]
        )
        st["removed_by_reason"]["duplicates"] += len_b - len(brenda_opt)

        out_dir = paths.data_processed_properties / "ph_opt"
        out_dir.mkdir(parents=True, exist_ok=True)
        brenda_opt.to_csv(out_dir / "brenda_phopt.csv", index=False)
        print("BRENDA phopt saved.")
        return common.finish_stats(st, len(brenda_opt))
    return None
