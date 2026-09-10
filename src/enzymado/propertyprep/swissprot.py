"""Swiss-Prot EC-number and residue-site dataset preparation."""

import pandas as pd

from . import common, contracts, parsing


def prep_swissprot(
    paths: contracts.Paths,
    constants: contracts.Constants,
) -> dict[str, object]:
    stats = {}
    print("\nProcessing UniProt EC numbers...")
    swissprot_path = paths.data_raw_swissprot_dir / "swissprot_ec.tsv"
    if swissprot_path.exists():
        # Do not use index_col=1 to retain standard columns correctly
        ec = pd.read_csv(swissprot_path, sep="\t")
        st = common.start_stats(len(ec))
        ec_columns = ["Entry", "Sequence", "EC number"]
        if all(c in ec.columns for c in ec_columns):
            ec = ec[ec_columns].rename(
                columns={
                    "Entry": constants.uniprot_id,
                    "Sequence": constants.protein_sequence,
                    "EC number": "ec_number",
                }
            )
            len_b = len(ec)
            ec = ec.dropna(subset=["ec_number"])
            st["removed_by_reason"]["missing_values"] += len_b - len(ec)
            len_b = len(ec)
            ec = ec.drop_duplicates(subset=[constants.protein_sequence, "ec_number"])
            st["removed_by_reason"]["duplicates"] += len_b - len(ec)

            out_dir = paths.data_processed_properties / "ec_number"
            out_dir.mkdir(parents=True, exist_ok=True)
            ec.to_csv(out_dir / "swissprot_ec.csv", index=False)
            print("UniProt EC numbers saved.")
            stats["swissprot_ec"] = common.finish_stats(st, len(ec))

    # Active sites (UniProt)
    print("\nProcessing UniProt Active sites...")
    if swissprot_path.exists():
        active = pd.read_csv(swissprot_path, sep="\t")
        st = common.start_stats(len(active))
        active_columns = ["Entry", "EC number", "Sequence", "Active site"]
        if all(c in active.columns for c in active_columns):
            active = active[active_columns].rename(
                columns={
                    "Entry": constants.uniprot_id,
                    "EC number": "ec_number",
                    "Sequence": constants.protein_sequence,
                    "Active site": "active_site",
                }
            )
            len_b = len(active)
            active.dropna(subset=["active_site"], inplace=True)
            st["removed_by_reason"]["missing_values"] += len_b - len(active)
            len_b = len(active)
            active.drop_duplicates(
                subset=[constants.protein_sequence, "active_site"], inplace=True
            )
            st["removed_by_reason"]["duplicates"] += len_b - len(active)

            out_dir_interim = paths.data_interim_dir / "swissprot"
            out_dir_interim.mkdir(parents=True, exist_ok=True)
            active.to_csv(out_dir_interim / "swissprot_active_site.csv", index=False)

            active["active_site"] = active["active_site"].apply(
                parsing.parse_active_sites
            )

            out_dir = paths.data_processed_properties / "active_sites"
            out_dir.mkdir(parents=True, exist_ok=True)
            active.to_csv(out_dir / "swissprot_active_site.csv", index=False)
            print("UniProt Active sites saved.")
            stats["swissprot_active_site"] = common.finish_stats(st, len(active))

    # Binding sites (UniProt)
    print("\nProcessing UniProt Binding sites...")
    if swissprot_path.exists():
        binding_site = pd.read_csv(swissprot_path, sep="\t")
        st = common.start_stats(len(binding_site))
        binding_columns = ["Entry", "EC number", "Sequence", "Binding site"]
        if all(c in binding_site.columns for c in binding_columns):
            binding_site = binding_site[binding_columns].rename(
                columns={
                    "Entry": constants.uniprot_id,
                    "EC number": "ec_number",
                    "Sequence": constants.protein_sequence,
                    "Binding site": "binding_site",
                }
            )
            len_b = len(binding_site)
            binding_site.dropna(subset=["binding_site"], inplace=True)
            st["removed_by_reason"]["missing_values"] += len_b - len(binding_site)
            len_b = len(binding_site)
            binding_site.drop_duplicates(
                subset=[constants.protein_sequence, "binding_site"], inplace=True
            )
            st["removed_by_reason"]["duplicates"] += len_b - len(binding_site)
            binding_site.to_csv(
                out_dir_interim / "swissprot_binding_site.csv", index=False
            )

            parsed = binding_site["binding_site"].apply(parsing.parse_binding)
            binding_site["binding_positions"] = parsed.apply(lambda x: x[0])
            binding_site["binding_ligand"] = parsed.apply(lambda x: x[1])
            binding_site["binding_ligand_id"] = parsed.apply(lambda x: x[2])
            binding_site["binding_ignored_count"] = parsed.apply(lambda x: x[3])

            binding_site.to_csv(
                out_dir_interim / "swissprot_binding_site2.csv", index=False
            )

            binding_site.drop(
                columns=[
                    "binding_ignored_count",
                    "binding_site",
                    "binding_ligand",
                    "binding_ligand_id",
                    "ec_number",
                ],
                inplace=True,
            )

            out_dir = paths.data_processed_properties / "binding_site"
            out_dir.mkdir(parents=True, exist_ok=True)
            binding_site.to_csv(out_dir / "swissprot_binding_site.csv", index=False)
            print("UniProt Binding sites saved.")
            stats["swissprot_binding_site"] = common.finish_stats(st, len(binding_site))
    return stats
