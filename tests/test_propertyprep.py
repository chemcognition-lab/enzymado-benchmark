"""Characterization tests for the property-preparation pipeline."""

import hashlib
import json
from pathlib import Path
import types

import pandas as pd
import scripts.data_prep.propertyprep as propertyprep

_OUTPUT_PATHS = (
    "data/interim/swissprot/swissprot_active_site.csv",
    "data/interim/swissprot/swissprot_binding_site.csv",
    "data/interim/swissprot/swissprot_binding_site2.csv",
    "data/processed/properties/active_sites/swissprot_active_site.csv",
    "data/processed/properties/binding/biolip_binding.csv",
    "data/processed/properties/binding_site/swissprot_binding_site.csv",
    "data/processed/properties/ddg_mut/fireprotdb_ddg.csv",
    "data/processed/properties/dtm/fireprotdb_dtm.csv",
    "data/processed/properties/ec_number/swissprot_ec.csv",
    "data/processed/properties/kcat/catpred_kcat.csv",
    "data/processed/properties/km/catpred_km.csv",
    "data/processed/properties/ph_opt/brenda_phopt.csv",
    "data/processed/properties/solubility/proteinea_solubility.csv",
    "results/dataprep_datacards/propertyprep_datacards/propertyprep_stats.json",
)

_EXPECTED_OUTPUT_HASHES = {
    "data/interim/swissprot/swissprot_active_site.csv": (
        "c4163f0f020f572c854f97f892e9be3ede2323eb31be9e8af08c69fcb9ff3e7a"
    ),
    "data/interim/swissprot/swissprot_binding_site.csv": (
        "bacc74c1ea801717d5175691386fa37254365775bd641c983561487f73bd4c35"
    ),
    "data/interim/swissprot/swissprot_binding_site2.csv": (
        "03abac60ac6197fb9efbd8852ce6df55b499c2f78a0cf2765ee5b6fe36a1f2df"
    ),
    "data/processed/properties/active_sites/swissprot_active_site.csv": (
        "99673aaa0df8184048e0544f3d3f884128787402aa3514bf751ecb40a294fefa"
    ),
    "data/processed/properties/binding/biolip_binding.csv": (
        "0e1d19c15ca4e31471ada9f953713faed6c6d9d96d7bfa3c930907518dd30199"
    ),
    "data/processed/properties/binding_site/swissprot_binding_site.csv": (
        "1f35df48a5dec4ed71ed7474886a8542447e0705434f586c9ba3cff35c1ebda1"
    ),
    "data/processed/properties/ddg_mut/fireprotdb_ddg.csv": (
        "2b9220effc8ebbb3391a08a86fe798b03d3fe4047b579202bca9f82d01a9bca3"
    ),
    "data/processed/properties/dtm/fireprotdb_dtm.csv": (
        "5ce134052b39f6175adb101e2895acb1a49748137042d68447f06eb6dc11d7e3"
    ),
    "data/processed/properties/ec_number/swissprot_ec.csv": (
        "1bae0a4394abeade454c804ea1eb6c84fe69b5d4e3542e3b433c69e984324280"
    ),
    "data/processed/properties/kcat/catpred_kcat.csv": (
        "116b908132cedb6237695f6a789e17d3e12832be6765b27f04c8f05eec74e1b3"
    ),
    "data/processed/properties/km/catpred_km.csv": (
        "a9197ca87687073e8613e4dd8c7153dd47eb0af19d3b6ffd98608c558a663ad7"
    ),
    "data/processed/properties/ph_opt/brenda_phopt.csv": (
        "7f2da8a9b8569d1c4e52cfba1979a2745d89bcdbc7b6ef250cff408a251f6474"
    ),
    "data/processed/properties/solubility/proteinea_solubility.csv": (
        "6f92d4f8c1d8bc9b82aa280ad35c807f9e1763b0359c33eb6735c2cf9c8adde4"
    ),
    "results/dataprep_datacards/propertyprep_datacards/propertyprep_stats.json": (
        "9bccd943dfe3be3d9e5d862016f1692ea10106ae6e080c5c7afeaa2d650fc79f"
    ),
}


def _paths(root: Path) -> types.SimpleNamespace:
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    return types.SimpleNamespace(
        data_interim_dir=data_dir / "interim",
        data_processed_datasets=processed_dir / "datasets",
        data_processed_properties=processed_dir / "properties",
        data_raw_catpred_dir=raw_dir / "catpred",
        data_raw_swissprot_dir=raw_dir / "swissprot",
        results_dir=root / "results",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    index: bool = False,
    sep: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=index, sep=sep)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: _hash(root / relative) for relative in paths}


def _build_fixture(root: Path) -> types.SimpleNamespace:
    paths = _paths(root)
    _write_csv(
        paths.data_processed_datasets
        / "biolip_binding/biolip_cleaned_with_negatives.csv",
        [
            {
                "protein_sequence": " acd ",
                "uniprot_id": "P1",
                "smiles": "C(C)O",
                "ligand_id": "",
                "ligand_id_raw": "CHEBI:16236",
                "ligand_name": "ethanol",
                "label": 1,
            },
            {
                "protein_sequence": "ACD",
                "uniprot_id": "",
                "smiles": "CCO",
                "ligand_id": "",
                "ligand_id_raw": "",
                "ligand_name": "",
                "label": 0,
            },
            {
                "protein_sequence": "AAA",
                "uniprot_id": "P2",
                "smiles": "not-smiles",
                "ligand_id": "L2",
                "ligand_id_raw": "L2",
                "ligand_name": "invalid",
                "label": 1,
            },
        ],
    )
    _write_csv(
        paths.data_interim_dir / "biolip/biolip_smiles_mapping.csv",
        [
            {"raw_smiles": "C(C)O", "canonical_smiles": "CCO"},
            {"raw_smiles": "CCO", "canonical_smiles": "CCO"},
        ],
    )
    _write_csv(
        paths.data_interim_dir / "fireprotdb/fireprotdb_ddg.csv",
        [
            {"EXPERIMENT_ID": 1, "UNIPROTKB": "P1", "SUBSTITUTION": "A1V", "DDG": 1.5},
            {"EXPERIMENT_ID": 2, "UNIPROTKB": "P1", "SUBSTITUTION": "A1V", "DDG": 1.5},
            {"EXPERIMENT_ID": 3, "UNIPROTKB": "P2", "SUBSTITUTION": None, "DDG": 2.0},
        ],
        index=True,
    )
    _write_csv(
        paths.data_interim_dir / "fireprotdb/fireprotdb_dtm.csv",
        [
            {"EXPERIMENT_ID": 4, "UNIPROTKB": "P1", "SUBSTITUTION": "D3K", "DTM": 3.5},
            {"EXPERIMENT_ID": 5, "UNIPROTKB": "P1", "SUBSTITUTION": "D3K", "DTM": 3.5},
        ],
        index=True,
    )
    _write_csv(
        paths.data_interim_dir / "phopt/brenda_phopt.csv",
        [
            {
                "uniprot_from_brenda": "P1",
                "phopt_median": 7.0,
                "n_measurements": 2,
                "phopt_std": 0.1,
            },
            {
                "uniprot_from_brenda": "P1",
                "phopt_median": 7.0,
                "n_measurements": 2,
                "phopt_std": 0.1,
            },
        ],
        index=True,
    )
    _write_csv(
        paths.data_raw_catpred_dir / "CatPred_kcat.csv",
        [
            {
                "uniprot": "P1",
                "sequence": " acd ",
                "reactant_smiles": "C(C)O",
                "temperature": 25,
                "ph": 7,
                "log10kcat_max": 1.2,
            },
            {
                "uniprot": "P2",
                "sequence": "AAA",
                "reactant_smiles": "not-smiles",
                "temperature": 30,
                "ph": 8,
                "log10kcat_max": 2.0,
            },
        ],
        index=True,
    )
    _write_csv(
        paths.data_raw_catpred_dir / "CatPred_km.csv",
        [
            {
                "uniprot": "P1",
                "sequence": "acd",
                "substrate_smiles": "C(C)O",
                "temperature": 25,
                "ph": 7,
                "log10km_mean": -1.0,
            },
            {
                "uniprot": "P2",
                "sequence": "AAA",
                "substrate_smiles": "not-smiles",
                "temperature": 30,
                "ph": 8,
                "log10km_mean": -2.0,
            },
        ],
        index=True,
    )
    _write_csv(
        paths.data_processed_properties / "solubility/proteinea_solubility.csv",
        [
            {
                "dataset_id": "S1",
                "protein_sequence": "acd",
                "solubility": 1,
                "split": "train",
            },
            {
                "dataset_id": "S2",
                "protein_sequence": "AAA",
                "solubility": 0,
                "split": "test",
            },
        ],
    )
    _write_csv(
        paths.data_raw_swissprot_dir / "swissprot_ec.tsv",
        [
            {
                "Entry": "P1",
                "Sequence": "ACD",
                "EC number": "1.2.3.4",
                "Active site": 'ACT_SITE 2; /note="Proton donor"',
                "Binding site": (
                    'BINDING 2; /ligand="ATP"; /ligand_id="ChEBI:CHEBI:15422"'
                ),
            },
            {
                "Entry": "P1_DUP",
                "Sequence": "ACD",
                "EC number": "1.2.3.4",
                "Active site": 'ACT_SITE 2; /note="Proton donor"',
                "Binding site": (
                    'BINDING 2; /ligand="ATP"; /ligand_id="ChEBI:CHEBI:15422"'
                ),
            },
            {
                "Entry": "P2",
                "Sequence": "AAA",
                "EC number": None,
                "Active site": None,
                "Binding site": None,
            },
        ],
        sep="\t",
    )
    checkpoint_dir = paths.data_interim_dir / "sequence_checkpoint"
    _write_json(
        checkpoint_dir / "uniprot_checkpoint.json",
        {"done": {"P1": "ACD", "P2": "AAA"}, "failed": {}},
    )
    _write_json(
        checkpoint_dir / "proteinea_solubility_to_enzyme_checkpoint.json",
        {
            "done": {
                "ACD": {
                    "accession": "P1",
                    "ec_number": "1.2.3.4",
                    "is_enzyme": True,
                    "match_source": "fixture",
                },
                "AAA": {
                    "accession": "P2",
                    "ec_number": None,
                    "is_enzyme": False,
                    "match_source": "fixture",
                },
            },
            "failed": {},
        },
    )
    return paths


def test_propertyprep_parsers_are_stable() -> None:
    """Lock mutation, feature parsing, and SMILES canonicalization behavior."""
    assert propertyprep.apply_substitutions("ACDE", "A1V;D3K") == "VCKE"
    assert propertyprep.parse_active_sites("ACT_SITE 9; ACT_SITE 3") == [9, 3]
    assert propertyprep.parse_binding(
        'BINDING 2..3; /ligand="ATP"; /ligand_id="CHEBI:15422"'
    ) == ([2, 3], ["ATP", "ATP"], ["CHEBI:15422", "CHEBI:15422"], 0)

    canonical, valid = propertyprep._canonicalize_smiles_column(
        pd.Series(["C(C)O", "not-smiles"])
    )
    assert canonical.tolist() == ["CCO", "not-smiles"]
    assert valid.tolist() == [True, False]


def test_propertyprep_pipeline_matches_frozen_outputs(
    tmp_path: Path,
) -> None:
    """Lock every source route, output byte order, and raw-data immutability."""
    paths = _build_fixture(tmp_path)
    raw_paths = tuple(
        str(path.relative_to(tmp_path))
        for path in sorted((tmp_path / "data/raw").rglob("*"))
        if path.is_file()
    )
    raw_before = _snapshot(tmp_path, raw_paths)

    propertyprep._UNIPROT_SEQUENCE_INDEX_CACHE.clear()
    propertyprep.main(paths=paths)

    assert _snapshot(tmp_path, raw_paths) == raw_before
    assert _snapshot(tmp_path, _OUTPUT_PATHS) == _EXPECTED_OUTPUT_HASHES
