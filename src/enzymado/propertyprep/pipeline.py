"""Sequential orchestration for all property dataset sources."""

import json

from . import biolip, brenda, catpred, contracts, fireprotdb, proteinea, swissprot


def run(paths: contracts.Paths, constants: contracts.Constants) -> None:
    """Prepare every available property source in the legacy execution order."""
    print("Running Propertyprep Script...")
    stats = {}

    biolip_stats = biolip.prep_biolip_binding(paths, constants)
    if biolip_stats is not None:
        stats["biolip_binding"] = biolip_stats

    ddg_stats = fireprotdb.prep_ddg(paths, constants)
    if ddg_stats is not None:
        stats["fireprotdb_ddg"] = ddg_stats

    dtm_stats = fireprotdb.prep_dtm(paths, constants)
    if dtm_stats is not None:
        stats["fireprotdb_dtm"] = dtm_stats

    phopt_stats = brenda.prep_phopt(paths, constants)
    if phopt_stats is not None:
        stats["brenda_phopt"] = phopt_stats

    kcat_stats = catpred.prep_kcat(paths, constants)
    if kcat_stats is not None:
        stats["catpred_kcat"] = kcat_stats

    km_stats = catpred.prep_km(paths, constants)
    if km_stats is not None:
        stats["catpred_km"] = km_stats

    stats.update(proteinea.prep_solubility(paths, constants))
    stats.update(swissprot.prep_swissprot(paths, constants))

    print("\nDone!")
    results_dir = paths.results_dir / "dataprep_datacards/propertyprep_datacards"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "propertyprep_stats.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(stats, stream, indent=2)
