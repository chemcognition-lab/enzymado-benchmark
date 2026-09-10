"""Molecular-string normalization used by property datasets."""

import pandas as pd
import rdkit.Chem
import rdkit.RDLogger


def canonicalize_smiles_column(
    smiles_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Validate and canonicalize a SMILES column.

    Returns (canonical_series, is_valid_mask). Rows where is_valid is False
    have SMILES that RDKit cannot parse and should be dropped.
    """
    rdkit.RDLogger.DisableLog("rdApp.*")
    canonical = []
    is_valid = []
    for smi in smiles_series.astype(str):
        mol = rdkit.Chem.MolFromSmiles(smi)
        if mol is None:
            canonical.append(smi)
            is_valid.append(False)
        else:
            canonical.append(rdkit.Chem.MolToSmiles(mol))
            is_valid.append(True)
    rdkit.RDLogger.EnableLog("rdApp.error")
    return (
        pd.Series(canonical, index=smiles_series.index),
        pd.Series(is_valid, index=smiles_series.index),
    )
