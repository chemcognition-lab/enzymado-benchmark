"""Helpers for featurizing ligands, proteins, and loading saved embeddings."""

import pandas as pd
import requests
import numpy as np
from pathlib import Path
import re
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def get_canonical_iso(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def get_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)                             
    AllChem.EmbedMolecule(mol, randomSeed=42)         
    AllChem.UFFOptimizeMolecule(mol)                  
    asph     = rdMolDescriptors.CalcAsphericity(mol)
    ecc      = rdMolDescriptors.CalcEccentricity(mol)
    inertia  = rdMolDescriptors.CalcInertialShapeFactor(mol)
    pbf      = rdMolDescriptors.CalcPBF(mol)
    arr = np.array([asph, ecc, inertia, pbf], dtype=np.float32)
    return torch.from_numpy(arr)

def get_more(smi):
    m = Chem.MolFromSmiles(smi)
    m = Chem.AddHs(m)
    #Topological 
    num_rings      = m.GetRingInfo().NumRings()                         
    num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(m)         
    dist_mat       = Chem.GetDistanceMatrix(m)                          
    longest_path   = float(np.max(dist_mat))                            

    # Physical
    mol_wt = rdMolDescriptors.CalcExactMolWt(m)                         
    logp   = Crippen.MolLogP(m)                                         
    psa    = rdMolDescriptors.CalcTPSA(m)                                 

    arr = np.array([
        num_rings,
        num_bridgehead,
        longest_path,
        mol_wt,
        logp,
        psa
    ], dtype=np.float32)

    return torch.from_numpy(arr)

def get_enzprop(seq):
    analysis = ProteinAnalysis(seq)                        
    mw    = analysis.molecular_weight()                    
    pI    = analysis.isoelectric_point()                   
    gravy = analysis.gravy()                               
    arom  = analysis.aromaticity()                         
    inst  = analysis.instability_index()                   
    return torch.tensor([mw, pI, gravy, arom, inst], dtype=torch.float32)

def load_embeddings(file_path: str) -> dict[str, np.ndarray]:
    import safetensors
    tensors = {}
    with safetensors.safe_open(str(file_path), framework='pt') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).numpy()
    return tensors

class ArrayMap:
    """
    Maps discrete values (e.g., sequences, ids, smiles) → vectors/embeddings.
    - keys: list/array of keys (must NOT have duplicates)
    - values: np.ndarray with shape [N, D] aligned with keys
    """
    def __init__(self, keys, values: np.ndarray):
        keys = np.asarray(keys)
        if len(keys) != values.shape[0]:
            raise ValueError(f"len(keys)={len(keys)} != values.shape[0]={values.shape[0]}")
        # fast key→position index
        self.mapper = pd.Series(data=np.arange(len(keys)), index=keys)
        self.values = values

        # Check for duplicates in keys
        dups = self.mapper.index[self.mapper.index.duplicated()].tolist()
        if dups:
            raise ValueError(f"Found {len(dups)} duplicated keys: {dups[:5]}...")

    @classmethod
    def from_dict(cls, d: dict[str, np.ndarray]) -> "ArrayMap":
        keys = np.array(list(d.keys()))
        vals = np.stack(list(d.values()))
        return cls(keys, vals)

    @classmethod
    def load(cls, fname: str) -> "ArrayMap":
        data = load_embeddings(fname)  # dict: key -> np.ndarray
        return cls.from_dict(data)

    @property
    def keys(self):
        return list(self.mapper.index)

    def __call__(self, key_array: np.ndarray) -> np.ndarray:
        """
        key_array: np.ndarray of keys (strings or other hashables present in self.keys)
        return: embeddings with shape [len(key_array), D]
        """
        if not isinstance(key_array, np.ndarray):
            raise TypeError("expected np.ndarray as input!")
        idx = self.mapper[key_array].values  # positions
        return self.values[idx]
