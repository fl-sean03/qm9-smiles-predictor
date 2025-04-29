# src/hybrid/utils.py
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def compute_ecfp4_array(smiles: str, radius: int = 2, nBits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def compute_ecfp4_list(smiles_list: list[str], radius: int = 2, nBits: int = 1024) -> np.ndarray:
    N = len(smiles_list)
    fps = np.zeros((N, nBits), dtype=np.int8)
    for i, smi in enumerate(smiles_list):
        fps[i] = compute_ecfp4_array(smi, radius, nBits)
    return fps

