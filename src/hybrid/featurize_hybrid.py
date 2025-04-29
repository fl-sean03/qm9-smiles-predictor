#!/usr/bin/env python3
"""
src/hybrid/featurize_hybrid.py

Compute and save hybrid ECFP4 + Mordred features as a plain .npy array
for downstream memory-mapped training (shape: N × (nBits + D_desc)).
"""
import os
import argparse
import numpy as np
import pandas as pd
from utils import compute_ecfp4_array

def featurize_hybrid(smiles_list, X_desc, radius=2, nBits=1024):
    """
    Build hybrid feature matrix by concatenating ECFP bits and descriptors.

    Parameters:
      smiles_list: List[str] of length N_desc
      X_desc: np.ndarray of shape (N_desc, D_desc)
      radius: Morgan fingerprint radius
      nBits: fingerprint length (number of bits)

    Returns:
      X_hybrid: np.ndarray of shape (N_desc, nBits + D_desc)
    """
    N = len(smiles_list)
    fps = np.zeros((N, nBits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        fps[i] = compute_ecfp4_array(smi, radius=radius, nBits=nBits)
    return np.hstack([fps, X_desc])

def main():
    parser = argparse.ArgumentParser(
        description="Compute hybrid ECFP4+Mordred features and save as .npy"
    )
    parser.add_argument(
        "--smiles_csv", required=True,
        help="QM9 CSV with 'SMILES' column"
    )
    parser.add_argument(
        "--desc_csv", required=True,
        help="Cleaned mordred CSV with 'orig_index' and descriptor columns"
    )
    parser.add_argument(
        "--output_npy", required=True,
        help="Output path for hybrid .npy file"
    )
    parser.add_argument(
        "--radius", type=int, default=2,
        help="Morgan fingerprint radius (default: 2)"
    )
    parser.add_argument(
        "--nBits", type=int, default=1024,
        help="Number of bits for fingerprint (default: 1024)"
    )
    args = parser.parse_args()

    # 1) Load SMILES
    df_sm = pd.read_csv(args.smiles_csv)
    if 'SMILES' not in df_sm.columns:
        raise KeyError("smiles_csv must contain 'SMILES' column")

    # 2) Load descriptors
    df_desc = pd.read_csv(args.desc_csv)
    if 'orig_index' not in df_desc.columns:
        raise KeyError("desc_csv must contain 'orig_index' column")

    # 3) Map SMILES via orig_index array
    orig_idx = df_desc['orig_index'].astype(int).values
    smiles_list = df_sm['SMILES'].iloc[orig_idx].tolist()

    # 4) Descriptor matrix in same order
    X_desc = df_desc.drop(columns=['orig_index']).values.astype(np.float32)

    # 5) Compute hybrid features
    X_hybrid = featurize_hybrid(
        smiles_list, X_desc,
        radius=args.radius,
        nBits=args.nBits
    )

    # 6) Save as .npy for memory mapping
    os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
    np.save(args.output_npy, X_hybrid)
    print(f"[INFO] Saved hybrid features {X_hybrid.shape} → {args.output_npy}")

if __name__ == '__main__':
    main()

