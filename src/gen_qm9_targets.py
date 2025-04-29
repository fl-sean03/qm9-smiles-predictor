#!/usr/bin/env python3
"""
Generate a QM9 targets CSV from the raw qm9_clean.csv file.

Usage:
    python gen_qm9_targets.py \
        --input_csv  /home/sf2/School/5555CHEM/proj/7-t/data/qm9/qm9_clean.csv \
        --output_csv qm9_targets.csv
"""
import argparse
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Extract target properties from QM9 CSV")
    p.add_argument(
        "--input_csv", required=True,
        help="Path to qm9_clean.csv with columns including U0, U, H, G, Cv, homo, gap, mu, alpha"
    )
    p.add_argument(
        "--output_csv", required=True,
        help="Path where qm9_targets.csv will be written"
    )
    return p.parse_args()

def main():
    args = parse_args()
    # 1) Load the full QM9 clean CSV
    df = pd.read_csv(args.input_csv)
    # 2) Add orig_index to align with descriptor files
    df["orig_index"] = df.index
    # 3) Select exactly the nine target properties + orig_index
    targets = df[[
        "orig_index",
        "U0",    # internal energy at 0 K
        "U",     # internal energy at 298.15 K
        "H",     # enthalpy at 298.15 K
        "G",     # free energy at 298.15 K
        "Cv",    # heat capacity at 298.15 K
        "homo",  # HOMO energy (eV)
        "gap",   # HOMO–LUMO gap (eV)
        "mu",    # dipole moment (Debye)
        "alpha"  # polarizability (a0^3)
    ]]
    # 4) Write out
    targets.to_csv(args.output_csv, index=False)
    print(f"Saved {len(targets)} rows × {len(targets.columns)} cols to {args.output_csv}")

if __name__ == "__main__":
    main()

