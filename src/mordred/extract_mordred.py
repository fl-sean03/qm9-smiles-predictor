#!/usr/bin/env python3
"""
Extract Mordred descriptors from a QM9 CSV with SMILES, with resume support and optional row limit.

Usage:
    python extract_mordred.py \
        --input_csv qm9_clean.csv \
        --output_csv mordred_raw.csv \
        --chunksize 5000 \
        [--max_rows 10000]

Options:
  --input_csv   Path to QM9 CSV with a 'SMILES' column.
  --output_csv  Path to write raw Mordred descriptors.
  --chunksize   Rows per chunk (default: 5000).
  --max_rows    If set, only process the first N rows (for debugging).
"""
import argparse
import numpy as np
import pandas as pd
import warnings
from rdkit import Chem
from mordred import Calculator, descriptors
import logging
import sys
import os

# suppress pandas dtype warnings
warnings.simplefilter("ignore", FutureWarning)

def parse_args():
    p = argparse.ArgumentParser(description="Compute Mordred features with resume and max_rows")
    p.add_argument("--input_csv",  required=True, help="QM9 CSV with at least a 'SMILES' column")
    p.add_argument("--output_csv", required=True, help="Where to save the raw Mordred CSV")
    p.add_argument("--chunksize",  type=int, default=5000, help="Number of rows per chunk")
    p.add_argument("--max_rows",   type=int, default=None, help="Only process the first N rows (debug mode)")
    return p.parse_args()

def main():
    args = parse_args()

    # ensure output directory exists
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s: %(message)s")
    calc = Calculator(descriptors, ignore_3D=True)

    # Resume logic: count already processed rows
    if os.path.exists(args.output_csv):
        total_lines = sum(1 for _ in open(args.output_csv, 'r'))
        processed = max(0, total_lines - 1)  # exclude header
        first_write = False
        logging.info(f"Resuming: {processed} rows already processed.")
    else:
        processed = 0
        first_write = True

    reader = pd.read_csv(args.input_csv, chunksize=args.chunksize)
    for chunk_idx, chunk in enumerate(reader):
        chunk_start = chunk_idx * args.chunksize
        chunk_end   = chunk_start + len(chunk)

        # Enforce max_rows if debugging
        if args.max_rows is not None:
            if chunk_start >= args.max_rows:
                break
            if chunk_end > args.max_rows:
                chunk = chunk.iloc[: args.max_rows - chunk_start]
                chunk_end = chunk_start + len(chunk)

        # Skip already processed
        if chunk_end <= processed:
            continue
        if chunk_start < processed < chunk_end:
            start = processed - chunk_start
            chunk = chunk.iloc[start:]
            chunk_start += start

        logging.info(f"Processing rows {chunk_start} to {chunk_start + len(chunk) - 1}")

        # Parse SMILES
        smiles_list = chunk["SMILES"].astype(str).tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        valid_indices = [i for i, m in enumerate(mols) if m is not None]
        mols_valid = [m for m in mols if m is not None]
        if not mols_valid:
            logging.warning("No valid molecules in this chunk; skipping")
            continue

        # Compute descriptors
        df_valid = calc.pandas(mols_valid)
        # Coerce any non-numeric to NaN
        df_valid = df_valid.apply(pd.to_numeric, errors='coerce')
        desc_cols = df_valid.columns

        # Build full-chunk DataFrame
        df_full = pd.DataFrame(np.nan, index=range(len(chunk)), columns=desc_cols, dtype=float)
        for out_i, in_i in enumerate(valid_indices):
            df_full.iloc[in_i] = df_valid.iloc[out_i]

        # Record original CSV line index
        df_full.insert(0, "orig_index", np.arange(chunk_start, chunk_end))

        # Write or append
        if first_write:
            df_full.to_csv(args.output_csv, index=False)
            first_write = False
        else:
            df_full.to_csv(args.output_csv, mode="a", header=False, index=False)

    logging.info(f"Finished. Mordred features saved to {args.output_csv}")

if __name__ == "__main__":
    main()
