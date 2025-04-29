#!/usr/bin/env python3
"""
Chunk-wise clean of large Mordred CSV:
  1) Drop molecules (rows) with no valid descriptors (all-NaN).
  2) Drop descriptor columns with zero variance across valid molecules.
  3) Drop descriptor columns that still have any NaNs among valid molecules.

Outputs the cleaned CSV in chunks.

Usage:
  python clean_mordred.py \
    --input_csv  ../outputs/mordred_csvs/mordred_raw.csv \
    --output_csv ../outputs/mordred_csvs/mordred_cleaned.csv \
    [--chunksize 50000]
"""
import argparse
import pandas as pd
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser(description="Chunk-clean Mordred CSV, ignoring invalid rows")
    p.add_argument("--input_csv",  required=True, help="Raw Mordred CSV with 'orig_index'")
    p.add_argument("--output_csv", required=True, help="Cleaned CSV output")
    p.add_argument("--chunksize",  type=int, default=50000, help="Rows per chunk")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    reader = pd.read_csv(args.input_csv, chunksize=args.chunksize, low_memory=False)
    # Identify columns & first valid-row values
    first = next(reader)
    cols = first.columns.tolist()
    desc_cols = [c for c in cols if c != "orig_index"]
    # Filter valid rows (at least one non-NaN in desc_cols)
    valid_mask = first[desc_cols].notna().any(axis=1)
    first_valid = first.loc[valid_mask].iloc[0]
    first_vals = first_valid[desc_cols].to_dict()
    # Initialize trackers
    is_const = {c: True for c in desc_cols}
    has_na   = {c: False for c in desc_cols}

    # Process first chunk
    valid_chunk = first[valid_mask]
    for c in desc_cols:
        col = valid_chunk[c]
        if col.dropna().ne(first_vals[c]).any():
            is_const[c] = False
        if col.isna().any():
            has_na[c] = True

    # Process remaining chunks
    for chunk in reader:
        mask = chunk[desc_cols].notna().any(axis=1)
        vc = chunk.loc[mask]
        for c in desc_cols:
            col = vc[c]
            if is_const[c] and col.dropna().ne(first_vals[c]).any():
                is_const[c] = False
            if not has_na[c] and col.isna().any():
                has_na[c] = True

    # Columns to drop
    const_cols = [c for c, v in is_const.items() if v]
    na_cols    = [c for c, v in has_na.items()   if v]
    drop_cols  = set(const_cols + na_cols)
    print(f"Dropping {len(const_cols)} constant cols, {len(na_cols)} with NaNs -> total {len(drop_cols)}")

    # Second pass: write cleaned csv
    reader = pd.read_csv(args.input_csv, chunksize=args.chunksize, low_memory=False)
    first_write = True
    for chunk in reader:
        # drop invalid rows
        mask = chunk[desc_cols].notna().any(axis=1)
        df_valid = chunk.loc[mask]
        # drop bad columns
        df_clean = df_valid.drop(columns=list(drop_cols))
        if first_write:
            df_clean.to_csv(args.output_csv, index=False)
            first_write = False
        else:
            df_clean.to_csv(args.output_csv, mode="a", header=False, index=False)

if __name__ == "__main__":
    main()