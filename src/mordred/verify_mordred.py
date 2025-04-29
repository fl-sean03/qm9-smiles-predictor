#!/usr/bin/env python3
"""
Chunk‐wise verify of a large Mordred descriptor CSV without loading everything into memory.

Usage:
    python verify_mordred.py \
        --input_csv path/to/mordred_raw.csv \
        [--chunksize 50000]

Outputs:
  - Total rows × columns
  - First 10 column names
  - orig_index coverage (min, max, unique)
  - List of any non‐numeric columns
  - Top 10 columns by NaN count
  - Approximate memory usage (MB)
"""
import argparse
import pandas as pd
import numpy as np
import os

def parse_args():
    p = argparse.ArgumentParser(description="Chunk‐wise verify Mordred CSV")
    p.add_argument("--input_csv",  required=True, help="Path to the Mordred CSV")
    p.add_argument("--chunksize",  type=int, default=50000, help="Rows per chunk")
    return p.parse_args()

def main():
    args = parse_args()
    # Read header only
    header = pd.read_csv(args.input_csv, nrows=0)
    cols = header.columns.tolist()
    ncols = len(cols)
    print("Columns:", ncols)
    print("First 10 columns:", cols[:10])

    total_rows = 0
    orig_min = None
    orig_max = None
    nan_accum = pd.Series(0, index=cols, dtype=np.int64)
    nonnum = set()
    mem_bytes = 0

    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize, low_memory=False):
        n = len(chunk)
        total_rows += n
        mem_bytes += chunk.memory_usage(deep=True).sum()

        # orig_index tracking
        if "orig_index" in chunk:
            mn, mx = chunk["orig_index"].min(), chunk["orig_index"].max()
            orig_min = mn if orig_min is None else min(orig_min, mn)
            orig_max = mx if orig_max is None else max(orig_max, mx)

        # NaN counts
        nan_accum += chunk.isna().sum()

        # non-numeric detection
        for c in cols:
            if chunk[c].dtype == object:
                nonnum.add(c)

    print("Shape:", (total_rows, ncols))
    if orig_min is not None:
        print("orig_index range:", orig_min, "to", orig_max,
              "(unique rows covered)", total_rows == (orig_max - orig_min + 1))
    else:
        print("Warning: no 'orig_index' column found")

    print("Non‐numeric columns:", sorted(nonnum))
    print("Top 10 columns by NaN count:")
    print(nan_accum.sort_values(ascending=False).head(10))

    print(f"Approximate memory usage: {mem_bytes / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
