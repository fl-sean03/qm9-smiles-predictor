#!/usr/bin/env python3
"""
Chunked evaluation of single‐task QM9 models vs. paper MAEs,
with per‐model JSON export and an aggregated all_metrics.csv.

This script ONLY computes metrics (no plots). To generate all visualizations,
run the accompanying viz_qm9.py.
"""
import os
import argparse
import json
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paper’s SMILES‐based MAEs from Pinheiro et al.
PAPER_MAES = {
    "U0":   0.0573, "U":    0.0582, "H":    0.0575, "G":    0.0562,
    "Cv":   0.1223, "homo": 0.0952, "gap":  0.1369, "mu":   0.5230,
    "alpha":0.3095
}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate QM9 models against paper MAEs")
    p.add_argument("--descriptors_csv", required=True,
                   help="Cleaned Mordred descriptors CSV (with orig_index)")
    p.add_argument("--targets_csv",     required=True,
                   help="QM9 targets CSV (with orig_index) ")
    p.add_argument("--run_dir",         required=True,
                   help="Directory containing model/, scalers/ subfolders")
    p.add_argument("--out_dir",         required=True,
                   help="Where to write per‐prop outputs and aggregate CSV")
    p.add_argument("--chunksize", type=int, default=5000,
                   help="Rows per chunk when streaming descriptors")
    return p.parse_args()

def load_targets(targets_csv):
    df = pd.read_csv(targets_csv).set_index("orig_index")
    return df

def get_test_indices(n_samples):
    idx = np.arange(n_samples)
    _, idx_test = train_test_split(idx, test_size=0.18, random_state=42)
    return set(idx_test)

def evaluate_one(prop, args, targets_df, test_idx_set):
    print(f"\n--- Evaluating {prop} ---")
    # load model & scalers
    model = tf.keras.models.load_model(
        os.path.join(args.run_dir, "model", f"{prop}_final.h5"),
        compile=False
    )
    scaler_X = joblib.load(os.path.join(args.run_dir, "scalers", f"{prop}_X.pkl"))
    scaler_y = joblib.load(os.path.join(args.run_dir, "scalers", f"{prop}_y.pkl"))

    # ground truth lookup
    truths = {idx: targets_df.at[idx, prop] for idx in test_idx_set}

    # descriptor columns
    first = pd.read_csv(args.descriptors_csv, nrows=0)
    desc_cols = [c for c in first.columns if c != "orig_index"]

    # streaming predict
    preds = {}
    for chunk in pd.read_csv(args.descriptors_csv,
                             chunksize=args.chunksize,
                             usecols=["orig_index"] + desc_cols):
        mask = chunk["orig_index"].isin(test_idx_set)
        if not mask.any():
            continue
        sub = chunk.loc[mask]
        idxs = sub["orig_index"].values
        Xs = scaler_X.transform(sub[desc_cols].values.astype(np.float32))
        ps = model.predict(Xs, verbose=0).ravel()
        preds.update(dict(zip(idxs, ps)))

    # align and report missing
    valid_idx = sorted(test_idx_set & preds.keys())
    missing = len(test_idx_set) - len(valid_idx)
    if missing:
        print(f"[warning] {missing} test samples missing predictions; skipping")

    y_true = np.array([truths[i] for i in valid_idx])
    y_pred_scaled = np.array([preds[i] for i in valid_idx]).reshape(-1,1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()

    # compute metrics
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    metrics = {
        "property": prop,
        "paper_MAE": PAPER_MAES[prop],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    # write outputs
    outdir = os.path.join(args.out_dir, prop)
    os.makedirs(outdir, exist_ok=True)

    # 1) metrics JSON
    with open(os.path.join(outdir, "metrics.json"), "w") as jf:
        json.dump(metrics, jf, indent=2)

    # 2) raw predictions for plotting
    df_pred = pd.DataFrame({
        "orig_index": valid_idx,
        "y_true":      y_true,
        "y_pred":      y_pred
    })
    df_pred.to_csv(os.path.join(outdir, "predictions.csv"), index=False)

    print(f"{prop}: MAE={mae:.4f}, paper={PAPER_MAES[prop]:.4f}")
    return metrics

def main():
    args = parse_args()
    targets_df = load_targets(args.targets_csv)
    n_samples = sum(1 for _ in open(args.descriptors_csv)) - 1
    test_idx = get_test_indices(n_samples)

    all_metrics = []
    for prop in PAPER_MAES:
        m = evaluate_one(prop, args, targets_df, test_idx)
        all_metrics.append(m)

    # aggregate
    df_all = pd.DataFrame(all_metrics)
    os.makedirs(args.out_dir, exist_ok=True)
    df_all.to_csv(os.path.join(args.out_dir, "all_metrics.csv"), index=False)
    print(f"\nWrote aggregate metrics to {os.path.join(args.out_dir, 'all_metrics.csv')}")


if __name__ == "__main__":
    main()
