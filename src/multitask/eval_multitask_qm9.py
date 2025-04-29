#!/usr/bin/env python3
"""
Evaluate a single multitask QM9 model on the held-out test set.
Computes per-property MAE, MSE, RMSE, R², writes metrics.json, all_metrics.csv,
and a predictions.csv with true vs predicted values for each property.

Usage:
  python eval_multitask_qm9.py \
    --descriptors_csv path/to/mordred_cleaned.csv \
    --targets_csv     path/to/qm9_targets.csv \
    --model_path      path/to/model/multitask_best.h5 \
    --scaler_X        path/to/scalers/scaler_X.pkl \
    --scaler_y        path/to/scalers/scaler_y.pkl \
    --out_dir         path/to/outputs/eval_multitask \
    [--chunksize      5000]
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COLS = ["U0","U","H","G","Cv","homo","gap","mu","alpha"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multitask QM9 model")
    p.add_argument("--descriptors_csv", required=True, help="Cleaned Mordred descriptors CSV with orig_index")
    p.add_argument("--targets_csv",     required=True, help="QM9 targets CSV with orig_index and properties")
    p.add_argument("--model_path",      required=True, help="Path to multitask_best.h5")
    p.add_argument("--scaler_X",        required=True, help="Path to scaler_X.pkl")
    p.add_argument("--scaler_y",        required=True, help="Path to scaler_y.pkl")
    p.add_argument("--out_dir",         required=True, help="Where to write metrics and predictions")
    p.add_argument("--chunksize",       type=int, default=5000, help="Rows per chunk for streaming descriptors")
    p.add_argument("--test_frac",       type=float, default=0.18, help="Test fraction (must match training) ")
    p.add_argument("--seed",            type=int,   default=42,    help="Random seed to reproduce splits")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load scalers and model
    scaler_X = joblib.load(args.scaler_X)
    scaler_y = joblib.load(args.scaler_y)
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Determine test indices from same split used in training
    # Count total samples in descriptors
    n_samples = sum(1 for _ in open(args.descriptors_csv)) - 1
    idx = np.arange(n_samples)
    _, idx_test = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    test_set = set(idx_test)

    # Load ground truths
    df_targets = pd.read_csv(args.targets_csv).set_index('orig_index')

    # Prepare for streaming
    preds = {}
    # Identify descriptor columns
    first = pd.read_csv(args.descriptors_csv, nrows=0)
    desc_cols = [c for c in first.columns if c != 'orig_index']

    for chunk in pd.read_csv(
        args.descriptors_csv,
        usecols=['orig_index'] + desc_cols,
        chunksize=args.chunksize
    ):
        mask = chunk['orig_index'].isin(test_set)
        if not mask.any():
            continue
        sub = chunk.loc[mask]
        idxs = sub['orig_index'].values
        Xs = scaler_X.transform(sub[desc_cols].values.astype(np.float32))
        y_pred_s = model.predict(Xs, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_s)
        for i, idx_val in enumerate(idxs):
            preds[idx_val] = y_pred[i]

    # Align predictions and compute metrics
    valid_idxs = sorted(test_set & preds.keys())
    missing = len(test_set) - len(valid_idxs)
    if missing:
        print(f"[warning] {missing} test samples missing predictions")

    y_true_mat = np.array([df_targets.loc[i, TARGET_COLS].values for i in valid_idxs])
    y_pred_mat = np.vstack([preds[i] for i in valid_idxs])

    # Save per-sample predictions
    df_out = pd.DataFrame({'orig_index': valid_idxs})
    for i, prop in enumerate(TARGET_COLS):
        df_out[f'{prop}_true'] = y_true_mat[:, i]
        df_out[f'{prop}_pred'] = y_pred_mat[:, i]
    df_out.to_csv(os.path.join(args.out_dir, 'predictions.csv'), index=False)

    # Compute per-property metrics
    metrics_list = []
    for i, prop in enumerate(TARGET_COLS):
        y_t = y_true_mat[:, i]
        y_p = y_pred_mat[:, i]
        mae  = mean_absolute_error(y_t, y_p)
        mse  = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_t, y_p)
        metrics_list.append({
            'property': prop,
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2
        })
        print(f"{prop}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.3f}")

    # Write aggregate metrics CSV and JSON
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(args.out_dir, 'all_metrics.csv'), index=False)
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as jf:
        json.dump(metrics_list, jf, indent=2)

    print(f"Saved metrics and predictions to {args.out_dir}")

if __name__ == '__main__':
    main()
