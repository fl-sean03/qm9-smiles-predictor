# src/hybrid/eval_hybrid.py
#!/usr/bin/env python3
"""
Evaluate QM9 FNNs trained on hybrid ECFP4+Mordred features.
Per-property JSON metrics + predictions.csv, plus aggregate all_metrics.csv.
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

# Paper’s SMILES‐based MAEs from Pinheiro et al.
PAPER_MAES = {
    "U0":   0.0573, "U":    0.0582, "H":    0.0575, "G":    0.0562,
    "Cv":   0.1223, "homo": 0.0952, "gap":  0.1369, "mu":   0.5230,
    "alpha":0.3095
}

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hybrid FNN models against paper MAEs")
    p.add_argument("--embeddings",  required=True, help=".npy file of hybrid features")
    p.add_argument("--orig_index",  required=True, help="Text file of orig_index (skip header)")
    p.add_argument("--targets_csv",  required=True, help="QM9 targets CSV with orig_index")
    p.add_argument("--run_dir",      required=True, help="Dir containing model/, scalers/ subfolders from training")
    p.add_argument("--out_dir",      required=True, help="Where to write per-prop outputs and all_metrics.csv")
    return p.parse_args()


def load_embeddings(emb_np, orig_index_file):
    # load features and indices
    X = np.load(emb_np, mmap_mode='r')
    orig_idx = np.loadtxt(orig_index_file, dtype=int, skiprows=1)
    if X.shape[0] != orig_idx.shape[0]:
        raise ValueError(f"Row count mismatch: {X.shape[0]} vs {orig_idx.shape[0]}")
    df_emb = pd.DataFrame(X, index=orig_idx)
    df_emb.index.name = 'orig_index'
    return df_emb


def get_test_indices(n_samples):
    idx = np.arange(n_samples)
    _, idx_test = train_test_split(idx, test_size=0.18, random_state=42)
    return set(idx_test)


def evaluate_one(prop, args, df_emb, targets_df, test_idx_set):
    print(f"\n--- Evaluating {prop} ---")
    # load model & scalers
    model = tf.keras.models.load_model(
        os.path.join(args.run_dir, 'model', f'{prop}_final.h5'), compile=False)
    scaler_X = joblib.load(os.path.join(args.run_dir, 'scalers', f'{prop}_X.pkl'))
    scaler_y = joblib.load(os.path.join(args.run_dir, 'scalers', f'{prop}_y.pkl'))

    # select test rows
    df_test = df_emb.reset_index().loc[df_emb.reset_index()['orig_index'].isin(test_idx_set)]
    idxs = df_test['orig_index'].values
    Xs = scaler_X.transform(df_test.drop('orig_index', axis=1).values.astype(np.float32))
    y_true = targets_df.loc[idxs, prop].values

    # predict
    y_pred_s = model.predict(Xs, verbose=0).ravel().reshape(-1,1)
    y_pred = scaler_y.inverse_transform(y_pred_s).ravel()

    # metrics
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    metrics = {"property": prop,
               "paper_MAE": PAPER_MAES[prop],
               "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    # write outputs
    outdir = os.path.join(args.out_dir, prop)
    os.makedirs(outdir, exist_ok=True)
    # JSON
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    # predictions CSV
    pd.DataFrame({'orig_index': idxs, 'y_true': y_true, 'y_pred': y_pred}) \
      .to_csv(os.path.join(outdir, 'predictions.csv'), index=False)

    print(f"{prop}: MAE={mae:.4f}, paper={PAPER_MAES[prop]:.4f}")
    return metrics


def main():
    args = parse_args()
    df_emb = load_embeddings(args.embeddings, args.orig_index)
    targets_df = pd.read_csv(args.targets_csv).set_index('orig_index')
    n_samples = df_emb.shape[0]
    test_idx = get_test_indices(n_samples)

    all_metrics = []
    for prop in PAPER_MAES:
        m = evaluate_one(prop, args, df_emb, targets_df, test_idx)
        all_metrics.append(m)

    # aggregate
    df_all = pd.DataFrame(all_metrics)
    os.makedirs(args.out_dir, exist_ok=True)
    df_all.to_csv(os.path.join(args.out_dir, 'all_metrics.csv'), index=False)
    print(f"\nWrote aggregate metrics to {os.path.join(args.out_dir, 'all_metrics.csv')}" )

if __name__ == '__main__':
    main()

