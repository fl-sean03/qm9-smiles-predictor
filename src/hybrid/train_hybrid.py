#!/usr/bin/env python3
"""
src/hybrid/train_hybrid.py

Train single-task FNNs on hybrid ECFP4+Mordred features, memory-mapped from .npy.
Each QM9 target is trained separately.
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train single-task FNNs on hybrid ECFP4+Mordred features"
    )
    parser.add_argument(
        "--embeddings", required=True,
        help="Path to hybrid features .npy (memmapable)"
    )
    parser.add_argument(
        "--orig_index", required=True,
        help="Text file listing orig_index (one per line) used in featurization"
    )
    parser.add_argument(
        "--targets_csv", required=True,
        help="QM9 targets CSV with 'orig_index' column"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save checkpoints, models, scalers, history"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Max training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--hidden_layers", type=str, default="256,256",
        help="Comma-separated hidden layer sizes"
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="EarlyStopping patience"
    )
    return parser.parse_args()

def build_model(input_dim, hidden_layers, lr):
    model = Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
        else:
            model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def main():
    args = parse_args()

    # Load hybrid features via memmap
    X = np.load(args.embeddings, mmap_mode='r')  # shape (N_samples, D_feats)

    # Load orig_index list (skip header if present)
    orig_idx = np.loadtxt(args.orig_index, dtype=int, skiprows=1)
    if X.shape[0] != orig_idx.shape[0]:
        raise ValueError(
            f"Row count mismatch: features {X.shape[0]} rows vs orig_index {orig_idx.shape[0]} entries"
        )

    # Load targets
    df_targ = pd.read_csv(args.targets_csv).set_index('orig_index')
    target_cols = df_targ.columns.tolist()

    # Loop over each QM9 target
    for prop in target_cols:
        print(f"\n=== Training target: {prop} ===")
        y_all = df_targ.loc[orig_idx, prop].values.astype(np.float32).reshape(-1,1)
        idx = np.arange(X.shape[0])

        # Split: 78% train+val, 18% test
        X_temp_idx, X_test_idx, y_temp, y_test = train_test_split(
            idx, y_all, test_size=0.18, random_state=42
        )
        val_frac = 0.04 / (1 - 0.18)
        X_train_idx, X_val_idx, y_train, y_val = train_test_split(
            X_temp_idx, y_temp, test_size=val_frac, random_state=42
        )

        X_train = X[X_train_idx]
        X_val   = X[X_val_idx]
        X_test  = X[X_test_idx]

        # Scale inputs & targets
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        X_train_s = scaler_X.transform(X_train)
        X_val_s   = scaler_X.transform(X_val)
        X_test_s  = scaler_X.transform(X_test)
        y_train_s = scaler_y.transform(y_train)
        y_val_s   = scaler_y.transform(y_val)

        # Save scalers
        os.makedirs(os.path.join(args.output_dir, 'scalers'), exist_ok=True)
        joblib.dump(scaler_X, os.path.join(args.output_dir, 'scalers', f'{prop}_X.pkl'))
        joblib.dump(scaler_y, os.path.join(args.output_dir, 'scalers', f'{prop}_y.pkl'))

        # Build model
        hidden = [int(x) for x in args.hidden_layers.split(',') if x]
        model = build_model(input_dim=X_train_s.shape[1], hidden_layers=hidden,
                            lr=args.learning_rate)

        # Callbacks
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, 'checkpoints', f'{prop}_best.h5')
        cb_ckpt = ModelCheckpoint(ckpt_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1)
        cb_es   = EarlyStopping(monitor='val_mae', mode='min', patience=args.patience,
                                 restore_best_weights=True, verbose=1)

        # Train
        history = model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[cb_ckpt, cb_es],
            verbose=2
        )

        # Evaluate
        loss, mae_s = model.evaluate(X_test_s, scaler_y.transform(y_test), verbose=0)
        mae_orig = mae_s * scaler_y.scale_[0]
        print(f"[{prop}] Test MAE (orig units): {mae_orig:.4f}")

        # Save final model & history
        os.makedirs(os.path.join(args.output_dir, 'model'), exist_ok=True)
        model.save(os.path.join(args.output_dir, 'model', f'{prop}_final.h5'))
        pd.DataFrame(history.history).to_csv(
            os.path.join(args.output_dir, 'history', f'{prop}_history.csv'), index=False
        )

if __name__ == '__main__':
    main()

