# src/autoencoder/train_qm9_with_ae.py
#!/usr/bin/env python3
"""
Train QM9 FNNs using autoencoder-derived embeddings.
Each QM9 target is learned separately, mirroring the original baseline.
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
    p = argparse.ArgumentParser(
        description="Train QM9 FNNs on AE embeddings"
    )
    p.add_argument("--embeddings", required=True,
                   help=".npy file with shape (N, D) of AE embeddings")
    p.add_argument("--descriptors_csv", required=True,
                   help="CSV of Mordred descriptors with orig_index to align embeddings")
    p.add_argument("--targets_csv", required=True,
                   help="CSV of QM9 targets with orig_index column")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save checkpoints, models, scalers, history")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs",     type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--hidden_layers", type=str, default="256,256",
                   help="Comma-separated hidden layer sizes")
    p.add_argument("--patience",    type=int, default=10)
    return p.parse_args()

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

    # 1) Load embeddings and alignment
    H = np.load(args.embeddings)
    df_desc = pd.read_csv(args.descriptors_csv)
    if len(df_desc) != H.shape[0]:
        raise ValueError(
            f"Mismatch: {len(df_desc)} descriptor rows vs {H.shape[0]} embeddings"
        )
    # Build DataFrame with orig_index and each AE dimension
    emb_cols = [f"AE_{i}" for i in range(H.shape[1])]
    df_emb = pd.DataFrame(H, index=df_desc["orig_index"], columns=emb_cols)
    df_emb = df_emb.reset_index()  # brings orig_index back as a column

    # 2) Merge with QM9 targets
    df_targ = pd.read_csv(args.targets_csv)
    df = df_emb.merge(df_targ, on="orig_index", how="inner")

    # Identify features & targets
    target_cols  = [c for c in df_targ.columns if c != "orig_index"]
    feature_cols = emb_cols

    X_all = df[feature_cols].values.astype(np.float32)

    # parse hidden layer sizes
    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x]

    # create output subdirs
    for sub in ("checkpoints","model","scalers","history"):  
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Loop over each QM9 target
    for target in target_cols:
        print(f"\n=== Training for target: {target} ===")
        y_all = df[[target]].values.astype(np.float32)

        # train/val/test split (78/4/18%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.18, random_state=42
        )
        val_frac = 0.04 / 0.82
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, random_state=42
        )

        # standardize inputs & outputs
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        X_train_s = scaler_X.transform(X_train)
        X_val_s   = scaler_X.transform(X_val)
        X_test_s  = scaler_X.transform(X_test)
        y_train_s = scaler_y.transform(y_train)
        y_val_s   = scaler_y.transform(y_val)

        # save scalers
        joblib.dump(scaler_X, os.path.join(args.output_dir, "scalers", f"{target}_X.pkl"))
        joblib.dump(scaler_y, os.path.join(args.output_dir, "scalers", f"{target}_y.pkl"))

        # build & compile model
        model = build_model(
            input_dim=X_train_s.shape[1],
            hidden_layers=hidden_layers,
            lr=args.learning_rate
        )

        # prepare callbacks
        ckpt_path = os.path.join(args.output_dir, "checkpoints", f"{target}_best.h5")
        cb_ckpt = ModelCheckpoint(
            ckpt_path, monitor="val_mae", mode="min",
            save_best_only=True, verbose=1
        )
        cb_es = EarlyStopping(
            monitor="val_mae", mode="min",
            patience=args.patience, restore_best_weights=True, verbose=1
        )

        # train
        history = model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[cb_ckpt, cb_es],
            verbose=2
        )

        # evaluate on test set
        loss, mae_s = model.evaluate(
            X_test_s, scaler_y.transform(y_test), verbose=0
        )
        mae_orig = mae_s * scaler_y.scale_[0]
        print(f"[{target}] Test MAE (orig units): {mae_orig:.4f}")

        # save final model & history
        model.save(os.path.join(args.output_dir, "model", f"{target}_final.h5"))
        pd.DataFrame(history.history).to_csv(
            os.path.join(args.output_dir, "history", f"{target}_history.csv"),
            index=False
        )

if __name__ == "__main__":
    main()

