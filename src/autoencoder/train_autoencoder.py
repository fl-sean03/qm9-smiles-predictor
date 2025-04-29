# src/autoencoder/train_autoencoder.py
#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.autoencoder.model import build_autoencoder

def load_desc(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        return df.values.astype(np.float32)
    return np.load(path)

def main(args):
    # 1) Load raw descriptors
    X = load_desc(args.input_desc)
    N, D = X.shape
    print(f"[INFO] Loaded {{N}}×{{D}} descriptors (raw)")

    # 2) Fit & save scaler
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    os.makedirs(args.output_dir, exist_ok=True)
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Fitted and saved scaler → {scaler_path}")

    # 3) Build AE
    ae, encoder = build_autoencoder(input_dim=D, bottleneck_dim=args.bottleneck_dim)

    # 4) Callbacks
    ae_ckpt = os.path.join(args.output_dir, "autoencoder_best.h5")
    enc_wts = os.path.join(args.output_dir, "encoder_best.weights.h5")
    cbs = [
        EarlyStopping("val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ae_ckpt, monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 5) Train on scaled data
    ae.fit(
        X_scaled, X_scaled,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        callbacks=cbs,
        verbose=2
    )

    # 6) Save encoder weights
    encoder.save_weights(enc_wts)
    print(f"[INFO] Saved autoencoder → {ae_ckpt}")
    print(f"[INFO] Saved encoder weights → {enc_wts}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-desc", required=True,
                   help="CSV or .npy of raw Mordred descriptors")
    p.add_argument("--output-dir", required=True,
                   help="Where to write autoencoder outputs")
    p.add_argument("--bottleneck-dim", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=562)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--validation-split", type=float, default=0.1)
    args = p.parse_args()
    main(args)
