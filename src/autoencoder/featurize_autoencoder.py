# src/autoencoder/featurize_autoencoder.py
#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.autoencoder.model import build_autoencoder

def load_desc(path):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        return df.values.astype(np.float32)
    return np.load(path)

def main(args):
    # 1) Load raw descriptors and scaler
    X = load_desc(args.input_desc)
    scaler: StandardScaler = joblib.load(args.scaler)
    X_scaled = scaler.transform(X)
    N, D = X_scaled.shape
    print(f"[INFO] Loaded and scaled {{N}}×{{D}} descriptors")

    # 2) Rebuild encoder and load weights
    _, encoder = build_autoencoder(input_dim=D, bottleneck_dim=args.bottleneck_dim)
    encoder.load_weights(args.encoder_weights)
    print(f"[INFO] Loaded encoder weights from {args.encoder_weights}")

    # 3) Generate embeddings
    H = encoder.predict(X_scaled, batch_size=args.batch_size, verbose=1)
    os.makedirs(os.path.dirname(args.output_emb), exist_ok=True)
    np.save(args.output_emb, H)
    print(f"[INFO] Saved embeddings {{H.shape}} → {args.output_emb}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-desc", required=True,
                   help="CSV or .npy of raw Mordred descriptors")
    p.add_argument("--encoder-weights", required=True,
                   help=".weights.h5 file from training step")
    p.add_argument("--scaler", required=True,
                   help=".pkl file from train_autoencoder fit")
    p.add_argument("--output-emb", required=True,
                   help="Where to write .npy embeddings")
    p.add_argument("--bottleneck-dim", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()
    main(args)

