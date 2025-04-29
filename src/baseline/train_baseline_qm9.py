#!/usr/bin/env python3
"""
Train single-task baseline networks on QM9 Mordred descriptors.

For each QM9 target property, trains a separate FNN:
  - 2 hidden layers (256,256), ReLU
  - Batch size = 562
  - Adam(lr=0.001, β1=0.9, β2=0.999)
  - EarlyStopping(patience=10)
  - Max epochs = 200
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
    p = argparse.ArgumentParser(description="Train single-task QM9 FNN baselines")
    p.add_argument("--descriptors_csv", required=True,
                   help="Cleaned Mordred descriptors CSV (with orig_index)")
    p.add_argument("--targets_csv", required=True,
                   help="QM9 targets CSV (with orig_index)")
    p.add_argument("--output_dir", required=True,
                   help="Dir to save per-target checkpoints, models, scalers, history")
    p.add_argument("--batch_size",   type=int,   default=562,
                   help="Batch size (default: 562)")
    p.add_argument("--epochs",       type=int,   default=200,
                   help="Max epochs (default: 200)")
    p.add_argument("--learning_rate",type=float, default=0.001,
                   help="Adam learning rate (default: 0.001)")
    p.add_argument("--hidden_layers",type=str,   default="256,256",
                   help="Comma-separated hidden layer sizes (default: 256,256)")
    p.add_argument("--patience",     type=int,   default=10,
                   help="EarlyStopping patience (default: 10)")
    return p.parse_args()

def build_model(input_dim, hidden_layers, lr):
    model = Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
        else:
            model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def main():
    args = parse_args()
    # load data
    df_desc = pd.read_csv(args.descriptors_csv)
    df_targ = pd.read_csv(args.targets_csv)
    df = df_desc.merge(df_targ, on="orig_index", how="inner")

    # feature columns
    target_cols  = [c for c in df_targ.columns if c != "orig_index"]
    feature_cols = [c for c in df.columns 
                    if c not in (["orig_index"] + target_cols)]
    X_all = df[feature_cols].values.astype(np.float32)

    # parse hidden sizes
    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x]

    # create dirs
    for sub in ("checkpoints", "model", "scalers", "history"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # loop targets
    for target in target_cols:
        print(f"\n=== Training for target: {target} ===")
        y_all = df[[target]].values.astype(np.float32)

        # split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.18, random_state=42)
        val_frac = 0.04 / 0.82
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_frac, random_state=42)

        # scale
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

        # build & train
        model = build_model(input_dim=X_train_s.shape[1],
                            hidden_layers=hidden_layers,
                            lr=args.learning_rate)

        ckpt_path = os.path.join(args.output_dir, "checkpoints", f"{target}_best.h5")
        cb_ckpt = ModelCheckpoint(ckpt_path, monitor="val_mae",
                                  mode="min", save_best_only=True, verbose=1)
        cb_es = EarlyStopping(monitor="val_mae", mode="min",
                              patience=args.patience,
                              restore_best_weights=True, verbose=1)

        history = model.fit(
            X_train_s, y_train_s,
            validation_data=(X_val_s, y_val_s),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[cb_ckpt, cb_es],
            verbose=2
        )

        # evaluate
        loss, mae_s = model.evaluate(X_test_s, scaler_y.transform(y_test), verbose=0)
        mae_orig = mae_s * scaler_y.scale_[0]
        print(f"[{target}] Test MAE (orig units): {mae_orig:.4f}")

        # save model and history
        model.save(os.path.join(args.output_dir, "model", f"{target}_final.h5"))
        pd.DataFrame(history.history).to_csv(
            os.path.join(args.output_dir, "history", f"{target}_history.csv"), index=False
        )

if __name__ == "__main__":
    main()

