#!/usr/bin/env python3
"""
Train a multitask FNN on QM9 Mordred descriptors for all 9 target properties,
using a shared 2×256 ReLU architecture with plain MSE on standardized targets,
so each task is equally weighted.

Architecture & hyperparameters:
  - 2 hidden layers of 256 units each, ReLU activations
  - Batch size = 562
  - Adam optimizer (lr=0.001, β1=0.9, β2=0.999)
  - Loss: MSE on scaled targets (each zero mean/unit variance)
  - Metrics: MAE on scaled targets
  - EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True)
  - Max epochs = 200
  - Data split: 78% train, 4% val, 18% test
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Nine QM9 target properties
TARGET_COLS = ["U0","U","H","G","Cv","homo","gap","mu","alpha"]

def parse_args():
    p = argparse.ArgumentParser(description="Train fixed multitask QM9 FNN")
    p.add_argument("--descriptors_csv", required=True,
                   help="Cleaned Mordred descriptors CSV (with orig_index)")
    p.add_argument("--targets_csv", required=True,
                   help="QM9 targets CSV (with orig_index)")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save models, scalers, history, metrics, predictions")
    p.add_argument("--batch_size", type=int, default=562,
                   help="Batch size (default: 562)")
    p.add_argument("--epochs",     type=int, default=200,
                   help="Max epochs (default: 200)")
    p.add_argument("--learning_rate", type=float, default=0.001,
                   help="Adam learning rate (default: 0.001)")
    p.add_argument("--hidden_layers", type=str, default="256,256",
                   help="Comma-separated hidden layer sizes (default: 256,256)")
    p.add_argument("--test_frac", type=float, default=0.18,
                   help="Fraction for test split (default: 0.18)")
    p.add_argument("--val_frac",  type=float, default=0.04,
                   help="Fraction for val split (default: 0.04)")
    p.add_argument("--patience",  type=int, default=10,
                   help="EarlyStopping patience (default: 10)")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


def build_model(input_dim, hidden_layers, lr):
    model = Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(Dense(units, activation='relu', input_shape=(input_dim,)))
        else:
            model.add(Dense(units, activation='relu'))
    model.add(Dense(len(TARGET_COLS), activation='linear'))  # multitask outputs
    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 1) Load data
    df_desc = pd.read_csv(args.descriptors_csv)
    df_targ = pd.read_csv(args.targets_csv)
    df = df_desc.merge(df_targ, on='orig_index', how='inner')

    # 2) Feature/Target split
    feat_cols = [c for c in df.columns if c not in ['orig_index'] + TARGET_COLS]
    targ_cols = TARGET_COLS

    # 3) Train/Val/Test split (78/4/18)
    df_train_val, df_test = train_test_split(
        df, test_size=args.test_frac, random_state=args.seed
    )
    val_rel = args.val_frac / (1 - args.test_frac)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_rel, random_state=args.seed
    )

    # 4) Convert to arrays
    X_train = df_train[feat_cols].values.astype(np.float32)
    X_val   = df_val[feat_cols].values.astype(np.float32)
    X_test  = df_test[feat_cols].values.astype(np.float32)
    y_train = df_train[targ_cols].values.astype(np.float32)
    y_val   = df_val[targ_cols].values.astype(np.float32)
    y_test  = df_test[targ_cols].values.astype(np.float32)

    # 5) Standardize
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train_s = scaler_X.transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train)
    y_val_s   = scaler_y.transform(y_val)

    # 6) Create output directories
    for sub in ['model','scalers','history','predictions']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # 7) Save scalers
    joblib.dump(scaler_X, os.path.join(args.output_dir,'scalers','scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(args.output_dir,'scalers','scaler_y.pkl'))

    # 8) Build model
    hidden = [int(x) for x in args.hidden_layers.split(',') if x]
    model = build_model(input_dim=X_train_s.shape[1], hidden_layers=hidden, lr=args.learning_rate)

    # 9) Callbacks monitoring val_mae
    best_path = os.path.join(args.output_dir,'model','multitask_best.h5')
    cb_ckpt = ModelCheckpoint(
        best_path, monitor='val_mae', mode='min', save_best_only=True, verbose=1
    )
    cb_es = EarlyStopping(
        monitor='val_mae', mode='min', patience=args.patience,
        restore_best_weights=True, verbose=1
    )

    # 10) Train
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[cb_ckpt, cb_es],
        verbose=2
    )

    # 11) Save final model & history
    model.save(os.path.join(args.output_dir,'model','multitask_final.h5'))
    pd.DataFrame(history.history).to_csv(
        os.path.join(args.output_dir,'history','history.csv'), index=False
    )

    # 12) Evaluate on test set
    model.load_weights(best_path)
    y_pred_s = model.predict(X_test_s)
    y_pred = scaler_y.inverse_transform(y_pred_s)

    # 13) Compute & save metrics
    metrics = {}
    for i, prop in enumerate(targ_cols):
        y_t, y_p = y_test[:,i], y_pred[:,i]
        mae = mean_absolute_error(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t, y_p)
        metrics[prop] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        print(f"{prop}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.3f}")
    with open(os.path.join(args.output_dir,'metrics.json'),'w') as jf:
        json.dump(metrics, jf, indent=2)

    # 14) Save per-sample predictions
    df_out = df_test[['orig_index']].copy()
    for i, prop in enumerate(targ_cols):
        df_out[f"{prop}_true"] = y_test[:,i]
        df_out[f"{prop}_pred"] = y_pred[:,i]
    df_out.to_csv(
        os.path.join(args.output_dir,'predictions','predictions.csv'), index=False
    )

    print(f"Training complete. Outputs in {args.output_dir}")

if __name__ == '__main__':
    main()
