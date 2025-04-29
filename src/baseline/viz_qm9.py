#!/usr/bin/env python3
"""
Enhanced visualization for QM9 evaluation results.

Generates:

1) Per-property parity & residual plots annotated with MAE, RMSE, R²
2) Aggregated comparison plots:
     • MAE paper vs ours (with bar labels)
     • % improvement scatter (with labels)
     • R² across properties
     • RMSE across properties
3) A CSV/Markdown summary table including MAE, RMSE, R², Δ(%)

Usage:
  python viz_qm9.py \
    --eval_dir        path/to/outputs/evaluation \
    --comparison_dir  path/to/outputs/evaluation/comparison
"""
import os
import glob
import json
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Publication-ready QM9 visualizations")
    p.add_argument("--eval_dir",       required=True,
                   help="Root dir containing per-prop eval outputs")
    p.add_argument("--comparison_dir", required=True,
                   help="Dir to write aggregated comparison plots & tables")
    return p.parse_args()

def plot_parity(df_pred, metrics, outpath):
    y_true = df_pred["y_true"].values
    y_pred = df_pred["y_pred"].values
    r2 = metrics["R2"]
    mae = metrics["MAE"]
    rmse = metrics["RMSE"]

    plt.figure(figsize=(4,4))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], "--", color="gray")
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title(f"Parity for {metrics['property']}")
    plt.text(0.05, 0.95,
             f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.3f}",
             transform=plt.gca().transAxes,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_residuals(df_pred, metrics, outpath):
    res = df_pred["y_pred"] - df_pred["y_true"]
    rmse = metrics["RMSE"]
    plt.figure(figsize=(4,3))
    plt.hist(res, bins=40, edgecolor="black", alpha=0.7)
    plt.xlabel("Residual (predicted − true)")
    plt.ylabel("Count")
    plt.title(f"Residuals for {metrics['property']}")
    plt.text(0.95, 0.95,
             f"RMSE = {rmse:.4f}",
             transform=plt.gca().transAxes,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def bar_with_labels(x, heights, ax, fmt="{:.3f}"):
    bars = ax.bar(x, heights, alpha=0.8)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01*h,
                fmt.format(h), ha="center", va="bottom", fontsize=8)

def main():
    args = parse_args()

    # --- 1) Per-property visuals ---
    for metaj in glob.glob(os.path.join(args.eval_dir, "*", "metrics.json")):
        prop_dir = os.path.dirname(metaj)
        metrics = json.load(open(metaj))
        preds_csv = os.path.join(prop_dir, "predictions.csv")
        if not os.path.exists(preds_csv):
            continue
        df_pred = pd.read_csv(preds_csv)

        # parity + residuals
        plot_parity(df_pred, metrics, os.path.join(prop_dir, "parity.png"))
        plot_residuals(df_pred, metrics, os.path.join(prop_dir, "residuals.png"))
        print(f" → Plots for {metrics['property']}")

    # --- 2) Aggregated comparison ---
    os.makedirs(args.comparison_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.eval_dir, "all_metrics.csv"))

    # percent improvement
    df["delta_pct"] = (df["paper_MAE"] - df["MAE"]) / df["paper_MAE"] * 100

    # ordering
    order = ["U0","U","H","G","Cv","homo","gap","mu","alpha"]
    df["prop"] = pd.Categorical(df["property"], categories=order, ordered=True)
    df = df.sort_values("prop")

    # 2a) MAE compare
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(df["prop"], df["paper_MAE"], ax, fmt="{:.4f}")
    bar_with_labels(df["prop"], df["MAE"], ax, fmt="{:.4f}")
    ax.set_ylabel("MAE")
    ax.set_title("MAE: paper vs ours")
    ax.legend(["paper","ours"])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, "MAE_compare.png"), dpi=300)
    plt.close(fig)

    # 2b) % improvement scatter
    fig, ax = plt.subplots(figsize=(6,3))
    ax.scatter(df["prop"], df["delta_pct"], s=50)
    for x,y in zip(df["prop"], df["delta_pct"]):
        ax.text(x, y + 0.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_ylabel("% Δ vs paper")
    ax.set_title("Relative improvement over paper")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, "delta_percent.png"), dpi=300)
    plt.close(fig)

    # 2c) R² bar chart
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(df["prop"], df["R2"], ax, fmt="{:.3f}")
    ax.set_ylabel("R²")
    ax.set_title("R² across properties")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, "R2_across_props.png"), dpi=300)
    plt.close(fig)

    # 2d) RMSE bar chart
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(df["prop"], df["RMSE"], ax, fmt="{:.4f}")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE across properties")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, "RMSE_across_props.png"), dpi=300)
    plt.close(fig)

    # --- 3) Expanded summary tables ---
    out_csv = os.path.join(args.comparison_dir, "summary_table.csv")
    df.to_csv(out_csv, index=False)

    md_lines = ["| Property | MAE (paper) | MAE (ours) | RMSE | R² | Δ (%) |",
                "|---|---|---|---|---|---|"]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['prop']} "
            f"| {row['paper_MAE']:.4f} "
            f"| {row['MAE']:.4f} "
            f"| {row['RMSE']:.4f} "
            f"| {row['R2']:.3f} "
            f"| {row['delta_pct']:.1f}% |"
        )
    with open(os.path.join(args.comparison_dir, "summary_table.md"), "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nWritten comparison figures and tables to {args.comparison_dir}")

if __name__ == "__main__":
    main()
