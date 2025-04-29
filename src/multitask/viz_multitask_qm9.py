#!/usr/bin/env python3
"""
Enhanced visualization for multi-task QM9 evaluation results.

Generates:

1) Per-property parity & residual plots (saved inside each property’s folder) annotated with MAE, RMSE, R²
2) Aggregated comparison plots:
     • MAE: paper vs ours
     • % improvement scatter
     • R² across properties
     • RMSE across properties
3) A CSV/Markdown summary table including MAE, RMSE, R², Δ(%).

Usage:
  python src/viz_multitask_qm9.py \
    --eval_dir       outputs/eval_multitask \
    --comparison_dir outputs/eval_multitask/comparison
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paper’s SMILES-based MAEs from Pinheiro et al.
PAPER_MAES = {
    "U0":   0.0573, "U":    0.0582, "H":    0.0575, "G":    0.0562,
    "Cv":   0.1223, "homo": 0.0952, "gap":  0.1369, "mu":   0.5230,
    "alpha":0.3095
}
TARGET_COLS = ["U0","U","H","G","Cv","homo","gap","mu","alpha"]


def plot_parity(x, y, metrics, outpath):
    r2 = metrics['R2']
    mae = metrics['MAE']
    rmse = metrics['RMSE']
    mn, mx = np.min([x.min(), y.min()]), np.max([x.max(), y.max()])
    plt.figure(figsize=(4,4))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], '--', color='gray')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'Parity: {metrics["property"]}')
    plt.text(0.05, 0.95,
             f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR²={r2:.3f}",
             transform=plt.gca().transAxes,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_residuals(x, y, metrics, outpath):
    resid = y - x
    rmse = metrics['RMSE']
    plt.figure(figsize=(4,3))
    plt.hist(resid, bins=40, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (pred - true)')
    plt.ylabel('Count')
    plt.title(f'Residuals: {metrics["property"]}')
    plt.text(0.95, 0.95,
             f"RMSE={rmse:.4f}",
             transform=plt.gca().transAxes,
             va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def bar_with_labels(ax, x, heights, fmt="{:.3f}"):
    bars = ax.bar(x, heights, alpha=0.8)
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x()+bar.get_width()/2, h + 0.01*abs(h),
                fmt.format(h), ha='center', va='bottom', fontsize=8)


def main():
    parser = argparse.ArgumentParser(description='Vis for multitask QM9 eval')
    parser.add_argument('--eval_dir',       required=True, help='Multi-task eval directory')
    parser.add_argument('--comparison_dir', required=True, help='Where to save comparison outputs')
    args = parser.parse_args()

    # Load predictions and metrics
    df_pred = pd.read_csv(os.path.join(args.eval_dir, 'predictions.csv'))
    df_metrics = pd.read_csv(os.path.join(args.eval_dir, 'all_metrics.csv'))

    # 1) Per-property plots inside each target folder
    for prop in TARGET_COLS:
        prop_dir = os.path.join(args.eval_dir, prop)
        os.makedirs(prop_dir, exist_ok=True)
        metrics = df_metrics[df_metrics['property']==prop].iloc[0].to_dict()
        metrics['property'] = prop
        x = df_pred[f'{prop}_true'].values
        y = df_pred[f'{prop}_pred'].values
        plot_parity(x, y, metrics, os.path.join(prop_dir, 'parity.png'))
        plot_residuals(x, y, metrics, os.path.join(prop_dir, 'residuals.png'))
        print(f'Plotted for {prop} in {prop_dir}')

    # 2) Aggregated comparison
    os.makedirs(args.comparison_dir, exist_ok=True)
    df = df_metrics.copy()
    df['paper_MAE'] = df['property'].map(PAPER_MAES)
    df['delta_pct'] = (df['paper_MAE'] - df['MAE']) / df['paper_MAE'] * 100
    df['prop'] = pd.Categorical(df['property'], categories=TARGET_COLS, ordered=True)
    df = df.sort_values('prop')

    # 2a) MAE compare
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(ax, df['prop'], df['paper_MAE'], fmt="{:.4f}")
    bar_with_labels(ax, df['prop'], df['MAE'],       fmt="{:.4f}")
    ax.set_ylabel('MAE')
    ax.set_title('MAE: paper vs ours')
    ax.legend(['paper','ours'])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, 'MAE_compare.png'), dpi=300)
    plt.close(fig)

    # 2b) % improvement
    fig, ax = plt.subplots(figsize=(6,3))
    ax.scatter(df['prop'], df['delta_pct'], s=50)
    for x_lbl, y_val in zip(df['prop'], df['delta_pct']):
        ax.text(x_lbl, y_val + 0.5, f"{y_val:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_ylabel('% Δ vs paper')
    ax.set_title('Relative improvement')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, 'delta_percent.png'), dpi=300)
    plt.close(fig)

    # 2c) R² across props
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(ax, df['prop'], df['R2'], fmt="{:.3f}")
    ax.set_ylabel('R²')
    ax.set_title('R² across properties')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, 'R2_across_props.png'), dpi=300)
    plt.close(fig)

    # 2d) RMSE across props
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(ax, df['prop'], df['RMSE'], fmt="{:.4f}")
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE across properties')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, 'RMSE_across_props.png'), dpi=300)
    plt.close(fig)

    # 3) Summary tables
    df.to_csv(os.path.join(args.comparison_dir, 'summary_table.csv'), index=False)
    md = ['| Property | MAE (paper) | MAE (ours) | RMSE | R² | Δ (%) |',
          '|---|---|---|---|---|---|']
    for _, row in df.iterrows():
        md.append(f"| {row['prop']} | {row['paper_MAE']:.4f} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.3f} | {row['delta_pct']:.1f}% |")
    with open(os.path.join(args.comparison_dir, 'summary_table.md'),'w') as f:
        f.write('\n'.join(md))

    print(f'Comparison plots & tables written to {args.comparison_dir}')

if __name__ == '__main__':
    main()
