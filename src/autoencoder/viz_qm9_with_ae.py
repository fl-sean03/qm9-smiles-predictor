# src/autoencoder/viz_qm9_with_ae.py
#!/usr/bin/env python3
"""
Visualization for QM9 AE-run evaluation results.
Generates per-property parity/residuals and aggregated comparison.
"""
import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="AE-run QM9 visualizations")
    p.add_argument("--eval_dir",       required=True,
                   help="Root dir containing per-prop AE-run eval outputs")
    p.add_argument("--comparison_dir", required=True,
                   help="Dir to write aggregated comparison plots & tables")
    return p.parse_args()

def plot_parity(df_pred, metrics, outpath):
    y_true, y_pred = df_pred['y_true'], df_pred['y_pred']
    r2, mae, rmse = metrics['R2'], metrics['MAE'], metrics['RMSE']
    plt.figure(figsize=(4,4))
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.plot([mn,mx],[mn,mx],'--', color='gray')
    plt.xlabel('True value'); plt.ylabel('Predicted value')
    plt.title(f"Parity for {metrics['property']}")
    plt.text(0.05,0.95,f"MAE={mae:.4f}\nRMSE={rmse:.4f}\nR²={r2:.3f}",
             transform=plt.gca().transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def plot_residuals(df_pred, metrics, outpath):
    res = df_pred['y_pred'] - df_pred['y_true']
    rmse = metrics['RMSE']
    plt.figure(figsize=(4,3))
    plt.hist(res, bins=40, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (pred−true)'); plt.ylabel('Count')
    plt.title(f"Residuals for {metrics['property']}")
    plt.text(0.95,0.95,f"RMSE={rmse:.4f}", transform=plt.gca().transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def bar_with_labels(x, heights, ax, fmt="{:.3f}"):
    bars = ax.bar(x, heights, alpha=0.8)
    for bar,h in zip(bars,heights):
        ax.text(bar.get_x()+bar.get_width()/2, h*1.01, fmt.format(h),
                ha='center', va='bottom', fontsize=8)

def main():
    args = parse_args()
    # per-prop visuals
    for metaj in glob.glob(os.path.join(args.eval_dir, '*', 'metrics.json')):
        prop_dir = os.path.dirname(metaj)
        metrics = json.load(open(metaj))
        preds_csv = os.path.join(prop_dir, 'predictions.csv')
        if not os.path.exists(preds_csv):
            continue
        df_pred = pd.read_csv(preds_csv)
        plot_parity(df_pred, metrics, os.path.join(prop_dir, 'parity.png'))
        plot_residuals(df_pred, metrics, os.path.join(prop_dir, 'residuals.png'))
        print(f" → Plots for {metrics['property']}")

    # aggregated comparison
    os.makedirs(args.comparison_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.eval_dir, 'all_metrics.csv'))
    df['delta_pct'] = (df['paper_MAE'] - df['MAE'])/df['paper_MAE']*100
    order = ['U0','U','H','G','Cv','homo','gap','mu','alpha']
    df['prop'] = pd.Categorical(df['property'], categories=order, ordered=True)
    df = df.sort_values('prop')

    # MAE compare
    fig, ax = plt.subplots(figsize=(8,4))
    bar_with_labels(df['prop'], df['paper_MAE'], ax, fmt="{:.4f}")
    bar_with_labels(df['prop'], df['MAE'],       ax, fmt="{:.4f}")
    ax.set_ylabel('MAE'); ax.set_title('MAE: paper vs AE-run');
    ax.legend(['paper','AE-run']); ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=15); plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, 'MAE_compare.png'), dpi=300); plt.close(fig)

    # delta percent
    fig, ax = plt.subplots(figsize=(6,3))
    ax.scatter(df['prop'], df['delta_pct'], s=50)
    for x,y in zip(df['prop'], df['delta_pct']): ax.text(x,y+0.5,f"{y:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.axhline(0, linestyle='--', color='gray'); ax.set_ylabel('% Δ vs paper'); ax.set_title('% Improvement')
    ax.grid(axis='y', linestyle='--', alpha=0.5); plt.xticks(rotation=15); plt.tight_layout()
    fig.savefig(os.path.join(args.comparison_dir, 'delta_pct.png'), dpi=300); plt.close(fig)

    # R2 & RMSE bars
    for metric_name in ['R2','RMSE']:
        fig, ax = plt.subplots(figsize=(8,4))
        bar_with_labels(df['prop'], df[metric_name], ax, fmt="{:.3f}" if metric_name=='R2' else "{:.4f}")
        ax.set_ylabel(metric_name); ax.set_title(f"{metric_name} across props"); ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=15); plt.tight_layout()
        fig.savefig(os.path.join(args.comparison_dir, f"{metric_name}_across_props.png"), dpi=300); plt.close(fig)

    # summary table
    df.to_csv(os.path.join(args.comparison_dir, 'summary_table.csv'), index=False)
    md = ['|Property|MAE_paper|MAE_AE|RMSE|R2|Δ%|','|---|---|---|---|---|---|']
    for _,r in df.iterrows(): md.append(f"|{r['prop']}|{r['paper_MAE']:.4f}|{r['MAE']:.4f}|{r['RMSE']:.4f}|{r['R2']:.3f}|{r['delta_pct']:.1f}%|")
    with open(os.path.join(args.comparison_dir,'summary.md'),'w') as f: f.write('\n'.join(md))
    print(f"\nWritten visuals & tables to {args.comparison_dir}")

if __name__ == '__main__':
    main()

