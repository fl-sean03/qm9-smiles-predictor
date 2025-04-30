#!/usr/bin/env python3
"""
Script to compare evaluation metrics across different QM9 models.

Reads all_metrics.csv from specified model evaluation directories,
combines them, generates summary files (CSV and JSON), and creates
visualizations comparing key metrics for each property.
"""
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define units for each property
UNITS = {
    "U0": "eV",
    "U": "eV",
    "H": "eV",
    "G": "eV",
    "Cv": "cal mol$^{-1}$ K$^{-1}$",
    "homo": "eV",
    "gap": "eV",
    "mu": "Debye",
    "alpha": "a$_0^3$"
}

# Define the properties to compare
PROPERTIES = ["U0", "U", "H", "G", "Cv", "homo", "gap", "mu", "alpha"]

# Paper metrics from Pinheiro et al. (Table 3, "none" column MAE)
# Mapping paper property names to script property names
PAPER_METRICS_MAE = {
    "U0": 0.0573,
    "U": 0.0582,
    "H": 0.0575,
    "G": 0.0562,
    "Cv": 0.1223,
    "homo": 0.0952, # ϵHOMO in paper
    "gap": 0.1369,  # Δϵ in paper
    "mu": 0.523,
    "alpha": 0.3095,
}

def parse_args():
    p = argparse.ArgumentParser(description="Compare evaluation metrics across QM9 models")
    p.add_argument("--baseline_eval_dir", required=True,
                   help="Directory containing all_metrics.csv for the baseline model")
    p.add_argument("--multitask_eval_dir", required=True,
                   help="Directory containing all_metrics.csv for the multitask model")
    p.add_argument("--autoencoder_eval_dir", required=True,
                   help="Directory containing all_metrics.csv for the autoencoder model")
    p.add_argument("--hybrid_eval_dir", required=True,
                   help="Directory containing all_metrics.csv for the hybrid model")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write combined metrics CSV/JSON and visualizations")
    return p.parse_args()

def load_and_combine_metrics(args):
    """Loads all_metrics.csv from each directory and combines them, including paper metrics."""
    all_metrics_list = []

    model_dirs = {
        "baseline": args.baseline_eval_dir,
        "multitask": args.multitask_eval_dir,
        "autoencoder": args.autoencoder_eval_dir,
        "hybrid": args.hybrid_eval_dir,
    }

    for model_name, eval_dir in model_dirs.items():
        metrics_path = os.path.join(eval_dir, "all_metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Warning: {metrics_path} not found. Skipping {model_name} model.")
            continue
        df = pd.read_csv(metrics_path)
        df["model"] = model_name
        all_metrics_list.append(df)

    # Add paper metrics
    paper_metrics_data = []
    for prop, mae in PAPER_METRICS_MAE.items():
        # Assuming the paper only provides MAE, fill other metrics with NaN or a placeholder
        paper_metrics_data.append({"property": prop, "MAE": mae, "RMSE": None, "R2": None, "model": "Pinheiro et al."})

    paper_df = pd.DataFrame(paper_metrics_data)
    all_metrics_list.append(paper_df)


    if not all_metrics_list:
        raise FileNotFoundError("No metrics data (including paper metrics) were found.")

    combined_df = pd.concat(all_metrics_list, ignore_index=True)
    return combined_df

def generate_visualizations(df_combined, out_dir):
    """Generates and saves visualizations comparing metrics across models, including paper metrics."""
    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    # Order the models for the plot
    model_order = ["Pinheiro et al.", "baseline", "multitask", "autoencoder", "hybrid"]

    # Filter out rows where MAE is None (from paper metrics for other metrics)
    df_mae = df_combined.dropna(subset=['MAE']).copy()

    # Ensure the 'model' column is a categorical type with the desired order
    df_mae['model'] = pd.Categorical(df_mae['model'], categories=model_order, ordered=True)

    # Generate separate plots for each property
    for property_name in PROPERTIES:
        plt.figure(figsize=(10, 6)) # Adjust figure size for better readability

        # Filter data for the current property
        df_prop = df_mae[df_mae['property'] == property_name].copy()

        if df_prop.empty:
            print(f"No MAE data found for property: {property_name}. Skipping plot generation.")
            plt.close()
            continue

        sns.barplot(x="model", y="MAE", data=df_prop, palette="viridis")

        # Get the unit for the current property
        unit = UNITS.get(property_name, '')

        plt.title(f"MAE Comparison for {property_name}", fontsize=14)
        plt.ylabel(f"Mean Absolute Error (MAE) ({unit})", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        plot_path = os.path.join(vis_dir, f"mae_{property_name.lower()}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Generated MAE plot for {property_name}: {plot_path}")


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load and combine metrics
    df_combined = load_and_combine_metrics(args)

    # Save combined metrics to CSV and JSON
    summary_csv_path = os.path.join(args.out_dir, "summary.csv")
    df_combined.to_csv(summary_csv_path, index=False)
    print(f"Wrote combined metrics to {summary_csv_path}")

    summary_json_path = os.path.join(args.out_dir, "summary.json")
    df_combined.to_json(summary_json_path, orient="records", indent=2)
    print(f"Wrote combined metrics to {summary_json_path}")

    # Generate visualizations
    generate_visualizations(df_combined, args.out_dir)

if __name__ == "__main__":
    main()