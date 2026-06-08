"""
Compare ablation experiment results for FIDSUS Top-k Audit.

Requires separate runs of each ablation mode (original, random, entropy_aware, hics_style).
Each run produces per_round_metrics.csv and per_class_metrics.csv in its own subdirectory.

Usage:
    python scripts/compare_ablation_results.py
    python scripts/compare_ablation_results.py --base-dir docs/audit/topk/logs

Expected directory structure (per-experiment):
    docs/audit/topk/logs/original/per_round_metrics.csv
    docs/audit/topk/logs/random/per_round_metrics.csv
    docs/audit/topk/logs/entropy_aware/per_round_metrics.csv
    docs/audit/topk/logs/hics_style/per_round_metrics.csv

If logs are all in the same directory (single experiment mode), this script
cannot generate comparison charts — it will print instructions for how to
run multiple experiments.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_LABELS = {
    "UNSW": {
        0: "Normal", 1: "Backdoor", 2: "Analysis", 3: "Fuzzers",
        4: "Shellcode", 5: "Reconnaissance", 6: "Exploits", 7: "DoS",
        8: "Worms", 9: "Generic",
    },
    "NSLKDD": {
        0: "DoS", 1: "Probe", 2: "U2R", 3: "R2L", 4: "Normal",
    },
}

RARE_CLASSES = {
    "UNSW": {"Backdoor", "Analysis", "Shellcode", "Worms"},
    "NSLKDD": {"U2R", "R2L", "Probe"},
}


def load_metrics(log_dir: str) -> dict:
    """Load per-round and per-class metrics from a log directory."""
    result = {}
    pr_path = os.path.join(log_dir, "per_round_metrics.csv")
    pc_path = os.path.join(log_dir, "per_class_metrics.csv")

    if os.path.exists(pr_path):
        result["per_round"] = pd.read_csv(pr_path)
    if os.path.exists(pc_path):
        result["per_class"] = pd.read_csv(pc_path)
    return result


def compute_rare_class_recall(df_pc: pd.DataFrame, dataset: str) -> pd.Series:
    """Compute average rare-class recall per round."""
    rare = RARE_CLASSES.get(dataset, set())
    rare_df = df_pc[df_pc["class_name"].isin(rare)]
    if len(rare_df) == 0:
        return None
    return rare_df.groupby("round")["recall"].mean()


def main():
    parser = argparse.ArgumentParser(
        description="Compare Top-k ablation experiment results"
    )
    parser.add_argument(
        "--base-dir", default="docs/audit/topk/logs",
        help="Base directory containing experiment subdirectories"
    )
    parser.add_argument(
        "--output-dir", default="docs/audit/topk/assets",
        help="Output directory for charts"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    assets_dir = Path(args.output_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Check if we have per-experiment subdirectories
    experiment_modes = ["original", "random", "entropy_aware", "hics_style"]
    found_experiments = {}

    # First, check if files are directly in base_dir (single experiment)
    single_mode = os.path.exists(base_dir / "per_round_metrics.csv")

    if single_mode:
        print("=" * 70)
        print("Single experiment mode detected.")
        print("Comparison charts need results from MULTIPLE experiment runs.")
        print()
        print("To run all ablation experiments sequentially:")
        print("  cd system")
        print("  python main.py -c experiments/topk_audit_unsw.json")
        print()
        print("Then, after training, organize logs into subdirectories:")
        print("  mkdir -p docs/audit/topk/logs/{original,random,entropy_aware,hics_style}")
        print()
        print("Then run this script again.")
        print("=" * 70)

        # Still generate what we can from single experiment
        metrics = load_metrics(str(base_dir))

        if "per_round" in metrics:
            df = metrics["per_round"]
            if "macro_f1" not in df.columns:
                print("\nWARNING: per_round_metrics.csv missing macro_f1 column.")
                print("This suggests the audit was run before per-class metrics were added.")
                return

        sys.exit(0)

    # Check each experiment mode
    for mode in experiment_modes:
        mode_dir = base_dir / mode
        if mode_dir.exists():
            metrics = load_metrics(str(mode_dir))
            if metrics:
                found_experiments[mode] = metrics

    if len(found_experiments) < 2:
        print(f"Found only {len(found_experiments)} experiment(s): {list(found_experiments.keys())}")
        print("Need at least 2 experiments to generate comparison charts.")
        print(f"Missing: {set(experiment_modes) - set(found_experiments.keys())}")
        sys.exit(1)

    print(f"Found {len(found_experiments)} experiments: {list(found_experiments.keys())}")

    # Determine dataset from the data
    sample_df = list(found_experiments.values())[0].get("per_class")
    dataset = "UNSW"
    if sample_df is not None and len(sample_df) > 0:
        for cls_name in sample_df["class_name"].unique():
            if cls_name in {"DoS", "Probe", "U2R", "R2L"}:
                dataset = "NSLKDD"
                break

    rare = RARE_CLASSES.get(dataset, set())
    label_map = CLASS_LABELS.get(dataset, {})

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
    })

    # Chart: Accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    color_map = {
        "original": "#3498db",
        "random": "#e74c3c",
        "entropy_aware": "#2ecc71",
        "hics_style": "#9b59b6",
    }
    label_map_plot = {
        "original": "Original Top-k",
        "random": "Random-k",
        "entropy_aware": "Entropy-aware Top-k",
        "hics_style": "HiCS-style Top-k",
    }

    # Subplot 1: Accuracy
    ax = axes[0, 0]
    for mode, metrics in found_experiments.items():
        df = metrics.get("per_round")
        if df is not None and "accuracy" in df.columns:
            ax.plot(df["round"], df["accuracy"],
                    color=color_map.get(mode, "#000"),
                    label=label_map_plot.get(mode, mode),
                    linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Macro-F1
    ax = axes[0, 1]
    for mode, metrics in found_experiments.items():
        df = metrics.get("per_round")
        if df is not None and "macro_f1" in df.columns:
            ax.plot(df["round"], df["macro_f1"],
                    color=color_map.get(mode, "#000"),
                    label=label_map_plot.get(mode, mode),
                    linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Balanced Accuracy
    ax = axes[1, 0]
    for mode, metrics in found_experiments.items():
        df = metrics.get("per_round")
        if df is not None and "balanced_accuracy" in df.columns:
            ax.plot(df["round"], df["balanced_accuracy"],
                    color=color_map.get(mode, "#000"),
                    label=label_map_plot.get(mode, mode),
                    linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy Comparison")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Rare-class recall
    ax = axes[1, 1]
    for mode, metrics in found_experiments.items():
        df_pc = metrics.get("per_class")
        if df_pc is not None:
            rare_df = df_pc[df_pc["class_name"].isin(rare)]
            if len(rare_df) > 0:
                rare_recall = rare_df.groupby("round")["recall"].mean()
                ax.plot(rare_recall.index, rare_recall.values,
                        color=color_map.get(mode, "#000"),
                        label=label_map_plot.get(mode, mode),
                        linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Rare-Class Recall")
    ax.set_title(f"Rare-Class Recall ({', '.join(sorted(rare))})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Top-k Ablation Comparison ({dataset})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(assets_dir / "original_vs_random_vs_entropy_topk.png")
    plt.close(fig)
    print(f"-> original_vs_random_vs_entropy_topk.png")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Final Metrics Comparison")
    print("=" * 70)
    header = f"{'Metric':<25s}"
    for mode in experiment_modes:
        if mode in found_experiments:
            header += f" {label_map_plot.get(mode, mode):>20s}"
    print(header)
    print("-" * (25 + 21 * len(found_experiments)))

    for metric_name in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]:
        row = f"{metric_name:<25s}"
        for mode in experiment_modes:
            if mode in found_experiments:
                df = found_experiments[mode].get("per_round")
                if df is not None and metric_name in df.columns:
                    # Last 10 rounds average
                    last10 = df[metric_name].tail(10).mean()
                    row += f" {last10:>20.4f}"
                else:
                    row += f" {'N/A':>20s}"
        print(row)

    # Rare-class metrics
    print()
    for mode in experiment_modes:
        if mode in found_experiments:
            df_pc = found_experiments[mode].get("per_class")
            if df_pc is not None:
                last_round = df_pc["round"].max()
                df_last = df_pc[
                    (df_pc["round"] == last_round)
                    & (df_pc["class_name"].isin(rare))
                ]
                print(f"{label_map_plot.get(mode, mode)} (round {int(last_round)}):")
                for _, row in df_last.iterrows():
                    print(f"  {row['class_name']:<15s}  "
                          f"Recall={row['recall']:.4f}  F1={row['f1']:.4f}  "
                          f"Support={int(row['support'])}")
                print()

    print(f"All comparison charts saved to {assets_dir}/")


if __name__ == "__main__":
    main()
