"""
Batch family-level evaluation across multiple methods and datasets.
==================================================================
Walks through saved prediction files (from training runs), runs
family-level evaluation on each, and produces a comparison report.

Usage:
    uv run python scripts/run_family_eval_batch.py \
        --pred_root results/predictions \
        --datasets NSLKDD,UNSW
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

from eval import generate_summary_json, generate_summary_text, run_full_evaluation, save_report


def discover_prediction_dirs(pred_root: str, datasets: list[str] = None) -> list[dict]:
    """Walk through results/predictions/ to find all y_true.npy files.

    Expected layout:
        results/predictions/<dataset>/<algorithm>/<goal>/run_<i>/{y_true.npy, y_pred.npy}
    """
    pred_root = Path(pred_root)
    runs = []
    for ds_dir in sorted(pred_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        if datasets and ds_dir.name not in datasets:
            continue
        for algo_dir in sorted(ds_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for goal_dir in sorted(algo_dir.iterdir()):
                if not goal_dir.is_dir():
                    continue
                for run_dir in sorted(goal_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    y_true_path = run_dir / "y_true.npy"
                    y_pred_path = run_dir / "y_pred.npy"
                    if y_true_path.exists() and y_pred_path.exists():
                        try:
                            run_id = int(run_dir.name.replace("run_", ""))
                        except ValueError:
                            run_id = 0
                        runs.append({
                            "dataset": ds_dir.name,
                            "algorithm": algo_dir.name,
                            "goal": goal_dir.name,
                            "run_id": run_id,
                            "y_true_path": y_true_path,
                            "y_pred_path": y_pred_path,
                        })
    return runs


def main():
    parser = argparse.ArgumentParser(description="Batch family-level evaluation")
    parser.add_argument("--pred_root", "-p", type=str,
                        default="results/predictions",
                        help="Root directory with saved predictions")
    parser.add_argument("--datasets", "-d", type=str, default=None,
                        help="Comma-separated list of datasets to process (default: all)")
    parser.add_argument("--output_base", "-o", type=str,
                        default="results/family_eval",
                        help="Base output directory for evaluation reports")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    runs = discover_prediction_dirs(args.pred_root, datasets)

    if not runs:
        print("No prediction files found. Run training first, or check --pred_root.")
        return

    print(f"Found {len(runs)} prediction run(s):")
    for r in runs:
        print(f"  {r['dataset']}/{r['algorithm']}/{r['goal']}/run_{r['run_id']}")

    print("\nRunning family-level evaluation for each...")
    all_reports = []

    for r in runs:
        try:
            y_true = np.load(r["y_true_path"])
            y_pred = np.load(r["y_pred_path"])
        except Exception as e:
            print(f"  SKIP {r['dataset']}/{r['algorithm']}/run_{r['run_id']}: {e}")
            continue

        report = run_full_evaluation(
            y_true_ids=y_true,
            y_pred_ids=y_pred,
            dataset=r["dataset"],
            algorithm=r["algorithm"],
            goal=r["goal"],
            run_id=r["run_id"],
        )

        output_dir = os.path.join(
            args.output_base, r["dataset"], r["algorithm"],
            r["goal"], f"run_{r['run_id']}"
        )
        save_report(report, output_dir, fmt="json")
        all_reports.append(report)
        print(f"  OK {r['dataset']}/{r['algorithm']}/run_{r['run_id']}: "
              f"fine_acc={report.fine_grained.accuracy:.4f}, "
              f"family_acc={report.family_level.accuracy:.4f}, "
              f"gap={report.family_fine_accuracy_gap:+.4f}")

    # Generate cross-method comparison
    summary_text = generate_summary_text(all_reports)
    summary_json = generate_summary_json(all_reports)

    print("\n" + summary_text)

    # Save summaries
    summary_dir = os.path.join(args.output_base, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    text_path = os.path.join(summary_dir, "family_eval_summary.txt")
    json_path = os.path.join(summary_dir, "family_eval_summary.json")

    with open(text_path, "w") as f:
        f.write(summary_text)
    with open(json_path, "w") as f:
        import json
        json.dump(summary_json, f, indent=2)

    print(f"\nSummary saved to: {text_path}")
    print(f"JSON saved to:   {json_path}")


if __name__ == "__main__":
    main()
