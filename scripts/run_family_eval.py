"""
Family-level evaluation script for FIDSUS.
=========================================
Can be used in two modes:

1. **Standalone**: load pre-saved prediction files (y_true.npy + y_pred.npy)
   and run family-level evaluation.

2. **Training-integrated**: call `run_family_eval_standalone(dataset, algorithm, goal, run_id, y_true, y_pred)`
   from training code after prediction collection.

Usage (standalone):
    uv run python scripts/run_family_eval.py \
        --dataset NSLKDD \
        --algorithm FIDSUS \
        --goal test \
        --run 0 \
        --pred_dir results/predictions/NSLKDD/test
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Ensure system/ is on path for eval module imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

from eval import generate_summary_json, generate_summary_text, run_full_evaluation, save_report


def run_family_eval_standalone(
    dataset: str,
    algorithm: str,
    goal: str,
    run_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_base: str = "results/family_eval",
) -> dict:
    """Run full family evaluation given raw predictions.

    Parameters
    ----------
    dataset : str
        One of 'NSLKDD', 'UNSW', 'UAV-NIDD'.
    algorithm : str
        Algorithm name.
    goal : str
        Experiment goal.
    run_id : int
        Run index.
    y_true : np.ndarray
        Ground truth label ids.
    y_pred : np.ndarray
        Predicted label ids.
    output_base : str
        Base output directory.

    Returns
    -------
    dict with keys: report, summary_text, summary_json, saved_files
    """
    report = run_full_evaluation(
        y_true_ids=y_true,
        y_pred_ids=y_pred,
        dataset=dataset,
        algorithm=algorithm,
        goal=goal,
        run_id=run_id,
    )

    output_dir = os.path.join(output_base, dataset, algorithm, goal, f"run_{run_id}")
    saved_files = save_report(report, output_dir, fmt="json")

    summary_text = generate_summary_text([report])
    summary_json = generate_summary_json([report])

    print(f"\n{'=' * 60}")
    print(f"Family evaluation for {dataset} / {algorithm} / {goal} / run_{run_id}")
    print(f"{'=' * 60}")
    print(summary_text)

    return {
        "report": report,
        "summary_text": summary_text,
        "summary_json": summary_json,
        "saved_files": saved_files,
    }


def main():
    parser = argparse.ArgumentParser(description="FIDSUS Family-Level Evaluation")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Dataset name: NSLKDD, UNSW, or UAV-NIDD")
    parser.add_argument("--algorithm", "-a", type=str, default="FIDSUS",
                        help="Algorithm name")
    parser.add_argument("--goal", "-g", type=str, default="test",
                        help="Experiment goal")
    parser.add_argument("--run", "-r", type=int, default=0,
                        help="Run index")
    parser.add_argument("--pred_dir", "-p", type=str, required=True,
                        help="Directory containing y_true.npy and y_pred.npy")
    parser.add_argument("--output_base", "-o", type=str,
                        default="results/family_eval",
                        help="Base output directory")

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    y_true_path = pred_dir / "y_true.npy"
    y_pred_path = pred_dir / "y_pred.npy"

    if not y_true_path.exists():
        print(f"Error: y_true.npy not found at {y_true_path}")
        sys.exit(1)
    if not y_pred_path.exists():
        print(f"Error: y_pred.npy not found at {y_pred_path}")
        sys.exit(1)

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    print(f"Loaded y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    print(f"Unique true labels: {np.unique(y_true)}")
    print(f"Unique pred labels: {np.unique(y_pred)}")

    run_family_eval_standalone(
        dataset=args.dataset,
        algorithm=args.algorithm,
        goal=args.goal,
        run_id=args.run,
        y_true=y_true,
        y_pred=y_pred,
        output_base=str(ROOT / args.output_base),
    )


if __name__ == "__main__":
    main()
