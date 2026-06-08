"""
Complete pipeline: train + family-level evaluation.
===================================================
1. Run training with the family_eval config (if predictions don't already exist)
2. Run batch family-level evaluation on all results
3. Print the comparison summary

Usage:
    uv run python scripts/run_family_eval_pipeline.py          # train + eval
    uv run python scripts/run_family_eval_pipeline.py --eval-only  # skip training
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_DIR = ROOT / "system"


def run_training(config: str) -> bool:
    """Run the main training pipeline. Returns True if successful."""
    print("\n" + "=" * 60)
    print("STEP 1: Training")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "main.py", "-c", config],
        cwd=SYSTEM_DIR,
    )
    return result.returncode == 0


def run_family_eval(datasets: list[str]) -> bool:
    """Run batch family-level evaluation. Returns True if successful."""
    print("\n" + "=" * 60)
    print("STEP 2: Family-Level Evaluation")
    print("=" * 60)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_family_eval_batch.py"),
            "--pred_root", str(ROOT / "results" / "predictions"),
            "--datasets", ",".join(datasets),
            "--output_base", str(ROOT / "results" / "family_eval"),
        ],
        cwd=ROOT,
    )
    return result.returncode == 0


def print_summary():
    """Print the saved summary if it exists."""
    summary_path = ROOT / "results" / "family_eval" / "summary" / "family_eval_summary.txt"
    if summary_path.exists():
        print("\n" + summary_path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="FIDSUS Family-Level Evaluation Pipeline"
    )
    parser.add_argument(
        "--config", "-c", type=str,
        default="experiments/family_eval.json",
        help="Experiment config file (inside system/experiments/)",
    )
    parser.add_argument(
        "--datasets", type=str, default="NSLKDD,UNSW",
        help="Datasets to evaluate (comma-separated)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only run evaluation on existing predictions",
    )
    args = parser.parse_args()

    config_path = args.config if "/" in args.config or args.config.startswith("experiments") else f"experiments/{args.config}"

    datasets = [d.strip() for d in args.datasets.split(",")]

    if not args.eval_only:
        ok = run_training(config_path)
        if not ok:
            print("\n[WARN] Training exited with errors. Proceeding to eval anyway.")
    else:
        print("Skipping training (--eval-only).")

    ok = run_family_eval(datasets)
    if ok:
        print_summary()


if __name__ == "__main__":
    main()
