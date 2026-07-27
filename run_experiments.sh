#!/bin/bash
# ==========================================================================
# FIDSUS rare-label comparison experiment (bash, for Linux / WSL)
#
# Produces a 3-caliber summary:
#   1. Personalized   (model_per, local)
#   2. Global head    (model_per.base + server global head)  [FIDSUS only]
#   3. Macro-F1       (class-balanced)
# Rare-class metrics are reported as tail-10-round mean +/- std.
#
# Usage:
#   bash run_experiments.sh                          # full experiment (100 rounds)
#   QUICK=1 bash run_experiments.sh                  # quick smoke (CPU, 2 rounds / 3 clients)
#   SUMMARY_ONLY=1 bash run_experiments.sh           # only summarize existing results
#   DATASETS="UNSW" ALGOS="FIDSUS FIDSUS_no_fusion" bash run_experiments.sh
#
# Recommended: uv run bash run_experiments.sh
# Or after `uv sync`, use the .venv python directly via PY=...
# ==========================================================================
set -e

cd "$(dirname "$0")/system"

QUICK="${QUICK:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
DEVICE="${DEVICE:-cuda}"
ROUNDS="${ROUNDS:-100}"
CLIENTS="${CLIENTS:-50}"
GOAL="${GOAL:-rare_label_exp}"
PY="${PY:-python}"

# Allow overrides; defaults reproduce the paper's main comparison
DATASETS="${DATASETS:-NSLKDD UNSW}"
ALGOS="${ALGOS:-FedAvg FIDSUS FIDSUS_no_fusion}"

if [ "$QUICK" = "1" ]; then
  ROUNDS=2
  CLIENTS=3
  DEVICE=cpu
  GOAL="quick_smoke"
  echo "[quick mode] rounds=2 clients=3 device=cpu"
fi

if [ "$SUMMARY_ONLY" = "1" ]; then
  echo "[summary-only] skipping training, aggregating existing results (goal=$GOAL)"
  # Pass datasets/algos as args to the summary
  $PY run_all.py --summary-only --goal "$GOAL" \
      --datasets $DATASETS --algos $ALGOS
  exit 0
fi

for dataset in $DATASETS; do
  case "$dataset" in
    NSLKDD) nb=5 ;;
    UNSW)   nb=10 ;;
    *)      echo "[ERROR] unknown dataset $dataset (expected NSLKDD or UNSW)"; exit 1 ;;
  esac
  for algo in $ALGOS; do
    echo "============================================================"
    echo "RUN: $algo on $dataset (rounds=$ROUNDS clients=$CLIENTS device=$DEVICE)"
    echo "============================================================"
    $PY main.py -algo "$algo" -data "$dataset" -nc "$CLIENTS" -nb "$nb" \
                -gr "$ROUNDS" -t 1 -dev "$DEVICE" -go "$GOAL"
  done
done

echo "============================================================"
echo "Training done. Aggregating 3-caliber summary..."
echo "============================================================"
$PY run_all.py --summary-only --goal "$GOAL" \
    --datasets $DATASETS --algos $ALGOS

echo ""
echo "============================================================"
echo "All done. Results in results/ , summary in results/summary.csv"
echo "Three calibers: personalized / global-head / Macro-F1"
echo "============================================================"
