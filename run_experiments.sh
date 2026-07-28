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
#   TORCH_VARIANT=cu126 bash run_experiments.sh      # override auto-detection
#
# torch CUDA variant auto-detection (via nvidia-smi compute_cap):
#   compute_cap <  7.5  -> cu118  (Pascal/Maxwell, e.g. GT 1030 sm_61)
#   compute_cap >= 7.5  -> cu126  (Turing and newer, e.g. RTX 4090 sm_89)
#   no GPU              -> cpu
# Override with TORCH_VARIANT=cu118|cu126|cpu.
# ==========================================================================

cd "$(dirname "$0")" || exit 1
ROOT="$PWD"

QUICK="${QUICK:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
DEVICE="${DEVICE:-cuda}"
ROUNDS="${ROUNDS:-100}"
CLIENTS="${CLIENTS:-50}"
GOAL="${GOAL:-rare_label_exp}"

# Allow overrides; defaults reproduce the paper's main comparison
DATASETS="${DATASETS:-NSLKDD UNSW}"
ALGOS="${ALGOS:-FedAvg FIDSUS FIDSUS_no_fusion}"

# NOTE: do NOT use `set -e`. Long experiments should not abort the whole run
# if a single subprocess crashes; failures are collected and reported.

# ==========================================================================
# 1. Detect GPU compute capability -> choose torch CUDA variant
# ==========================================================================
detect_torch_variant() {
  # Honor explicit override
  case "${TORCH_VARIANT:-}" in
    cu118|cu126|cpu) echo "$TORCH_VARIANT"; return 0 ;;
  esac

  # Find nvidia-smi
  local nsmi=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nsmi="nvidia-smi"
  fi

  if [ -z "$nsmi" ]; then
    echo "cpu"; return 0
  fi

  # Query compute capability of GPU 0 (e.g. "7.5", "8.9", "6.1")
  local cap
  cap="$($nsmi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  if [ -z "$cap" ]; then
    echo "cpu"; return 0
  fi

  # Compare as integer major*10+minor: "7.5" -> 75, "6.1" -> 61, "8.9" -> 89
  local major minor cc
  major="${cap%%.*}"
  minor="${cap#*.}"
  [ "$minor" = "$cap" ] && minor=0
  cc=$(( major * 10 + minor ))

  if [ "$cc" -lt 75 ]; then
    echo "cu118"   # Pascal/Maxwell: needs the last sm_61 build (torch 2.6.*+cu118)
  else
    echo "cu126"   # Turing and newer: stable sm_89 support
  fi
}

# ==========================================================================
# 2. Resolve Python interpreter + ensure uv environment with correct torch
# ==========================================================================
VARIANT="$(detect_torch_variant)"
echo "[1/4] torch variant: $VARIANT"

# Install uv if missing (uv is the intended environment manager for this project)
if ! command -v uv >/dev/null 2>&1; then
  echo "  uv not found, installing via official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090,SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.cargo/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv still not available. Install manually: https://docs.astral.sh/uv/"
  exit 1
fi

# Sync dependencies with the detected CUDA variant extra
echo "  running: uv sync --extra $VARIANT --group dev"
if ! uv sync --extra "$VARIANT" --group dev; then
  echo "[ERROR] uv sync failed for variant=$VARIANT"
  exit 1
fi
PY="uv run python"

# ==========================================================================
# 3. Verify CUDA actually works (warn loudly if torch is CPU-only)
# ==========================================================================
echo "[2/4] verifying torch / CUDA ..."
CUDA_INFO="$($PY -c 'import torch; print(torch.__version__); print("1" if torch.cuda.is_available() else "0"); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null)"
TORCH_VER="$(echo "$CUDA_INFO" | sed -n '1p')"
CUDA_OK="$(echo "$CUDA_INFO" | sed -n '2p')"
DEV_NAME="$(echo "$CUDA_INFO" | sed -n '3p')"
echo "  torch=$TORCH_VER  cuda_available=$CUDA_OK  device=$DEV_NAME"

if [ "$CUDA_OK" = "0" ]; then
  echo "  ----------------------------------------------------------------"
  echo "  [WARN] torch is CPU-only but DEVICE=$DEVICE requested CUDA."
  echo "  This usually means:"
  echo "    - nvidia driver missing, OR"
  echo "    - torch installed from a CPU index (check 'uv pip list | grep torch')"
  echo "  Falling back to CPU training (will be slow)."
  echo "  To force a CUDA rebuild: TORCH_VARIANT=cu126 uv sync --extra cu126"
  echo "  ----------------------------------------------------------------"
  DEVICE="cpu"
fi

cd "$ROOT/system" || exit 1

if [ "$QUICK" = "1" ]; then
  ROUNDS=2
  CLIENTS=3
  DEVICE=cpu
  GOAL="quick_smoke"
  echo "[quick mode] rounds=2 clients=3 device=cpu"
fi

# ==========================================================================
# 4. Summary-only mode: skip training
# ==========================================================================
if [ "$SUMMARY_ONLY" = "1" ]; then
  echo "[3/4] summary-only: skipping training (goal=$GOAL)"
  $PY run_all.py --summary-only --goal "$GOAL" \
      --datasets $DATASETS --algos $ALGOS || echo "[WARN] summary step failed"
  exit 0
fi

# ==========================================================================
# 5. Training loop (fault-tolerant: one failure does not abort the rest)
# ==========================================================================
echo "[3/4] training sweep (device=$DEVICE) ..."
FAILURES=""
for dataset in $DATASETS; do
  case "$dataset" in
    NSLKDD) nb=5 ;;
    UNSW)   nb=10 ;;
    *)      echo "[ERROR] unknown dataset '$dataset' (expected NSLKDD or UNSW)"; exit 1 ;;
  esac
  for algo in $ALGOS; do
    echo "============================================================"
    echo "RUN: $algo on $dataset (rounds=$ROUNDS clients=$CLIENTS device=$DEVICE)"
    echo "============================================================"
    if $PY main.py -algo "$algo" -data "$dataset" -nc "$CLIENTS" -nb "$nb" \
                -gr "$ROUNDS" -t 1 -dev "$DEVICE" -go "$GOAL"; then
      echo "[OK] $algo on $dataset"
    else
      rc=$?
      echo "[FAIL] $algo on $dataset (exit=$rc) — continuing with remaining combos"
      FAILURES="$FAILURES $algo/$dataset"
    fi
  done
done

echo "============================================================"
echo "[4/4] aggregating 3-caliber summary..."
echo "============================================================"
$PY run_all.py --summary-only --goal "$GOAL" \
    --datasets $DATASETS --algos $ALGOS || echo "[WARN] summary step failed"

echo ""
echo "============================================================"
echo "All done. Results in results/ , summary in results/summary.csv"
echo "Three calibers: personalized / global-head / Macro-F1"
if [ -n "$FAILURES" ]; then
  echo "[WARNING] these combos failed (partial results only):$FAILURES"
fi
echo "============================================================"
