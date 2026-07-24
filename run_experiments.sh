#!/bin/bash
# 一键运行「少数标签对比实验」（bash 版，供 Linux / WSL 使用）。
#
# 覆盖需求 1-4：
#   1. FIDSUS 在少数标签上的准确率/召回率
#   2. FedAvg 在少数标签上的准确率/召回率
#   3. FIDSUS_no_fusion 在少数标签上的准确率/召回率
#   4. FIDSUS vs FIDSUS_no_fusion 收敛速度对比
#
# 用法：
#   bash run_experiments.sh           # 真实实验（100 轮，两数据集）
#   QUICK=1 bash run_experiments.sh   # 快速冒烟（CPU，2 轮 / 3 客户端）
#
# 推荐使用 uv 运行：uv run bash run_experiments.sh
# 或先 uv sync 后用 .venv 里的 python。

set -e

cd "$(dirname "$0")/system"

QUICK="${QUICK:-0}"
DEVICE="${DEVICE:-cuda}"
ROUNDS="${ROUNDS:-100}"
CLIENTS="${CLIENTS:-50}"
GOAL="${GOAL:-rare_label_exp}"
PY="${PY:-python}"

if [ "$QUICK" = "1" ]; then
  ROUNDS=2
  CLIENTS=3
  DEVICE=cpu
  GOAL="quick_smoke"
  echo "[quick 模式] rounds=2 clients=3 device=cpu"
fi

ALGOS=("FedAvg" "FIDSUS" "FIDSUS_no_fusion")
DATASETS=("NSLKDD" "UNSW")

declare -A NC=( ["NSLKDD"]=5 ["UNSW"]=10 )

for dataset in "${DATASETS[@]}"; do
  nb=${NC[$dataset]}
  for algo in "${ALGOS[@]}"; do
    echo "============================================================"
    echo "RUN: $algo on $dataset (rounds=$ROUNDS clients=$CLIENTS)"
    echo "============================================================"
    $PY main.py -algo "$algo" -data "$dataset" -nc "$CLIENTS" -nb "$nb" \
                -gr "$ROUNDS" -t 1 -dev "$DEVICE" -go "$GOAL"
  done
done

echo "============================================================"
echo "全部完成。结果位于 system/results/ 目录。"
echo "汇总请运行: $PY run_all.py --summary-only --goal $GOAL"
echo "============================================================"
