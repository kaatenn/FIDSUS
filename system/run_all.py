#!/usr/bin/env python
"""一键运行「少数标签对比实验」。

覆盖需求 1-4：
  1. FIDSUS 在少数标签上的准确率/召回率
  2. FedAvg 在少数标签上的准确率/召回率
  3. FIDSUS_no_fusion（去掉跨轮特征融合）在少数标签上的准确率/召回率
  4. FIDSUS vs FIDSUS_no_fusion 收敛速度对比

用法（在 system/ 目录运行）：
  # 真实实验（100 轮，两数据集）—— 需 GPU
  uv run python run_all.py

  # 快速冒烟测试（CPU 也能跑，2 轮 / 3 客户端）
  uv run python run_all.py --quick

  # 仅指定数据集
  uv run python run_all.py --datasets UNSW
  uv run python run_all.py --datasets NSLKDD UNSW

本脚本通过 subprocess 调用 main.py 执行每个算法，最后汇总结果表格并写入
results/summary.csv。
"""

import argparse
import os
import subprocess
import sys
import time

# 数据集 → num_classes 映射
DATASET_NUM_CLASSES = {
    "NSLKDD": 5,
    "UNSW": 10,
}

# 默认实验算法
DEFAULT_ALGOS = ["FedAvg", "FIDSUS", "FIDSUS_no_fusion"]


def run_experiment(algo, dataset, num_classes, global_rounds, num_clients,
                   times, device, goal, extra_args):
    """调用 main.py 运行单个实验组合。"""
    cmd = [
        sys.executable, "main.py",
        "-algo", algo,
        "-data", dataset,
        "-nc", str(num_clients),
        "-nb", str(num_classes),
        "-gr", str(global_rounds),
        "-t", str(times),
        "-dev", device,
        "-go", goal,
    ]
    cmd += extra_args
    print("\n" + "=" * 70)
    print("RUN: {} on {} (rounds={}, clients={})".format(
        algo, dataset, global_rounds, num_clients))
    print("CMD:", " ".join(cmd))
    print("=" * 70)
    start = time.time()
    ret = subprocess.call(cmd)
    elapsed = time.time() - start
    status = "OK" if ret == 0 else "FAIL(ret={})".format(ret)
    print("[{}] {} on {} finished in {:.1f}s".format(status, algo, dataset, elapsed))
    return ret == 0


def summarize(datasets, algos, goal, times, results_dir):
    """读取所有运行结果，打印汇总表并写 CSV。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.result_utils import load_run_result
    from utils.metrics_utils import compute_convergence_speed

    rows = []
    for dataset in datasets:
        num_classes = DATASET_NUM_CLASSES.get(dataset, 10)
        for algo in algos:
            for t in range(times):
                res = load_run_result(dataset, algo, goal, time_idx=t,
                                      results_dir=results_dir)
                acc = res['rs_test_acc']
                if acc is None or len(acc) == 0:
                    continue
                conv = compute_convergence_speed(acc)
                rare = res['rare_labels']
                rare = list(rare.astype(int)) if rare is not None else []
                prec = res['rs_test_precision']
                rec = res['rs_test_recall']
                row = {
                    "dataset": dataset,
                    "algo": algo,
                    "run": t,
                    "best_acc": round(conv['best_acc'], 4),
                    "converge_round": conv['converge_round'],
                    "tail_std": round(conv['tail_std'], 4),
                }
                if prec is not None and rec is not None and len(rare) > 0:
                    for lbl in rare:
                        row["P_label{}".format(lbl)] = round(float(prec[-1, lbl]), 4)
                        row["R_label{}".format(lbl)] = round(float(rec[-1, lbl]), 4)
                rows.append(row)

    if not rows:
        print("\n[summary] 没有找到结果文件。请确认实验已运行。")
        return

    # 写 CSV
    csv_path = os.path.join(results_dir, "summary.csv")
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    os.makedirs(results_dir, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(all_keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in all_keys) + "\n")
    print("\n汇总已写入: {}".format(csv_path))

    # 打印表格
    print("\n" + "=" * 70)
    print("实验汇总（少数标签 precision/recall + 收敛速度）")
    print("=" * 70)
    for r in rows:
        rare_p = {k: v for k, v in r.items() if k.startswith("P_label")}
        rare_r = {k: v for k, v in r.items() if k.startswith("R_label")}
        print("\n[{} / {} / run{}]".format(r["dataset"], r["algo"], r["run"]))
        print("  Best Acc = {}  | 收敛轮次 = {}  | 末尾 std = {}".format(
            r["best_acc"], r["converge_round"], r["tail_std"]))
        if rare_p:
            print("  少数标签 Precision:", rare_p)
            print("  少数标签 Recall:   ", rare_r)

    # 收敛速度对比（需求 4）
    print("\n" + "-" * 70)
    print("收敛速度对比 (FIDSUS vs FIDSUS_no_fusion)")
    print("-" * 70)
    for dataset in datasets:
        for algo in ["FIDSUS", "FIDSUS_no_fusion"]:
            matches = [r for r in rows if r["dataset"] == dataset and r["algo"] == algo]
            if matches:
                r = matches[0]
                print("  {:<18} on {:<8}: best_acc={} converge_round={} tail_std={}".format(
                    algo, dataset, r["best_acc"], r["converge_round"], r["tail_std"]))


def main():
    parser = argparse.ArgumentParser(description="一键运行少数标签对比实验")
    parser.add_argument("--datasets", nargs="+", default=["NSLKDD", "UNSW"],
                        choices=list(DATASET_NUM_CLASSES.keys()),
                        help="参与实验的数据集")
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS,
                        help="参与实验的算法")
    parser.add_argument("--rounds", type=int, default=100,
                        help="全局轮次（默认 100）")
    parser.add_argument("--clients", type=int, default=50,
                        help="客户端数（默认 50）")
    parser.add_argument("--times", type=int, default=1,
                        help="每个组合重复次数（默认 1）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 cuda / cpu")
    parser.add_argument("--goal", type=str, default="rare_label_exp",
                        help="实验标识，用于结果文件命名")
    parser.add_argument("--quick", action="store_true",
                        help="快速冒烟模式：3 客户端 / 2 轮 / cpu，用于验证可运行")
    parser.add_argument("--summary-only", action="store_true",
                        help="跳过训练，仅汇总已有结果")
    args = parser.parse_args()

    if args.quick:
        args.rounds = 2
        args.clients = 3
        args.device = "cpu"
        args.goal = "quick_smoke"
        print("[quick 模式] rounds=2 clients=3 device=cpu")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

    if not args.summary_only:
        failures = []
        for dataset in args.datasets:
            num_classes = DATASET_NUM_CLASSES[dataset]
            for algo in args.algos:
                ok = run_experiment(
                    algo, dataset, num_classes,
                    global_rounds=args.rounds,
                    num_clients=args.clients,
                    times=args.times,
                    device=args.device,
                    goal=args.goal,
                    extra_args=[],
                )
                if not ok:
                    failures.append((algo, dataset))
        if failures:
            print("\n以下组合运行失败:", failures)

    summarize(args.datasets, args.algos, args.goal, args.times, results_dir)
    print("\n全部完成。")


if __name__ == "__main__":
    main()
