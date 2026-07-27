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
    """读取所有运行结果，打印三口径并列汇总表并写 CSV。

    三口径：
      1. 个性化口径 (personalized)：model_per 本地模型
      2. 全局分类头口径 (global head)：model_per.base 特征 + 服务器全局 head
      3. Macro-F1：类别平衡指标
    稀有类一律用「末尾 10 轮均值±std」，过滤小样本噪声。
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.result_utils import summarize_rare_labels

    summaries = []
    for dataset in datasets:
        for algo in algos:
            for t in range(times):
                s = summarize_rare_labels(dataset, algo, goal, time_idx=t,
                                          results_dir=results_dir)
                if s['convergence'] is None:
                    continue
                s['dataset'] = dataset
                s['algo'] = algo
                s['run'] = t
                summaries.append(s)

    if not summaries:
        print("\n[summary] 没有找到结果文件。请确认实验已运行。")
        return

    # ---- 打印三口径并列表 ----
    print("\n" + "=" * 78)
    print("实验汇总（三口径并列：个性化 / 全局头 / Macro-F1）")
    print("稀有类指标为末尾 10 轮均值±std（过滤小样本噪声）")
    print("=" * 78)
    for s in summaries:
        conv = s['convergence']
        print("\n[{} / {} / run{}]".format(s["dataset"], s["algo"], s["run"]))
        print("  总体: Best Acc = {:.4f} | 收敛轮次 = {} | 末尾 std = {:.4f}".format(
            conv['best_acc'], conv['converge_round'], conv['tail_std']))
        print("  Macro-F1 (个性化):    {:.4f}".format(
            _fmt(s.get('macro_f1_tail'))))
        if s.get('global_head_macro_f1_tail') is not None:
            print("  Macro-F1 (全局头):    {:.4f}".format(
                _fmt(s.get('global_head_macro_f1_tail'))))
        for lbl in s['rare_labels']:
            pm = s['tail_precision_mean'].get(lbl, float('nan'))
            rm = s['tail_recall_mean'].get(lbl, float('nan'))
            rs = s['tail_recall_std'].get(lbl, float('nan'))
            print("  label{} 个性化   R = {:.4f} ± {:.4f} | P = {:.4f}".format(
                lbl, rm, rs, pm))
            if lbl in s.get('global_head_tail_recall_mean', {}):
                grm = s['global_head_tail_recall_mean'].get(lbl, float('nan'))
                grs = s['global_head_tail_recall_std'].get(lbl, float('nan'))
                print("  label{} 全局头   R = {:.4f} ± {:.4f}".format(lbl, grm, grs))

    # ---- 写 CSV（扁平化所有字段）----
    rows = []
    for s in summaries:
        conv = s['convergence']
        row = {
            "dataset": s["dataset"],
            "algo": s["algo"],
            "run": s["run"],
            "best_acc": round(conv['best_acc'], 4),
            "converge_round": conv['converge_round'],
            "tail_std": round(conv['tail_std'], 4),
            "macro_f1_tail": _fmt(s.get('macro_f1_tail')),
            "global_head_macro_f1_tail": _fmt(s.get('global_head_macro_f1_tail')),
        }
        for lbl in s['rare_labels']:
            row["pers_R{}_tail_mean".format(lbl)] = round(_fmt(s['tail_recall_mean'].get(lbl)), 4)
            row["pers_R{}_tail_std".format(lbl)] = round(_fmt(s['tail_recall_std'].get(lbl)), 4)
            row["pers_P{}_tail_mean".format(lbl)] = round(_fmt(s['tail_precision_mean'].get(lbl)), 4)
            if lbl in s.get('global_head_tail_recall_mean', {}):
                row["gh_R{}_tail_mean".format(lbl)] = round(_fmt(s['global_head_tail_recall_mean'].get(lbl)), 4)
                row["gh_R{}_tail_std".format(lbl)] = round(_fmt(s['global_head_tail_recall_std'].get(lbl)), 4)
        rows.append(row)

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

    # ---- 收敛速度对比（需求 4）----
    print("\n" + "-" * 78)
    print("收敛速度对比 (FIDSUS vs FIDSUS_no_fusion)")
    print("-" * 78)
    for dataset in datasets:
        for algo in ["FIDSUS", "FIDSUS_no_fusion"]:
            matches = [s for s in summaries if s["dataset"] == dataset and s["algo"] == algo]
            if matches:
                s = matches[0]
                conv = s['convergence']
                print("  {:<18} on {:<8}: best_acc={:.4f} converge_round={} tail_std={:.4f}".format(
                    algo, dataset, conv['best_acc'], conv['converge_round'], conv['tail_std']))


def _fmt(v):
    """None / nan -> 0.0 的辅助，便于格式化与 CSV。"""
    if v is None:
        return 0.0
    try:
        if v != v:  # nan
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


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
