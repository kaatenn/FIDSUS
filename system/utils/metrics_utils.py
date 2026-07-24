"""标签级指标与收敛速度分析工具。

本模块提供：
1. 基于逐标签 TP/FP/FN 统计计算每个标签的 precision / recall；
2. 从准确率曲线计算收敛速度（达到阈值轮次、末尾稳定性）。

设计原则：纯 numpy 实现，无设备依赖，便于单元测试。
"""

import numpy as np


def compute_per_label_metrics(tp, fp, fn, num_classes):
    """根据逐标签的 TP/FP/FN 计算每个标签的 precision 和 recall。

    Args:
        tp: 长度为 num_classes 的数组或列表，每个标签的真正例数。
        fp: 长度为 num_classes 的数组或列表，每个标签的假正例数。
        fn: 长度为 num_classes 的数组或列表，每个标签的假反例数。
        num_classes: 类别总数。

    Returns:
        (precision, recall): 两个长度为 num_classes 的 numpy 数组。
            某标签无预测(TP+FP=0)时 precision 记 0；无真实样本(TP+FN=0)时 recall 记 0。
    """
    tp = np.asarray(tp, dtype=np.float64).reshape(-1)
    fp = np.asarray(fp, dtype=np.float64).reshape(-1)
    fn = np.asarray(fn, dtype=np.float64).reshape(-1)

    # 兼容输入长度不足 num_classes 的情况（补 0）
    def _pad(arr):
        if len(arr) < num_classes:
            arr = np.concatenate([arr, np.zeros(num_classes - len(arr))])
        return arr[:num_classes]

    tp = _pad(tp)
    fp = _pad(fp)
    fn = _pad(fn)

    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)

    denom_p = tp + fp
    denom_r = tp + fn
    mask_p = denom_p > 0
    mask_r = denom_r > 0
    precision[mask_p] = tp[mask_p] / denom_p[mask_p]
    recall[mask_r] = tp[mask_r] / denom_r[mask_r]

    return precision, recall


def update_confusion_counts(confusion, y_true, y_pred, num_classes):
    """根据一批真实标签和预测标签增量更新逐标签 TP/FP/FN。

    confusion 是一个 dict，包含 'tp'/'fp'/'fn' 三个 numpy 数组（会被原地修改）。
    若 confusion 为 None，则新建。

    Args:
        confusion: dict 或 None。
        y_true: 真实标签的类数组。
        y_pred: 预测标签的类数组。
        num_classes: 类别总数。

    Returns:
        更新后的 confusion dict。
    """
    if confusion is None:
        confusion = {
            'tp': np.zeros(num_classes, dtype=np.int64),
            'fp': np.zeros(num_classes, dtype=np.int64),
            'fn': np.zeros(num_classes, dtype=np.int64),
        }
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)
        if t == p:
            if 0 <= t < num_classes:
                confusion['tp'][t] += 1
        else:
            if 0 <= p < num_classes:
                confusion['fp'][p] += 1
            if 0 <= t < num_classes:
                confusion['fn'][t] += 1
    return confusion


def new_confusion(num_classes):
    """创建空的逐标签混淆统计字典。"""
    return {
        'tp': np.zeros(num_classes, dtype=np.int64),
        'fp': np.zeros(num_classes, dtype=np.int64),
        'fn': np.zeros(num_classes, dtype=np.int64),
    }


def add_confusion(a, b):
    """合并两个 confusion dict（对应数组相加）。"""
    return {
        'tp': a['tp'] + b['tp'],
        'fp': a['fp'] + b['fp'],
        'fn': a['fn'] + b['fn'],
    }


def compute_convergence_speed(acc_list, threshold_ratio=0.9, tail=10):
    """从准确率曲线计算收敛速度指标。

    Args:
        acc_list: 每轮的测试准确率（list 或 1D 数组）。
        threshold_ratio: 认为收敛时达到的最佳准确率比例，默认 0.9。
        tail: 计算末尾稳定性时使用的最后 N 轮，默认 10。

    Returns:
        dict:
            - best_acc: 最高准确率
            - best_round: 达到最高准确率的轮次索引
            - converge_round: 首次达到 threshold_ratio * best_acc 的轮次索引
            - tail_std: 末尾 tail 轮准确率标准差（越小越稳定）
            - tail_mean: 末尾 tail 轮准确率均值
    """
    acc = np.asarray(acc_list, dtype=np.float64).reshape(-1)
    if len(acc) == 0:
        return {
            'best_acc': float('nan'),
            'best_round': -1,
            'converge_round': -1,
            'tail_std': float('nan'),
            'tail_mean': float('nan'),
        }
    best_acc = float(np.max(acc))
    best_round = int(np.argmax(acc))
    target = threshold_ratio * best_acc
    # 首次达到目标的轮次
    reach = np.where(acc >= target)[0]
    converge_round = int(reach[0]) if len(reach) > 0 else best_round
    tail_arr = acc[-tail:] if len(acc) >= tail else acc
    return {
        'best_acc': best_acc,
        'best_round': best_round,
        'converge_round': converge_round,
        'tail_std': float(np.std(tail_arr)),
        'tail_mean': float(np.mean(tail_arr)),
    }


# 各数据集的"少数标签"参考（基于全局训练样本量统计）。
# 可在运行时通过命令行 --rare_target_labels 覆盖。
RARE_LABELS = {
    'NSLKDD': [2],                      # 2: 仅 87 样本
    'UNSW': [8, 4, 1, 2],               # 8:139, 4:1210, 1:1824, 2:2154
    'UAV-NIDD': [],
    'mnist': [],
    'FashionMNIST': [],
}


def get_rare_labels(dataset, override=None):
    """获取某数据集的少数标签列表。

    Args:
        dataset: 数据集名称。
        override: 若非 None，则直接使用该列表（来自命令行覆盖）。

    Returns:
        标签列表。override 为空列表时视为"不限定"返回 []。
    """
    if override is not None and len(override) > 0:
        return list(override)
    return list(RARE_LABELS.get(dataset, []))
