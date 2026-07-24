"""metrics_utils 单元测试：纯逻辑，无设备依赖。"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "system"))

from utils.metrics_utils import (
    compute_per_label_metrics,
    update_confusion_counts,
    new_confusion,
    add_confusion,
    compute_convergence_speed,
    get_rare_labels,
)


def test_perfect_predictions():
    tp = np.array([10, 5, 3])
    fp = np.array([0, 0, 0])
    fn = np.array([0, 0, 0])
    p, r = compute_per_label_metrics(tp, fp, fn, num_classes=3)
    assert np.allclose(p, 1.0)
    assert np.allclose(r, 1.0)


def test_all_wrong_predictions():
    # label 0 全部预测错
    tp = np.array([0, 5, 3])
    fp = np.array([10, 0, 0])  # label 0 被错误预测为很多次
    fn = np.array([10, 0, 0])  # label 0 真实样本全部漏掉
    p, r = compute_per_label_metrics(tp, fp, fn, num_classes=3)
    assert p[0] == 0.0
    assert r[0] == 0.0
    assert np.isclose(p[1], 1.0)
    assert np.isclose(r[1], 1.0)


def test_update_confusion_counts():
    c = new_confusion(3)
    y_true = [0, 0, 1, 2, 2]
    y_pred = [0, 1, 1, 2, 0]
    c = update_confusion_counts(c, y_true, y_pred, num_classes=3)
    # label 0: 1 correct, 1 missed (fn), 1 false alarm (fp from true=0 pred=1 -> fp for 1)
    assert c['tp'][0] == 1
    assert c['fn'][0] == 1  # one true-0 predicted as 1
    assert c['tp'][1] == 1
    assert c['fp'][1] == 1  # true-0 predicted as 1
    assert c['tp'][2] == 1
    assert c['fn'][2] == 1  # true-2 predicted as 0


def test_add_confusion():
    a = new_confusion(2)
    b = new_confusion(2)
    a['tp'][0] = 3
    b['tp'][0] = 5
    merged = add_confusion(a, b)
    assert merged['tp'][0] == 8


def test_compute_convergence_speed():
    # 单调上升后稳定
    acc = list(np.linspace(0.3, 0.9, 20)) + [0.9] * 10
    info = compute_convergence_speed(acc, threshold_ratio=0.9, tail=10)
    assert info['best_acc'] == 0.9
    assert info['converge_round'] <= info['best_round']
    assert info['tail_std'] < 0.01  # 末尾稳定


def test_compute_convergence_empty():
    info = compute_convergence_speed([])
    assert info['best_round'] == -1


def test_get_rare_labels():
    # 默认表
    assert get_rare_labels("NSLKDD") == [2]
    assert 8 in get_rare_labels("UNSW")
    # override
    assert get_rare_labels("UNSW", override=[1, 2]) == [1, 2]
    # 空 override 回退到默认
    assert get_rare_labels("UNSW", override=[]) == [8, 4, 1, 2]
    # 未知数据集
    assert get_rare_labels("UNKNOWN") == []
