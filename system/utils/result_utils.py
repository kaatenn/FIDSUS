
import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


# ---------------------------------------------------------------------------
# 以下为「少数标签对比实验」新增的辅助函数
# ---------------------------------------------------------------------------

def load_run_result(dataset, algorithm, goal, time_idx=0, results_dir="../results"):
    """读取某次运行的完整 h5 结果。

    Returns:
        dict, 包含 rs_test_acc / rs_test_auc / rs_train_loss /
        rs_test_precision / rs_test_recall / rare_labels（缺失键回退为 None）。
    """
    file_name = "{}_{}_{}_{}.h5".format(dataset, algorithm, goal, time_idx)
    file_path = os.path.join(results_dir, file_name)
    result = {
        'rs_test_acc': None,
        'rs_test_auc': None,
        'rs_train_loss': None,
        'rs_test_precision': None,
        'rs_test_recall': None,
        'rare_labels': None,
    }
    if not os.path.exists(file_path):
        return result
    with h5py.File(file_path, 'r') as hf:
        for key in result.keys():
            if key in hf:
                result[key] = np.array(hf.get(key))
    return result


def summarize_rare_labels(dataset, algorithm, goal, time_idx=0, results_dir="../results"):
    """汇总某次运行中少数标签的最终轮 precision/recall 与收敛速度。

    Returns:
        dict:
            - rare_labels: 标签列表
            - final_precision: 末轮各标签 precision
            - final_recall: 末轮各标签 recall
            - best_precision: 历史最佳各标签 precision
            - best_recall: 历史最佳各标签 recall
            - convergence: 总体准确率收敛指标 (dict)
    """
    from utils.metrics_utils import compute_convergence_speed
    res = load_run_result(dataset, algorithm, goal, time_idx, results_dir)
    out = {
        'rare_labels': [],
        'final_precision': {},
        'final_recall': {},
        'best_precision': {},
        'best_recall': {},
        'convergence': None,
    }
    acc = res['rs_test_acc']
    if acc is None:
        return out
    out['convergence'] = compute_convergence_speed(acc)

    rare = res['rare_labels']
    if rare is None or len(rare) == 0:
        return out
    out['rare_labels'] = list(rare.astype(int))

    prec = res['rs_test_precision']
    rec = res['rs_test_recall']
    if prec is None or rec is None:
        return out
    for lbl in out['rare_labels']:
        lbl_i = int(lbl)
        out['final_precision'][lbl_i] = float(prec[-1, lbl_i])
        out['final_recall'][lbl_i] = float(rec[-1, lbl_i])
        out['best_precision'][lbl_i] = float(np.max(prec[:, lbl_i]))
        out['best_recall'][lbl_i] = float(np.max(rec[:, lbl_i]))
    return out
