
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
        'rs_global_head_precision': None,
        'rs_global_head_recall': None,
        'rs_macro_f1': None,
        'rs_global_head_macro_f1': None,
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
    """汇总某次运行中少数标签的 precision/recall（个性化 + 全局头双口径）、
    末尾 N 轮均值±std、Macro-F1 与收敛速度。

    诚实性：稀有类指标以「末尾 tail 轮均值±std」为主（过滤小样本噪声），
    历史最佳 best_* 保留作参考（易被噪声虚高，不建议作为主结论）。

    Returns:
        dict:
            - rare_labels: 标签列表
            - final_precision / final_recall: 末轮各标签（个性化口径）
            - tail_precision_mean/std, tail_recall_mean/std: 末尾 tail 轮（个性化口径）
            - best_precision / best_recall: 历史最佳（个性化口径）
            - global_head_*: 全局分类头口径对应字段（仅 FIDSUS 系，否则空）
            - macro_f1_tail: 个性化口径末尾 Macro-F1 均值
            - global_head_macro_f1_tail: 全局头口径末尾 Macro-F1 均值（仅 FIDSUS 系）
            - convergence: 总体准确率收敛指标 (dict)
    """
    from utils.metrics_utils import compute_convergence_speed, tail_mean_std
    res = load_run_result(dataset, algorithm, goal, time_idx, results_dir)
    out = {
        'rare_labels': [],
        'final_precision': {},
        'final_recall': {},
        'tail_precision_mean': {},
        'tail_precision_std': {},
        'tail_recall_mean': {},
        'tail_recall_std': {},
        'best_precision': {},
        'best_recall': {},
        'global_head_final_precision': {},
        'global_head_final_recall': {},
        'global_head_tail_recall_mean': {},
        'global_head_tail_recall_std': {},
        'macro_f1_tail': None,
        'global_head_macro_f1_tail': None,
        'convergence': None,
    }
    acc = res['rs_test_acc']
    if acc is None:
        return out
    out['convergence'] = compute_convergence_speed(acc)

    # Macro-F1（个性化口径）末尾均值
    macro = res['rs_macro_f1']
    if macro is not None and len(macro) > 0:
        out['macro_f1_tail'] = tail_mean_std(macro)[0]
    # Macro-F1（全局头口径）末尾均值
    gh_macro = res['rs_global_head_macro_f1']
    if gh_macro is not None and len(gh_macro) > 0:
        out['global_head_macro_f1_tail'] = tail_mean_std(gh_macro)[0]

    rare = res['rare_labels']
    if rare is None or len(rare) == 0:
        return out
    out['rare_labels'] = list(rare.astype(int))

    # 个性化口径
    prec = res['rs_test_precision']
    rec = res['rs_test_recall']
    if prec is not None and rec is not None:
        for lbl in out['rare_labels']:
            lbl_i = int(lbl)
            out['final_precision'][lbl_i] = float(prec[-1, lbl_i])
            out['final_recall'][lbl_i] = float(rec[-1, lbl_i])
            pm, ps = tail_mean_std(prec[:, lbl_i])
            rm, rs = tail_mean_std(rec[:, lbl_i])
            out['tail_precision_mean'][lbl_i] = pm
            out['tail_precision_std'][lbl_i] = ps
            out['tail_recall_mean'][lbl_i] = rm
            out['tail_recall_std'][lbl_i] = rs
            out['best_precision'][lbl_i] = float(np.max(prec[:, lbl_i]))
            out['best_recall'][lbl_i] = float(np.max(rec[:, lbl_i]))

    # 全局分类头口径（仅 FIDSUS 系算法有）
    gh_prec = res['rs_global_head_precision']
    gh_rec = res['rs_global_head_recall']
    if gh_prec is not None and gh_rec is not None:
        for lbl in out['rare_labels']:
            lbl_i = int(lbl)
            out['global_head_final_precision'][lbl_i] = float(gh_prec[-1, lbl_i])
            out['global_head_final_recall'][lbl_i] = float(gh_rec[-1, lbl_i])
            rm, rs = tail_mean_std(gh_rec[:, lbl_i])
            out['global_head_tail_recall_mean'][lbl_i] = rm
            out['global_head_tail_recall_std'][lbl_i] = rs
    return out
