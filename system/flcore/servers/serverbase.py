
import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data_un
from utils.metrics_utils import (
    new_confusion,
    add_confusion,
    compute_per_label_metrics,
    compute_macro_f1,
    get_rare_labels,
)


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.server_learning_rate = args.server_learning_rate
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.clients = []
        self.selected_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        # 逐标签 precision/recall 曲线（每轮一个长度为 num_classes 的数组）
        self.rs_test_precision = []
        self.rs_test_recall = []
        # 全局分类头评估口径的逐标签 precision/recall（仅 FIDSUS 系算法填充）
        self.rs_global_head_precision = []
        self.rs_global_head_recall = []
        # 每轮 Macro-F1（个性化口径 与 全局头口径）
        self.rs_macro_f1 = []
        self.rs_global_head_macro_f1 = []
        # 少数标签列表（用于打印与汇总）
        self.rare_labels = get_rare_labels(
            self.dataset,
            getattr(args, 'rare_target_labels', None),
        )

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_activity_rate = args.client_activity_rate
        self.batch_num_per_client = args.batch_num_per_client
    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            # train_data = read_client_data(self.dataset, i, is_train=True)
            # test_data = read_client_data(self.dataset, i, is_train=False)
            train_data = read_client_data_un(self.dataset, i, is_train=True)
            test_data = read_client_data_un(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), )
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients


    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(
            self.selected_clients, int(self.client_activity_rate * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                # 逐标签 precision / recall 曲线（形状: rounds × num_classes）
                if self.rs_test_precision:
                    hf.create_dataset('rs_test_precision', data=np.array(self.rs_test_precision))
                    hf.create_dataset('rs_test_recall', data=np.array(self.rs_test_recall))
                # 全局分类头口径的逐标签 precision / recall（仅 FIDSUS 系算法）
                if self.rs_global_head_precision:
                    hf.create_dataset('rs_global_head_precision',
                                      data=np.array(self.rs_global_head_precision))
                    hf.create_dataset('rs_global_head_recall',
                                      data=np.array(self.rs_global_head_recall))
                # 每轮 Macro-F1（个性化口径）
                if self.rs_macro_f1:
                    hf.create_dataset('rs_macro_f1', data=np.array(self.rs_macro_f1))
                # 每轮 Macro-F1（全局头口径，仅 FIDSUS 系算法）
                if self.rs_global_head_macro_f1:
                    hf.create_dataset('rs_global_head_macro_f1',
                                      data=np.array(self.rs_global_head_macro_f1))
                # 少数标签列表（便于下游汇总脚本识别）
                hf.create_dataset('rare_labels', data=np.array(self.rare_labels, dtype=np.int64))

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):


        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def test_metrics_per_label(self):
        """聚合所有客户端的逐标签混淆统计。

        Returns:
            confusion: dict with 'tp'/'fp'/'fn' numpy 数组（长度 num_classes）。
        """
        confusion = new_confusion(self.num_classes)
        for c in self.clients:
            if hasattr(c, 'test_metrics_per_label'):
                _, _, _, cm = c.test_metrics_per_label()
                confusion = add_confusion(confusion, cm)
        return confusion

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # 逐标签 precision / recall
        self._record_per_label_metrics()

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        self._print_rare_label_summary()

    def _record_per_label_metrics(self):
        """计算并记录本轮所有标签的 precision/recall 与 Macro-F1 到曲线数组。"""
        confusion = self.test_metrics_per_label()
        precision, recall = compute_per_label_metrics(
            confusion['tp'], confusion['fp'], confusion['fn'], self.num_classes
        )
        self.rs_test_precision.append(precision)
        self.rs_test_recall.append(recall)
        self.rs_macro_f1.append(compute_macro_f1(precision, recall))

    def _print_rare_label_summary(self):
        """打印少数标签的 precision/recall（若有定义）。"""
        if not self.rare_labels:
            return
        if not self.rs_test_precision:
            return
        prec = self.rs_test_precision[-1]
        rec = self.rs_test_recall[-1]
        parts = []
        for lbl in self.rare_labels:
            if 0 <= lbl < self.num_classes:
                parts.append(
                    "label{}: P={:.4f} R={:.4f}".format(lbl, prec[lbl], rec[lbl])
                )
        if parts:
            print("Rare labels | " + "  ".join(parts))

    def _print_rare_label_summary_global_head(self):
        """打印少数标签在「全局分类头」口径下的 precision/recall（仅 FIDSUS 系）。"""
        if not self.rare_labels:
            return
        if not self.rs_global_head_precision:
            return
        prec = self.rs_global_head_precision[-1]
        rec = self.rs_global_head_recall[-1]
        parts = []
        for lbl in self.rare_labels:
            if 0 <= lbl < self.num_classes:
                parts.append(
                    "label{}: P={:.4f} R={:.4f}".format(lbl, prec[lbl], rec[lbl])
                )
        if parts:
            print("Rare labels (global head) | " + "  ".join(parts))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

