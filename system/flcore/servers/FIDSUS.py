import torch
import time
import copy
import random
import numpy as np
from torch.utils.data import DataLoader

from flcore.clients.clientFIDSUS import clientFIDSUS
from flcore.servers.serverbase import Server

import torch.nn as nn

class FIDSUS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientFIDSUS)
        self.P = torch.diag(torch.ones(self.num_clients, device=self.device))
        self.uploaded_ids = []
        self.M = min(args.M, self.num_join_clients)
        self.client_models = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.head = self.client_models[0].head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)


    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i
            self.selected_clients = self.select_clients()
            self.send_models()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate_personalized()
            for client in self.selected_clients:
                client.train()
            self.receive_models()
            self.aggregate_parameters()
            self.train_head()
            if getattr(self.args, 'enable_affinity_diagnosis', False) and hasattr(self, 'diagnosis_logger'):
                self.diagnosis_logger.flush()
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def send_models(self):
        assert (len(self.selected_clients) > 0)
        for client in self.clients:
            start_time = time.time()

            M_ = min(self.M, len(self.uploaded_ids))
            indices = torch.topk(self.P[client.id], M_).indices.tolist()
            send_ids = []
            send_models = []
            for i in indices:
                send_ids.append(i)
                send_models.append(self.client_models[i])

            client.receive_models(send_ids, send_models)
            client.set_parameters(self.head)

            # Diagnosis: log top-n selection
            if getattr(self.args, 'enable_affinity_diagnosis', False) and hasattr(self, 'diagnosis_logger'):
                scores = self.P[client.id][indices].detach().cpu().tolist()
                self.diagnosis_logger.log_topn_selection(
                    round_num=self.current_round,
                    client_id=client.id,
                    neighbor_ids=indices,
                    affinity_scores=scores,
                    is_active=(client in self.selected_clients),
                    num_selected=len(self.selected_clients),
                    num_total=self.num_clients,
                    topn_size=len(indices))
                # Pass diagnosis_logger reference to client
                client.diagnosis_logger = self.diagnosis_logger

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int(self.client_activity_rate * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_protos = []
        self.uploaded_models = []
        tot_samples = 0
        enable_diag = getattr(self.args, 'enable_affinity_diagnosis', False) and hasattr(self, 'diagnosis_logger')
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                for cc in client.protos.keys():
                    y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                    self.uploaded_protos.append((client.protos[cc], y))
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                self.client_models[client.id] = copy.deepcopy(client.model)
                # Save old affinity before update
                if enable_diag:
                    old_P_row = self.P[client.id].detach().cpu().clone()
                self.P[client.id] += client.weight_vector
                # Log affinity update for each neighbor
                if enable_diag:
                    wv = client.weight_vector.detach().cpu()
                    for nid in range(self.num_clients):
                        delta = wv[nid].item()
                        if abs(delta) > 1e-12:
                            self.diagnosis_logger.log_affinity_update(
                                round_num=self.current_round,
                                client_id=client.id,
                                neighbor_id=nid,
                                old_affinity=old_P_row[nid].item(),
                                weight_delta=delta,
                                new_affinity=self.P[client.id][nid].item(),
                                L_old=getattr(client, '_diag_L_old', 0.0),
                                L_received=getattr(client, '_diag_L_received', 0.0),
                                param_distance=getattr(client, '_diag_param_norms', {}).get(nid, 0.0),
                                computed_weight=delta,
                                was_clipped=getattr(client, '_diag_was_clipped', False),
                                normalized_weight=getattr(client, '_diag_normalized_weights', {}).get(nid, None))
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def train_head(self):
        proto_loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        for p, y in proto_loader:
            out = self.head(p)
            loss = self.CEloss(out, y)
            self.opt_h.zero_grad()
            loss.backward()
            self.opt_h.step()
    def train_metrics_personalized(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses
    def test_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
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
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        # Diagnosis: log per-sample predictions for fine-grained eval
        if getattr(self.args, 'enable_affinity_diagnosis', False) and \
           getattr(self.args, 'enable_family_eval', False) and \
           hasattr(self, 'diagnosis_logger'):
            self._log_client_predictions()

    def _log_client_predictions(self):
        """Log per-sample predictions for all clients (diagnosis only)."""
        import torch.nn.functional as F

        family_mapping = getattr(self, 'family_mapping', None)
        label_names = {}
        if family_mapping and self.dataset in family_mapping:
            label_names = family_mapping[self.dataset].get('label_names', {})

        for client in self.clients:
            testloader = client.load_test_data()
            client.model_per.eval()
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client.model_per(x)
                    preds = torch.argmax(output, dim=1)
                    probs = F.softmax(output, dim=1)
                    confs = torch.max(probs, dim=1).values

                    for j in range(len(y)):
                        y_true_id = int(y[j].item())
                        y_pred_id = int(preds[j].item())
                        y_true_name = label_names.get(y_true_id, str(y_true_id))
                        y_pred_name = label_names.get(y_pred_id, str(y_pred_id))
                        self.diagnosis_logger.log_prediction(
                            round_num=self.current_round,
                            client_id=client.id,
                            y_true=y_true_id,
                            y_pred=y_pred_id,
                            y_true_name=y_true_name,
                            y_pred_name=y_pred_name,
                            confidence=confs[j].item())

