"""
Instrumented FIDSUS server for Top-k audit.

Extends the original FIDSUS server with comprehensive audit logging:
- Client label distribution analysis
- Top-k selection logging (per round, per client, per neighbor)
- Candidate ranking logging (periodic)
- Affinity matrix snapshots
- Per-class metrics (macro-F1, balanced accuracy, per-class recall/precision/F1)
- Confusion matrices

Also supports ablation modes via config:
- Original Top-k (baseline)
- Random-k (random neighbor selection)
- Entropy-aware Top-k (affinity + lambda * entropy)
- HiCS-style Top-k (affinity + lambda * estimated_entropy)
"""

import copy
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flcore.clients.audited_clientFIDSUS import audited_clientFIDSUS
from flcore.servers.FIDSUS import FIDSUS
from audit.instrument import (
    AuditLogger,
    classify_client_tags,
    classify_client_type,
    compute_client_label_stats,
    compute_comprehensive_metrics,
    compute_js_divergence,
    compute_label_entropy,
    compute_norm_entropy,
    get_label_map,
    get_num_classes,
    get_rare_classes,
    load_all_client_labels,
)


class AuditedFIDSUS(FIDSUS):
    """Instrumented FIDSUS server with Top-k audit logging.

    Additional config parameters:
        audit_output_dir: str = "docs/audit/topk"
        ablation_mode: str = "original" | "random" | "entropy_aware" | "hics_style"
        entropy_lambda: float = 0.5  (for entropy-aware / hics modes)
        rare_coverage_mu: float = 0.1  (for entropy-aware mode)
        candidate_log_interval: int = 5  (log candidate rankings every N rounds)
        affinity_save_interval: int = 5  (save affinity matrix every N rounds)
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # Audit configuration
        self.audit_output_dir = getattr(args, "audit_output_dir", "docs/audit/topk")
        self.ablation_mode = getattr(args, "ablation_mode", "original")
        self.entropy_lambda = getattr(args, "entropy_lambda", 0.5)
        self.rare_coverage_mu = getattr(args, "rare_coverage_mu", 0.1)
        self.candidate_log_interval = getattr(args, "candidate_log_interval", 5)
        self.affinity_save_interval = getattr(args, "affinity_save_interval", 5)
        self.is_audit_run = True

        # Set up audit logger
        self.audit_logger = AuditLogger(
            output_dir=os.path.join(self.audit_output_dir, "logs")
        )

        # Load client label statistics
        print("\n[Audit] Computing client label distributions...")
        self.all_client_labels = load_all_client_labels(
            self.dataset, self.num_clients,
            root_dir=os.path.join("..", "dataset"),
        )
        self.client_stats = compute_client_label_stats(
            self.all_client_labels, self.num_clients, self.dataset
        )
        self.audit_logger.save_client_distribution(self.client_stats, self.dataset)

        # Build client stats map for fast lookup
        self.client_stats_map = {s["client_id"]: s for s in self.client_stats}

        # Set up client type classification
        self.label_map = get_label_map(self.dataset)
        self.num_classes_audit = get_num_classes(self.dataset)
        self.rare_classes = get_rare_classes(self.dataset)

        # Print summary
        n_balanced = sum(
            1 for s in self.client_stats
            if classify_client_type(s) == "balanced"
        )
        n_imbalanced = sum(
            1 for s in self.client_stats
            if classify_client_type(s) == "severely_imbalanced"
        )
        n_middle = sum(
            1 for s in self.client_stats
            if classify_client_type(s) == "middle"
        )
        n_rare = sum(
            1 for s in self.client_stats
            if s["rare_attack_sample_count"] > 0
        )
        print(f"  Clients: {n_balanced} balanced, {n_imbalanced} severely imbalanced, "
              f"{n_middle} middle, {n_rare} with rare attacks")
        print(f"  Ablation mode: {self.ablation_mode}")
        if self.ablation_mode in ("entropy_aware", "hics_style"):
            print(f"  Entropy lambda: {self.entropy_lambda}")
        print(f"  Audit output: {self.audit_output_dir}")

    # Override set_clients to use audited client class
    def set_clients(self, clientObj):
        from utils.data_utils import read_client_data_un

        for i in range(self.num_clients):
            train_data = read_client_data_un(self.dataset, i, is_train=True)
            test_data = read_client_data_un(self.dataset, i, is_train=False)
            client = audited_clientFIDSUS(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
            )
            self.clients.append(client)

    # ── Override send_models for ablation modes ───────────────────────

    def send_models(self):
        assert len(self.selected_clients) > 0

        for client in self.clients:
            start_time = time.time()
            M_ = min(self.M, len(self.uploaded_ids))

            if self.ablation_mode == "random":
                indices = self._select_random_k(client, M_)
            elif self.ablation_mode == "entropy_aware":
                indices = self._select_entropy_aware_k(client, M_)
            elif self.ablation_mode == "hics_style":
                indices = self._select_hics_style_k(client, M_)
            else:
                # Original Top-k
                indices = torch.topk(self.P[client.id], M_).indices.tolist()

            send_ids = []
            send_models = []
            for i in indices:
                send_ids.append(i)
                send_models.append(self.client_models[i])

            client.receive_models(send_ids, send_models)
            client.set_parameters(self.head)
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def _select_random_k(self, client, M_):
        """Select k random neighbors (excluding self)."""
        candidates = [i for i in range(self.num_clients) if i != client.id]
        return random.sample(candidates, min(M_, len(candidates)))

    def _select_entropy_aware_k(self, client, M_):
        """Select top-k using: score = affinity + lambda * entropy."""
        scores = []
        for cid in range(self.num_clients):
            if cid == client.id:
                scores.append(-float("inf"))
                continue
            affinity = self.P[client.id, cid].item()
            ent = self.client_stats_map[cid]["normalized_label_entropy"]
            rare_bonus = (
                self.rare_coverage_mu
                if self.client_stats_map[cid]["rare_attack_sample_count"] > 0
                else 0.0
            )
            score = affinity + self.entropy_lambda * ent + rare_bonus
            scores.append(score)

        scores_t = torch.tensor(scores, device=self.device)
        return torch.topk(scores_t, M_).indices.tolist()

    def _select_hics_style_k(self, client, M_):
        """Select top-k using estimated entropy from head bias update.

        HiCS-FL style: H_hat_i = H(softmax(delta_bias / T))
        score = affinity + lambda * H_hat_i

        Uses the head bias from each client's model_per as a proxy.
        """
        T = getattr(self.args, "hics_temperature", 1.0)
        scores = []
        for cid in range(self.num_clients):
            if cid == client.id:
                scores.append(-float("inf"))
                continue
            affinity = self.P[client.id, cid].item()

            # Estimate entropy from classifier head bias
            client_model = self.client_models[cid]
            head_bias = None
            for name, param in client_model.head.named_parameters():
                if "bias" in name:
                    head_bias = param.data
                    break

            if head_bias is not None:
                # Avoid modifying original
                # delta_bias = head_bias (absolute bias, not delta)
                # since we don't track previous bias
                logits = head_bias / T
                probs = torch.softmax(logits, dim=0)
                probs_np = probs.detach().cpu().numpy()
                probs_np = np.maximum(probs_np, 1e-12)
                probs_np /= probs_np.sum()
                H_hat = float(-np.sum(probs_np * np.log(probs_np)))
            else:
                H_hat = 0.0

            score = affinity + self.entropy_lambda * H_hat
            scores.append(score)

        scores_t = torch.tensor(scores, device=self.device)
        return torch.topk(scores_t, M_).indices.tolist()

    # ── Override train for audit logging ──────────────────────────────

    def train(self):
        """Main training loop with audit logging."""
        # Open metrics logs
        self.audit_logger.open_metrics_logs()
        self.audit_logger.open_topk_log()

        # Track round counter for periodic logging
        self._audit_round = 0

        for i in range(self.global_rounds + 1):
            self._audit_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # Log top-k selections
            if i % self.eval_gap == 0:
                self._log_topk_selections(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate_personalized()
                # Compute comprehensive metrics
                self._compute_and_log_metrics(i)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.train_head()
            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            # Periodic candidate ranking log
            if (
                i % self.candidate_log_interval == 0
                and i > 0
            ):
                self._log_candidate_rankings(i)

            # Periodic affinity matrix save
            if i % self.affinity_save_interval == 0:
                self.audit_logger.save_affinity_matrix(
                    self.P, i, self.client_stats
                )

        # End of training: save final affinity matrix
        self.audit_logger.save_affinity_matrix(
            self.P, self.global_rounds, self.client_stats
        )
        self._log_candidate_rankings(self.global_rounds)

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

        # Close audit logs
        self.audit_logger.close_all()

        # Save audit metadata
        self._save_audit_metadata()

    # ── Audit logging methods ─────────────────────────────────────────

    def _log_topk_selections(self, round_num: int):
        """Log top-k selections for all clients in this round."""
        active_ids = [c.id for c in self.selected_clients]

        for client in self.clients:
            M_ = min(self.M, len(self.uploaded_ids))
            if M_ == 0:
                continue

            # Determine which indices were selected
            if self.ablation_mode == "random":
                indices = self._select_random_k(client, M_)
            elif self.ablation_mode == "entropy_aware":
                indices = self._select_entropy_aware_k(client, M_)
            elif self.ablation_mode == "hics_style":
                indices = self._select_hics_style_k(client, M_)
            else:
                indices = torch.topk(self.P[client.id], M_).indices.tolist()

            affinity_scores = [
                self.P[client.id, nid].item() for nid in indices
            ]

            neighbor_stats = [
                self.client_stats_map.get(nid, self.client_stats_map[0])
                for nid in indices
            ]

            self.audit_logger.log_topk_selection(
                round_num=round_num,
                target_client_id=client.id,
                active_clients=active_ids,
                selected_topk_neighbors=indices,
                affinity_scores=affinity_scores,
                target_stats=self.client_stats_map[client.id],
                neighbor_stats_list=neighbor_stats,
                dataset=self.dataset,
            )

    def _log_candidate_rankings(self, round_num: int):
        """Log full candidate ranking for each client."""
        for client in self.clients:
            M_ = min(self.M, self.num_clients - 1)

            # Build candidate list
            candidates = []
            for cid in range(self.num_clients):
                if cid != client.id:
                    candidates.append((cid, self.P[client.id, cid].item()))

            # Determine top-k set
            if self.ablation_mode == "random":
                topk_set = set(self._select_random_k(client, M_))
            elif self.ablation_mode == "entropy_aware":
                topk_set = set(self._select_entropy_aware_k(client, M_))
            elif self.ablation_mode == "hics_style":
                topk_set = set(self._select_hics_style_k(client, M_))
            else:
                topk_set = set(
                    torch.topk(self.P[client.id], M_).indices.tolist()
                )

            self.audit_logger.log_candidate_ranking(
                round_num=round_num,
                target_client_id=client.id,
                candidate_affinities=candidates,
                topk_indices=topk_set,
                client_stats_map=self.client_stats_map,
            )

    def _compute_and_log_metrics(self, round_num: int):
        """Compute comprehensive per-class metrics across all clients."""
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

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
                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(output, dim=1)
                    all_y_true.append(y.cpu().numpy())
                    all_y_pred.append(preds.cpu().numpy())
                    all_y_prob.append(probs.cpu().numpy())

        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        all_y_prob = np.concatenate(all_y_prob)

        result = compute_comprehensive_metrics(
            all_y_true, all_y_pred, all_y_prob, self.num_classes_audit
        )

        # Log per-round metrics
        train_loss = (
            self.rs_train_loss[-1] if self.rs_train_loss else 0.0
        )
        self.audit_logger.log_round_metrics(
            round_num=round_num,
            accuracy=result["accuracy"],
            macro_f1=result["macro_f1"],
            weighted_f1=result["weighted_f1"],
            balanced_accuracy=result["balanced_accuracy"],
            train_loss=train_loss,
        )

        # Log per-class metrics
        self.audit_logger.log_per_class_metrics(
            round_num=round_num,
            per_class_precision=result["per_class_precision"],
            per_class_recall=result["per_class_recall"],
            per_class_f1=result["per_class_f1"],
            per_class_support=result["per_class_support"],
            class_names=self.label_map,
        )

        # Log confusion matrix
        class_names = [
            self.label_map.get(i, f"class_{i}")
            for i in range(self.num_classes_audit)
        ]
        self.audit_logger.save_confusion_matrix(
            result["confusion_matrix"], round_num, class_names
        )

        # Print key metrics
        print(f"  [Audit] Accuracy: {result['accuracy']:.4f}, "
              f"Macro-F1: {result['macro_f1']:.4f}, "
              f"Balanced Acc: {result['balanced_accuracy']:.4f}")

        # Print rare class metrics
        for cid in range(self.num_classes_audit):
            cname = self.label_map.get(cid, f"class_{cid}")
            if cname in self.rare_classes:
                rec = result["per_class_recall"].get(cid, 0.0)
                f1 = result["per_class_f1"].get(cid, 0.0)
                support = result["per_class_support"].get(cid, 0)
                if support > 0:
                    print(f"    Rare [{cname}]: Recall={rec:.4f}, F1={f1:.4f}, "
                          f"Support={support}")

    def _save_audit_metadata(self):
        """Save audit run metadata."""
        meta = {
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "M": self.M,
            "global_rounds": self.global_rounds,
            "local_epochs": self.local_epochs,
            "ablation_mode": self.ablation_mode,
            "entropy_lambda": self.entropy_lambda,
            "rare_coverage_mu": self.rare_coverage_mu,
            "join_ratio": self.join_ratio,
            "client_activity_rate": self.client_activity_rate,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes_audit,
            "best_accuracy": max(self.rs_test_acc) if self.rs_test_acc else None,
        }
        path = os.path.join(self.audit_output_dir, "logs", "audit_metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
