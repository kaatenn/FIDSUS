"""Diagnosis logger for FIDSUS affinity/top-n analysis.

Adds logging hooks without modifying FIDSUS training logic.
All data is detached/cpu before writing. Controlled by config flags:
  -- enable_affinity_diagnosis (default: false)
  -- enable_family_eval (default: false)
  -- diagnosis_output_dir (default: "audit/fidsus_real_training_diagnosis")
"""

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.data_utils import read_client_data_un


def _compute_label_entropy(label_counts):
    """Compute entropy and normalized entropy from label counts."""
    total = sum(label_counts)
    if total == 0:
        return 0.0, 0.0
    probs = [c / total for c in label_counts]
    entropy = -sum(p * np.log(max(p, 1e-12)) for p in probs)
    n_classes = len(label_counts)
    max_entropy = np.log(max(n_classes, 1))
    if max_entropy > 1e-12:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
    return entropy, normalized_entropy


def _compute_js_divergence(dist1, dist2):
    """Compute Jensen-Shannon divergence between two probability distributions."""
    m = [(p + q) / 2 for p, q in zip(dist1, dist2)]
    kl1 = sum(p * np.log(max(p, 1e-12) / max(mi, 1e-12)) for p, mi in zip(dist1, m))
    kl2 = sum(q * np.log(max(q, 1e-12) / max(mi, 1e-12)) for q, mi in zip(dist2, m))
    return (kl1 + kl2) / 2


def _compute_kl_divergence(dist1, dist2):
    """Compute KL divergence from dist1 to dist2."""
    return sum(p * np.log(max(p, 1e-12) / max(q, 1e-12)) for p, q in zip(dist1, dist2))


class DiagnosisLogger:
    """Handles all diagnostic logging for FIDSUS affinity/top-n analysis."""

    def __init__(self, args, num_clients, num_classes, dataset=None, seed=None):
        self.args = args
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.dataset = dataset if dataset else args.dataset
        self.method = args.algorithm
        self.seed = seed if seed is not None else getattr(args, 'prev', 0)
        self.output_dir = Path(getattr(args, 'diagnosis_output_dir',
                                       'audit/fidsus_real_training_diagnosis'))

        # Output paths
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.metrics_dir = self.output_dir / "metrics"
        for d in [self.raw_dir, self.processed_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # CSV file paths
        self.topn_log_path = self.raw_dir / "topn_selection_log.csv"
        self.affinity_log_path = self.raw_dir / "affinity_update_log.csv"
        self.pred_log_path = self.raw_dir / "prediction_log_fine.csv"
        self.label_profile_path = self.raw_dir / "client_label_profile.csv"

        # Row buffers
        self._topn_buffer = []
        self._affinity_buffer = []
        self._pred_buffer = []
        self._label_profiles = {}  # client_id -> dict

        # Header flags
        self._topn_header_written = False
        self._affinity_header_written = False
        self._pred_header_written = False
        self._profile_header_written = False

    # ── CLIENT LABEL PROFILE ──────────────────────────────────────────────

    def compute_client_label_profiles(self, dataset, seed, num_clients,
                                       family_mapping=None):
        """Scan all clients' train+test data to build label profiles."""
        self.dataset = dataset
        self.seed = seed
        self.num_clients = num_clients
        self.family_mapping = family_mapping

        all_labels = []
        global_label_counts = np.zeros(self.num_classes)

        for client_id in range(num_clients):
            train_data = read_client_data_un(dataset, client_id, is_train=True)
            test_data = read_client_data_un(dataset, client_id, is_train=False)
            combined = [(x.item() if torch.is_tensor(x) else int(x))
                        for _, labels in [train_data + test_data]
                        for x in (labels if isinstance(labels, torch.Tensor)
                                   else torch.tensor(labels))]
            # Fix: iterate properly
            combined = []
            for _, y in train_data + test_data:
                if isinstance(y, torch.Tensor):
                    for yi in y.flatten().tolist():
                        combined.append(int(yi))
                else:
                    combined.append(int(y))

            all_labels.append(combined)
            for label in combined:
                global_label_counts[label] += 1

        total_samples = sum(global_label_counts)
        global_dist = [global_label_counts[i] / max(total_samples, 1)
                       for i in range(self.num_classes)]

        # Family-level global distribution
        family_global_counts = defaultdict(int)
        if family_mapping and dataset in family_mapping:
            fm = family_mapping[dataset]
            for c in range(self.num_classes):
                family_global_counts[fm.get('mapping', {}).get(c, str(c))] += global_label_counts[c]
        family_global_total = sum(family_global_counts.values())
        family_global_dist = {f: family_global_counts[f] / max(family_global_total, 1)
                              for f in family_global_counts}

        for client_id in range(num_clients):
            self._compute_single_client_profile(client_id, all_labels[client_id],
                                                 global_dist, family_global_dist,
                                                 family_mapping)

        self._flush_label_profiles()
        return self._label_profiles

    def _compute_single_client_profile(self, client_id, labels, global_dist,
                                        family_global_dist, family_mapping):
        total = len(labels)
        label_counts = np.zeros(self.num_classes)
        for l in labels:
            if 0 <= l < self.num_classes:
                label_counts[l] += 1

        dominant_label = int(np.argmax(label_counts))
        dominant_ratio = label_counts[dominant_label] / max(total, 1)

        observed_count = int(np.sum(label_counts > 0))
        entropy, norm_entropy = _compute_label_entropy(label_counts)

        local_dist = [label_counts[i] / max(total, 1) for i in range(self.num_classes)]
        js_global = _compute_js_divergence(local_dist, global_dist)
        kl_global = _compute_kl_divergence(local_dist, global_dist)

        minority_count = int(np.sum((label_counts > 0) & (label_counts < total * 0.1)))
        minority_ratio = minority_count / max(self.num_classes, 1)

        # Family-level
        family_counts = defaultdict(int)
        dominant_family = "Unknown"
        dominant_family_ratio = 0.0
        observed_family_count = 0
        family_entropy = 0.0
        js_family = 0.0
        family_dist_json = "{}"

        if family_mapping and self.dataset in family_mapping:
            fm = family_mapping[self.dataset]
            label_to_family = fm.get('mapping', {})
            for c in range(self.num_classes):
                family = label_to_family.get(c, str(c))
                family_counts[family] += label_counts[c]

            if family_counts:
                dominant_family = max(family_counts, key=family_counts.get)
                dominant_family_ratio = family_counts[dominant_family] / max(total, 1)
                observed_family_count = int(sum(1 for v in family_counts.values() if v > 0))

                families_list = sorted(family_counts.keys())
                fam_counts_list = [family_counts[f] for f in families_list]
                family_entropy, _ = _compute_label_entropy(fam_counts_list)

                family_local_dist = [family_counts[f] / max(total, 1) for f in families_list]
                fam_global_dist_list = [family_global_dist.get(f, 0.0) for f in families_list]
                js_family = _compute_js_divergence(family_local_dist, fam_global_dist_list)

                family_dist_json = json.dumps(
                    {f: float(family_counts[f]) for f in families_list})

        profile = {
            'dataset': self.dataset,
            'seed': self.seed,
            'client_id': client_id,
            'total_samples': total,
            'label_distribution_json': json.dumps(
                {str(i): int(label_counts[i]) for i in range(self.num_classes)}),
            'dominant_label': dominant_label,
            'dominant_label_ratio': round(dominant_ratio, 6),
            'observed_class_count': observed_count,
            'label_entropy': round(entropy, 6),
            'normalized_label_entropy': round(norm_entropy, 6),
            'js_to_global_distribution': round(js_global, 8),
            'kl_to_global_distribution': round(kl_global, 8),
            'minority_class_count': minority_count,
            'minority_class_ratio': round(minority_ratio, 6),
            'family_distribution_json': family_dist_json,
            'dominant_family': dominant_family,
            'dominant_family_ratio': round(dominant_family_ratio, 6),
            'observed_family_count': observed_family_count,
            'family_entropy': round(family_entropy, 6),
            'js_to_global_family_distribution': round(js_family, 8),
        }
        self._label_profiles[client_id] = profile

    def get_label_profile(self, client_id):
        return self._label_profiles.get(client_id, {})

    def get_all_label_profiles(self):
        return dict(self._label_profiles)

    def _flush_label_profiles(self):
        if not self._label_profiles:
            return
        fieldnames = [
            'dataset', 'seed', 'client_id', 'total_samples',
            'label_distribution_json', 'dominant_label', 'dominant_label_ratio',
            'observed_class_count', 'label_entropy', 'normalized_label_entropy',
            'js_to_global_distribution', 'kl_to_global_distribution',
            'minority_class_count', 'minority_class_ratio',
            'family_distribution_json', 'dominant_family', 'dominant_family_ratio',
            'observed_family_count', 'family_entropy',
            'js_to_global_family_distribution',
        ]
        with open(self.label_profile_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cid in sorted(self._label_profiles.keys()):
                writer.writerow(self._label_profiles[cid])

    # ── TOP-N SELECTION LOG ────────────────────────────────────────────────

    def log_topn_selection(self, round_num, client_id, neighbor_ids,
                           affinity_scores, is_active, num_selected,
                           num_total, topn_size):
        for rank, (nid, score) in enumerate(zip(neighbor_ids, affinity_scores)):
            row = {
                'dataset': self.dataset,
                'method': self.method,
                'seed': self.seed,
                'round': round_num,
                'client_id': client_id,
                'neighbor_id': nid,
                'neighbor_rank': rank,
                'affinity_score_before_selection': round(float(score), 8),
                'affinity_score_after_update': '',  # filled later if available
                'is_selected_client_active': int(is_active),
                'selected_clients_this_round': num_selected,
                'num_total_clients': num_total,
                'topn_size': topn_size,
            }
            self._topn_buffer.append(row)

        if len(self._topn_buffer) >= 5000:
            self._flush_topn()

    def _flush_topn(self):
        if not self._topn_buffer:
            return
        fieldnames = [
            'dataset', 'method', 'seed', 'round', 'client_id', 'neighbor_id',
            'neighbor_rank', 'affinity_score_before_selection',
            'affinity_score_after_update', 'is_selected_client_active',
            'selected_clients_this_round', 'num_total_clients', 'topn_size',
        ]
        mode = 'a' if self._topn_header_written else 'w'
        with open(self.topn_log_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._topn_header_written:
                writer.writeheader()
                self._topn_header_written = True
            writer.writerows(self._topn_buffer)
        self._topn_buffer = []

    # ── AFFINITY UPDATE LOG ─────────────────────────────────────────────────

    def log_affinity_update(self, round_num, client_id, neighbor_id,
                            old_affinity, weight_delta, new_affinity,
                            L_old, L_received, param_distance,
                            computed_weight, was_clipped, normalized_weight=None):
        loss_improvement = L_old - L_received
        row = {
            'dataset': self.dataset,
            'method': self.method,
            'seed': self.seed,
            'round': round_num,
            'client_id': client_id,
            'neighbor_id': neighbor_id,
            'old_affinity': round(float(old_affinity), 8),
            'weight_delta': round(float(weight_delta), 8),
            'new_affinity': round(float(new_affinity), 8),
            'L_old': round(float(L_old), 8),
            'L_received': round(float(L_received), 8),
            'loss_improvement': round(float(loss_improvement), 8),
            'param_distance': round(float(param_distance), 8),
            'computed_weight': round(float(computed_weight), 8),
            'whether_weight_clipped_or_normalized': int(was_clipped),
            'normalized_weight': round(float(normalized_weight), 8) if normalized_weight is not None else '',
        }
        self._affinity_buffer.append(row)

        if len(self._affinity_buffer) >= 5000:
            self._flush_affinity()

    def _flush_affinity(self):
        if not self._affinity_buffer:
            return
        fieldnames = [
            'dataset', 'method', 'seed', 'round', 'client_id', 'neighbor_id',
            'old_affinity', 'weight_delta', 'new_affinity',
            'L_old', 'L_received', 'loss_improvement', 'param_distance',
            'computed_weight', 'whether_weight_clipped_or_normalized',
            'normalized_weight',
        ]
        mode = 'a' if self._affinity_header_written else 'w'
        with open(self.affinity_log_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._affinity_header_written:
                writer.writeheader()
                self._affinity_header_written = True
            writer.writerows(self._affinity_buffer)
        self._affinity_buffer = []

    # ── PREDICTION LOG ──────────────────────────────────────────────────────

    def log_prediction(self, round_num, client_id, y_true, y_pred,
                       y_true_name, y_pred_name, confidence=None,
                       sample_id=None):
        row = {
            'dataset': self.dataset,
            'method': self.method,
            'seed': self.seed,
            'round': round_num,
            'client_id': client_id,
            'sample_id': sample_id if sample_id is not None else '',
            'y_true_id': int(y_true),
            'y_true_name': y_true_name,
            'y_pred_id': int(y_pred),
            'y_pred_name': y_pred_name,
            'confidence': round(float(confidence), 6) if confidence is not None else '',
        }
        self._pred_buffer.append(row)

        if len(self._pred_buffer) >= 10000:
            self._flush_pred()

    def _flush_pred(self):
        if not self._pred_buffer:
            return
        fieldnames = [
            'dataset', 'method', 'seed', 'round', 'client_id', 'sample_id',
            'y_true_id', 'y_true_name', 'y_pred_id', 'y_pred_name', 'confidence',
        ]
        mode = 'a' if self._pred_header_written else 'w'
        with open(self.pred_log_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._pred_header_written:
                writer.writeheader()
                self._pred_header_written = True
            writer.writerows(self._pred_buffer)
        self._pred_buffer = []

    # ── FLUSH ALL ───────────────────────────────────────────────────────────

    def flush(self):
        self._flush_topn()
        self._flush_affinity()
        self._flush_pred()

    def __del__(self):
        self.flush()
