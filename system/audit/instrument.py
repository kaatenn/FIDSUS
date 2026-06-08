"""
Core audit instrumentation for FIDSUS Top-k mechanism.

Provides:
- Client label distribution statistics and logging
- Top-k selection logging (per round, per client)
- Candidate ranking logging
- Affinity matrix logging
- Per-class metrics (precision, recall, F1, confusion matrix)
- Utility functions: entropy, JS divergence, client classification
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

# ── Dataset class label maps ──────────────────────────────────────────────

UNSW_LABEL_MAP = {
    0: "Normal", 1: "Backdoor", 2: "Analysis", 3: "Fuzzers",
    4: "Shellcode", 5: "Reconnaissance", 6: "Exploits", 7: "DoS",
    8: "Worms", 9: "Generic",
}

NSLKDD_LABEL_MAP = {
    0: "DoS", 1: "Probe", 2: "U2R", 3: "R2L", 4: "Normal",
}

# Rare / critical attack classes per dataset
UNSW_RARE_CLASSES = {"Backdoor", "Analysis", "Shellcode", "Worms"}
NSLKDD_RARE_CLASSES = {"U2R", "R2L", "Probe"}


def get_label_map(dataset: str) -> dict[int, str]:
    if dataset.upper() in ("UNSW", "UNSW-NB15"):
        return UNSW_LABEL_MAP
    elif dataset.upper() in ("NSLKDD", "KDD"):
        return NSLKDD_LABEL_MAP
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_rare_classes(dataset: str) -> set[str]:
    if dataset.upper() in ("UNSW", "UNSW-NB15"):
        return UNSW_RARE_CLASSES
    elif dataset.upper() in ("NSLKDD", "KDD"):
        return NSLKDD_RARE_CLASSES
    else:
        return set()


def get_num_classes(dataset: str) -> int:
    if dataset.upper() in ("UNSW", "UNSW-NB15"):
        return 10
    elif dataset.upper() in ("NSLKDD", "KDD"):
        return 5
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Information-theoretic utilities ───────────────────────────────────────

def compute_label_entropy(class_counts: np.ndarray) -> float:
    """Compute Shannon entropy H = -sum_c p_c log(p_c)."""
    total = class_counts.sum()
    if total == 0:
        return 0.0
    probs = class_counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_norm_entropy(class_counts: np.ndarray, num_classes: int) -> float:
    """Normalized entropy H / log(C)."""
    H = compute_label_entropy(class_counts)
    logC = np.log(num_classes)
    return float(H / logC) if logC > 0 else 0.0


def compute_js_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence between two label distributions."""
    p_total = p_counts.sum()
    q_total = q_counts.sum()
    if p_total == 0 or q_total == 0:
        return 1.0
    p = p_counts.astype(np.float64) / p_total
    q = q_counts.astype(np.float64) / q_total
    # Zero-pad to same length
    max_len = max(len(p), len(q))
    if len(p) < max_len:
        p = np.pad(p, (0, max_len - len(p)), constant_values=1e-12)
    if len(q) < max_len:
        q = np.pad(q, (0, max_len - len(q)), constant_values=1e-12)
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def compute_client_label_stats(
    all_labels: list[np.ndarray],
    num_clients: int,
    dataset: str,
) -> list[dict[str, Any]]:
    """Compute per-client label distribution statistics.

    Args:
        all_labels: list of label arrays, one per client
        num_clients: total number of clients
        dataset: dataset name ("UNSW" or "NSLKDD")

    Returns:
        list of dicts, one per client, with keys:
            client_id, total_samples, class_counts, class_distribution,
            label_entropy, normalized_label_entropy, dominant_class,
            dominant_class_ratio, rare_attack_classes_present,
            rare_attack_sample_count, rare_attack_ratio
    """
    label_map = get_label_map(dataset)
    rare_classes = get_rare_classes(dataset)
    num_classes = get_num_classes(dataset)
    stats_list = []

    for cid in range(num_clients):
        y = all_labels[cid]
        total = len(y)
        class_counts = np.zeros(num_classes, dtype=np.int64)
        unique, cnts = np.unique(y, return_counts=True)
        for u, cnt in zip(unique, cnts):
            if 0 <= int(u) < num_classes:
                class_counts[int(u)] = cnt

        entropy = compute_label_entropy(class_counts)
        norm_entropy = compute_norm_entropy(class_counts, num_classes)

        dominant_idx = int(np.argmax(class_counts))
        dominant_class = label_map.get(dominant_idx, f"class_{dominant_idx}")
        dominant_ratio = float(class_counts[dominant_idx] / total) if total > 0 else 0.0

        rare_present = []
        rare_count = 0
        for c_idx, cnt in enumerate(class_counts):
            c_name = label_map.get(c_idx, f"class_{c_idx}")
            if c_name in rare_classes and cnt > 0:
                rare_present.append(c_name)
                rare_count += cnt

        rare_ratio = float(rare_count / total) if total > 0 else 0.0

        class_distribution = {
            label_map.get(i, f"class_{i}"): int(c)
            for i, c in enumerate(class_counts)
        }

        stats_list.append({
            "client_id": cid,
            "total_samples": total,
            "class_counts": class_counts.tolist(),
            "class_distribution": class_distribution,
            "label_entropy": entropy,
            "normalized_label_entropy": norm_entropy,
            "dominant_class": dominant_class,
            "dominant_class_ratio": dominant_ratio,
            "rare_attack_classes_present": rare_present,
            "rare_attack_sample_count": rare_count,
            "rare_attack_ratio": rare_ratio,
        })

    return stats_list


def classify_client_type(
    stats: dict,
    entropy_threshold_balanced: float = 0.8,
    entropy_threshold_imbalanced: float = 0.3,
) -> str:
    """Classify a client as balanced, severely_imbalanced, or middle.

    Also returns flags:
      - is_rare_attack: contains any rare attack class samples
      - is_normal_heavy: dominant class is Normal with ratio >= 0.7
      - is_majority_heavy: dominant class ratio >= 0.7 (any class)
    """
    norm_ent = stats["normalized_label_entropy"]
    if norm_ent >= entropy_threshold_balanced:
        return "balanced"
    elif norm_ent <= entropy_threshold_imbalanced:
        return "severely_imbalanced"
    else:
        return "middle"


def classify_client_tags(stats: dict, dataset: str) -> dict[str, bool]:
    """Get boolean tags for a client."""
    rare_classes = get_rare_classes(dataset)
    return {
        "balanced": stats["normalized_label_entropy"] >= 0.8,
        "severely_imbalanced": stats["normalized_label_entropy"] <= 0.3,
        "rare_attack": stats["rare_attack_sample_count"] > 0,
        "normal_heavy": (
            stats["dominant_class"] == "Normal"
            and stats["dominant_class_ratio"] >= 0.7
        ),
        "majority_heavy": stats["dominant_class_ratio"] >= 0.7,
    }


# ── AuditLogger ───────────────────────────────────────────────────────────

class AuditLogger:
    """Manages all audit log files and writes.

    Usage:
        logger = AuditLogger(output_dir="docs/audit/topk/logs")
        logger.save_client_distribution(stats)
        logger.log_topk_selection(round_num, target_id, active_clients, ...)
    """

    def __init__(self, output_dir: str = "docs/audit/topk/logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = self.output_dir.parent / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # CSV writers
        self._topk_writer = None
        self._topk_file = None
        self._candidate_writer = None
        self._candidate_file = None
        self._metrics_writer = None
        self._metrics_file = None
        self._perclass_writer = None
        self._perclass_file = None

    # ── 1. Client distribution ────────────────────────────────────────

    def save_client_distribution(self, stats: list[dict], dataset: str):
        csv_path = self.output_dir / "client_label_distribution.csv"
        json_path = self.output_dir / "client_label_distribution.json"

        # CSV
        fieldnames = [
            "client_id", "total_samples", "label_entropy",
            "normalized_label_entropy", "dominant_class",
            "dominant_class_ratio", "rare_attack_classes_present",
            "rare_attack_sample_count", "rare_attack_ratio",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in stats:
                row = {k: s[k] for k in fieldnames}
                row["rare_attack_classes_present"] = ",".join(
                    s["rare_attack_classes_present"]
                )
                writer.writerow(row)

        # JSON (full)
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"[Audit] Client distribution saved to {csv_path} and {json_path}")

    # ── 2. Top-k selection log ────────────────────────────────────────

    def open_topk_log(self):
        path = self.output_dir / "topk_selection_log.csv"
        self._topk_file = open(path, "w", newline="")
        fieldnames = [
            "round", "target_client_id", "active_clients",
            "selected_topk_neighbors", "affinity_scores_of_selected_neighbors",
            "target_client_entropy", "target_client_dominant_class",
            "target_client_dominant_class_ratio",
            "neighbor_client_id", "neighbor_entropy",
            "neighbor_dominant_class", "neighbor_dominant_class_ratio",
            "neighbor_rare_attack_sample_count",
            "entropy_gap_between_target_and_neighbor",
            "label_distribution_js_divergence_between_target_and_neighbor",
            "whether_neighbor_is_balanced",
            "whether_neighbor_is_severely_imbalanced",
            "whether_neighbor_contains_rare_attack",
        ]
        self._topk_writer = csv.DictWriter(self._topk_file, fieldnames=fieldnames)
        self._topk_writer.writeheader()

    def log_topk_selection(
        self,
        round_num: int,
        target_client_id: int,
        active_clients: list[int],
        selected_topk_neighbors: list[int],
        affinity_scores: list[float],
        target_stats: dict,
        neighbor_stats_list: list[dict],
        dataset: str,
    ):
        if self._topk_writer is None:
            self.open_topk_log()

        for nid, ascore, nstats in zip(
            selected_topk_neighbors, affinity_scores, neighbor_stats_list
        ):
            t_ent = target_stats["label_entropy"]
            n_ent = nstats["label_entropy"]
            t_class_counts = np.array(target_stats["class_counts"])
            n_class_counts = np.array(nstats["class_counts"])
            js_div = compute_js_divergence(t_class_counts, n_class_counts)
            tags = classify_client_tags(nstats, dataset)

            row = {
                "round": round_num,
                "target_client_id": target_client_id,
                "active_clients": ",".join(map(str, active_clients)),
                "selected_topk_neighbors": nid,
                "affinity_scores_of_selected_neighbors": f"{ascore:.6f}",
                "target_client_entropy": f"{t_ent:.4f}",
                "target_client_dominant_class": target_stats["dominant_class"],
                "target_client_dominant_class_ratio": f"{target_stats['dominant_class_ratio']:.4f}",
                "neighbor_client_id": nid,
                "neighbor_entropy": f"{n_ent:.4f}",
                "neighbor_dominant_class": nstats["dominant_class"],
                "neighbor_dominant_class_ratio": f"{nstats['dominant_class_ratio']:.4f}",
                "neighbor_rare_attack_sample_count": nstats["rare_attack_sample_count"],
                "entropy_gap_between_target_and_neighbor": f"{t_ent - n_ent:.4f}",
                "label_distribution_js_divergence_between_target_and_neighbor": f"{js_div:.6f}",
                "whether_neighbor_is_balanced": tags["balanced"],
                "whether_neighbor_is_severely_imbalanced": tags["severely_imbalanced"],
                "whether_neighbor_contains_rare_attack": tags["rare_attack"],
            }
            self._topk_writer.writerow(row)

    def close_topk_log(self):
        if self._topk_file:
            self._topk_file.close()
            self._topk_writer = None
            self._topk_file = None

    # ── 3. Candidate ranking log ──────────────────────────────────────

    def open_candidate_log(self):
        path = self.output_dir / "topk_candidate_ranking.csv"
        self._candidate_file = open(path, "w", newline="")
        fieldnames = [
            "round", "target_client_id", "candidate_client_id",
            "affinity_score", "candidate_rank", "selected_or_not",
            "candidate_entropy", "candidate_dominant_class",
            "candidate_rare_attack_sample_count",
            "entropy_gap", "js_divergence",
        ]
        self._candidate_writer = csv.DictWriter(
            self._candidate_file, fieldnames=fieldnames
        )
        self._candidate_writer.writeheader()

    def log_candidate_ranking(
        self,
        round_num: int,
        target_client_id: int,
        candidate_affinities: list[tuple[int, float]],
        topk_indices: set[int],
        client_stats_map: dict[int, dict],
    ):
        """Log candidate ranking for a single target client.

        Args:
            candidate_affinities: list of (client_id, affinity_score) for all clients
            topk_indices: set of client_ids that were selected as top-k
            client_stats_map: dict mapping client_id -> stats dict
        """
        if self._candidate_writer is None:
            self.open_candidate_log()

        t_stats = client_stats_map[target_client_id]
        t_class_counts = np.array(t_stats["class_counts"])

        sorted_candidates = sorted(
            candidate_affinities, key=lambda x: x[1], reverse=True
        )
        for rank, (cid, ascore) in enumerate(sorted_candidates, 1):
            if cid == target_client_id:
                continue  # skip self

            selected = cid in topk_indices
            c_stats = client_stats_map.get(cid)
            if c_stats is None:
                continue

            c_ent = c_stats["label_entropy"]
            c_class_counts = np.array(c_stats["class_counts"])
            js_div = compute_js_divergence(t_class_counts, c_class_counts)

            row = {
                "round": round_num,
                "target_client_id": target_client_id,
                "candidate_client_id": cid,
                "affinity_score": f"{ascore:.6f}",
                "candidate_rank": rank,
                "selected_or_not": selected,
                "candidate_entropy": f"{c_ent:.4f}",
                "candidate_dominant_class": c_stats["dominant_class"],
                "candidate_rare_attack_sample_count": c_stats["rare_attack_sample_count"],
                "entropy_gap": f"{t_stats['label_entropy'] - c_ent:.4f}",
                "js_divergence": f"{js_div:.6f}",
            }
            self._candidate_writer.writerow(row)

    def close_candidate_log(self):
        if self._candidate_file:
            self._candidate_file.close()
            self._candidate_writer = None
            self._candidate_file = None

    # ── 4. Affinity matrix logging ────────────────────────────────────

    def save_affinity_matrix(
        self, P: torch.Tensor, round_num: int, client_stats: list[dict]
    ):
        """Save affinity matrix as .npy and summary CSV."""
        P_np = P.detach().cpu().numpy()
        npy_path = self.output_dir / f"affinity_matrix_round_{round_num}.npy"
        np.save(npy_path, P_np)

        # Summary CSV
        num_clients = P_np.shape[0]
        summary_rows = []
        for cid in range(num_clients):
            row_affinity = P_np[cid]
            in_degree = int(np.sum(
                np.argpartition(-P_np[:, cid], 5)[:5]  # clients who rank cid in their top5
            ))
            # Actually compute: cid appears in how many other clients' top-k
            in_degree_count = 0
            for other in range(num_clients):
                if other == cid:
                    continue
                top5 = np.argpartition(-P_np[other], min(5, num_clients - 1))[:5]
                if cid in top5:
                    in_degree_count += 1

            mean_affinity = float(np.mean(row_affinity))
            tags = classify_client_tags(client_stats[cid], "UNSW")

            summary_rows.append({
                "client_id": cid,
                "in_degree_top5": in_degree_count,
                "mean_affinity_score": f"{mean_affinity:.6f}",
                "balanced": tags["balanced"],
                "severely_imbalanced": tags["severely_imbalanced"],
                "rare_attack": tags["rare_attack"],
                "normal_heavy": tags["normal_heavy"],
                "majority_heavy": tags["majority_heavy"],
                "dominant_class": client_stats[cid]["dominant_class"],
                "normalized_entropy": f"{client_stats[cid]['normalized_label_entropy']:.4f}",
            })

        csv_path = self.output_dir / f"affinity_matrix_summary_round_{round_num}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

        # Also write an aggregated summary
        self._save_affinity_summary_aggregate(summary_rows, round_num)

    def _save_affinity_summary_aggregate(
        self, rows: list[dict], round_num: int
    ):
        """Compute and save aggregate in-degree stats by client type."""
        groups = defaultdict(list)
        for r in rows:
            if r["balanced"] == "True":
                groups["balanced"].append(r)
            if r["severely_imbalanced"] == "True":
                groups["severely_imbalanced"].append(r)
            if r["rare_attack"] == "True":
                groups["rare_attack"].append(r)
            if r["normal_heavy"] == "True":
                groups["normal_heavy"].append(r)
            if r["majority_heavy"] == "True":
                groups["majority_heavy"].append(r)

        agg = {}
        for gname, group_rows in groups.items():
            agg[gname] = {
                "count": len(group_rows),
                "mean_in_degree": float(np.mean([r["in_degree_top5"] for r in group_rows])),
                "mean_affinity": float(np.mean([
                    float(r["mean_affinity_score"]) for r in group_rows
                ])),
            }

        agg_path = (
            self.output_dir / f"affinity_matrix_aggregate_round_{round_num}.json"
        )
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)

    # ── 5. Per-class metrics logging ──────────────────────────────────

    def open_metrics_logs(self):
        """Open per-round and per-class metrics CSV files."""
        path1 = self.output_dir / "per_round_metrics.csv"
        self._metrics_file = open(path1, "w", newline="")
        self._metrics_writer = csv.DictWriter(
            self._metrics_file,
            fieldnames=[
                "round", "accuracy", "macro_f1", "weighted_f1",
                "balanced_accuracy", "train_loss",
            ],
        )
        self._metrics_writer.writeheader()

        path2 = self.output_dir / "per_class_metrics.csv"
        self._perclass_file = open(path2, "w", newline="")
        self._perclass_writer = csv.DictWriter(
            self._perclass_file,
            fieldnames=[
                "round", "class_name", "class_id",
                "precision", "recall", "f1", "support",
            ],
        )
        self._perclass_writer.writeheader()

    def log_round_metrics(
        self,
        round_num: int,
        accuracy: float,
        macro_f1: float,
        weighted_f1: float,
        balanced_accuracy: float,
        train_loss: float,
    ):
        if self._metrics_writer is None:
            self.open_metrics_logs()
        self._metrics_writer.writerow({
            "round": round_num,
            "accuracy": f"{accuracy:.6f}",
            "macro_f1": f"{macro_f1:.6f}",
            "weighted_f1": f"{weighted_f1:.6f}",
            "balanced_accuracy": f"{balanced_accuracy:.6f}",
            "train_loss": f"{train_loss:.6f}",
        })

    def log_per_class_metrics(
        self,
        round_num: int,
        per_class_precision: dict[int, float],
        per_class_recall: dict[int, float],
        per_class_f1: dict[int, float],
        per_class_support: dict[int, int],
        class_names: dict[int, str],
    ):
        if self._perclass_writer is None:
            self.open_metrics_logs()
        for cid in sorted(per_class_precision.keys()):
            self._perclass_writer.writerow({
                "round": round_num,
                "class_name": class_names.get(cid, f"class_{cid}"),
                "class_id": cid,
                "precision": f"{per_class_precision[cid]:.6f}",
                "recall": f"{per_class_recall[cid]:.6f}",
                "f1": f"{per_class_f1[cid]:.6f}",
                "support": per_class_support.get(cid, 0),
            })

    def save_confusion_matrix(
        self, cm: np.ndarray, round_num: int, class_names: list[str]
    ):
        path = self.output_dir / f"confusion_matrix_round_{round_num}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + class_names)
            for i, name in enumerate(class_names):
                writer.writerow([name] + cm[i].tolist())

    def close_metrics_logs(self):
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_writer = None
            self._metrics_file = None
        if self._perclass_file:
            self._perclass_file.close()
            self._perclass_writer = None
            self._perclass_file = None

    def close_all(self):
        self.close_topk_log()
        self.close_candidate_log()
        self.close_metrics_logs()


# ── Comprehensive metrics computation ─────────────────────────────────────

def compute_comprehensive_metrics(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    all_y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, weighted-F1, balanced accuracy,
    per-class precision/recall/F1, and confusion matrix.

    Args:
        all_y_true: ground truth labels, shape (N,)
        all_y_pred: predicted labels, shape (N,)
        all_y_prob: predicted probabilities, shape (N, C)
        num_classes: number of classes

    Returns:
        dict with keys: accuracy, macro_f1, weighted_f1, balanced_accuracy,
            per_class_precision, per_class_recall, per_class_f1, per_class_support,
            confusion_matrix
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    accuracy = float(accuracy_score(all_y_true, all_y_pred))
    macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(all_y_true, all_y_pred))

    prec, rec, f1, support = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=list(range(num_classes)), zero_division=0
    )

    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(num_classes)))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_precision": {i: float(prec[i]) for i in range(num_classes)},
        "per_class_recall": {i: float(rec[i]) for i in range(num_classes)},
        "per_class_f1": {i: float(f1[i]) for i in range(num_classes)},
        "per_class_support": {i: int(support[i]) for i in range(num_classes)},
        "confusion_matrix": cm,
    }


# ── Data loading helpers ──────────────────────────────────────────────────

def load_all_client_labels(
    dataset: str, num_clients: int, root_dir: str = "dataset"
) -> list[np.ndarray]:
    """Load all client labels from pre-partitioned dataset files.

    Returns:
        list of label arrays, one per client (train + test concatenated)
    """
    root = Path(root_dir) / dataset
    all_labels = []

    for cid in range(num_clients):
        train_path = root / "train" / f"{cid}.npz"
        test_path = root / "test" / f"{cid}.npz"

        labels = []
        for path in [train_path, test_path]:
            if path.exists():
                with open(path, "rb") as f:
                    data = np.load(f, allow_pickle=True)["data"].tolist()
                    labels.extend(data["y"])

        all_labels.append(np.array(labels, dtype=np.int64))

    return all_labels
