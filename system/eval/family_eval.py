"""
Family-level evaluation module for FIDSUS.
==========================================
Evaluates whether the model implicitly learns attack-family-level knowledge
by remapping fine-grained predicted labels to coarse family labels and
recomputing all metrics at the family level.

All label-to-family mappings are configurable; nothing is hardcoded.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Label definitions (id -> name) — verified against actual .npz files.
#
# NSLKDD: raw CSV labels are {1,2,3,4,5} -> minus-1 in generator -> {0,1,2,3,4}
#          mapping: 1=DoS, 2=Probe, 3=U2R, 4=R2L, 5=Normal
# UNSW:    raw CSV labels are {1..10} -> minus-1 in generator -> {0..9}
#          mapping: 1=Normal, 2=Backdoor, 3=Analysis, 4=Fuzzers,
#                   5=Shellcode, 6=Reconnaissance, 7=Exploits, 8=DoS,
#                   9=Worms, 10=Generic
# ═══════════════════════════════════════════════════════════════════════════════

NSLKDD_LABEL_MAP: Dict[int, str] = {
    0: "DoS",       # raw=1 -> -1 -> 0
    1: "Probe",     # raw=2 -> -1 -> 1
    2: "U2R",       # raw=3 -> -1 -> 2
    3: "R2L",       # raw=4 -> -1 -> 3
    4: "Normal",    # raw=5 -> -1 -> 4
}

UNSW_LABEL_MAP: Dict[int, str] = {
    0: "Normal",          # raw=1 -> -1 -> 0
    1: "Backdoor",        # raw=2 -> -1 -> 1
    2: "Analysis",        # raw=3 -> -1 -> 2
    3: "Fuzzers",         # raw=4 -> -1 -> 3
    4: "Shellcode",       # raw=5 -> -1 -> 4
    5: "Reconnaissance",  # raw=6 -> -1 -> 5
    6: "Exploits",        # raw=7 -> -1 -> 6
    7: "DoS",             # raw=8 -> -1 -> 7
    8: "Worms",           # raw=9 -> -1 -> 8
    9: "Generic",         # raw=10 -> -1 -> 9
}

# UAV-NIDD uses a 12-class label set; label encoding is determined by
# the ordinal encoder in the generator script.
UAV_NIDD_LABEL_MAP: Dict[int, str] = {
    0: "Normal",
    1: "DoS",
    2: "DDoS",
    3: "Scanning",
    4: "Reconnaissance",
    5: "MITM",
    6: "Evil Twin",
    7: "De-authentication",
    8: "Fake Landing",
    9: "Brute Force",
    10: "GPS Jamming",
    11: "Replay",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Label -> family mappings (configurable per dataset)
# Names in family maps MUST exactly match values in LABEL_MAP.
# ═══════════════════════════════════════════════════════════════════════════════

NSLKDD_FAMILY_MAP: Dict[str, str] = {
    "Normal": "Normal",
    "DoS": "DoS",
    "Probe": "Probe",
    "U2R": "Privilege/Access",
    "R2L": "Privilege/Access",
}

UNSW_FAMILY_MAP: Dict[str, str] = {
    "Normal": "Normal",
    "Generic": "Generic",
    "Exploits": "Exploits",
    "DoS": "DoS",
    "Reconnaissance": "Reconnaissance",
    "Fuzzers": "Fuzzing",
    "Analysis": "Backdoor/Analysis",
    "Backdoor": "Backdoor/Analysis",
    "Shellcode": "Shellcode/Worms",
    "Worms": "Shellcode/Worms",
}

UAV_NIDD_FAMILY_MAP: Dict[str, str] = {
    "Normal": "Normal",
    "DoS": "Flooding/Availability",
    "DDoS": "Flooding/Availability",
    "Scanning": "Recon/Scanning",
    "Reconnaissance": "Recon/Scanning",
    "MITM": "Link/Control Manipulation",
    "Evil Twin": "Link/Control Manipulation",
    "De-authentication": "Link/Control Manipulation",
    "Fake Landing": "Link/Control Manipulation",
    "Brute Force": "Authentication",
    "GPS Jamming": "GPS/Signal Interference",
    "Replay": "Replay",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Dataset registry
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "NSLKDD": {
        "label_map": NSLKDD_LABEL_MAP,
        "family_map": NSLKDD_FAMILY_MAP,
    },
    "UNSW": {
        "label_map": UNSW_LABEL_MAP,
        "family_map": UNSW_FAMILY_MAP,
    },
    "UAV-NIDD": {
        "label_map": UAV_NIDD_LABEL_MAP,
        "family_map": UAV_NIDD_FAMILY_MAP,
    },
}


@dataclass
class EvalResult:
    """Container for all evaluation metrics at one granularity level."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion: np.ndarray
    class_names: List[str]


@dataclass
class IntraFamilyConfusion:
    """A confusion pair within or across families."""
    family: str          # true family
    true_class: str      # fine-grained true label name
    pred_class: str      # fine-grained predicted label name
    count: int


@dataclass
class FullEvalReport:
    """Complete evaluation report for a single dataset/method run."""
    dataset: str
    algorithm: str
    goal: str
    run_id: int
    fine_grained: EvalResult
    family_level: EvalResult
    family_fine_accuracy_gap: float
    family_fine_macro_f1_gap: float
    intra_family_pairs: List[IntraFamilyConfusion] = field(default_factory=list)
    top_overall_pairs: List[IntraFamilyConfusion] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Core mapping logic
# ═══════════════════════════════════════════════════════════════════════════════

def label_id_to_name(label_ids: np.ndarray, label_map: Dict[int, str]) -> np.ndarray:
    """Convert label ids (int) to label names (str). Unknown ids get a fallback name."""
    names = []
    for lid in label_ids:
        names.append(label_map.get(int(lid), f"UNKNOWN_{int(lid)}"))
    return np.array(names)


def label_name_to_family(
    label_names: np.ndarray, family_map: Dict[str, str]
) -> np.ndarray:
    """Map fine-grained label names to family names."""
    families = []
    for name in label_names:
        families.append(family_map.get(name, "UNKNOWN_FAMILY"))
    return np.array(families)


# ═══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eval_result(
    y_true: np.ndarray, y_pred: np.ndarray, class_labels: Sequence[str]
) -> EvalResult:
    """Compute all metrics at a given granularity (fine or family).

    Only classes that appear in y_true or y_pred are included, so missing
    classes never cause errors.
    """
    present_labels = sorted(set(y_true) | set(y_pred))
    ordered_labels = [c for c in class_labels if c in present_labels]
    if not ordered_labels:
        ordered_labels = sorted(present_labels)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=ordered_labels, zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=ordered_labels, zero_division=0))

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ordered_labels, zero_division=0
    )
    per_class_precision = {ordered_labels[i]: float(precision[i]) for i in range(len(ordered_labels))}
    per_class_recall = {ordered_labels[i]: float(recall[i]) for i in range(len(ordered_labels))}

    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)

    return EvalResult(
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        confusion=cm,
        class_names=ordered_labels,
    )


def compute_intra_family_analysis(
    y_true_fine_names: np.ndarray,
    y_pred_fine_names: np.ndarray,
    family_map: Dict[str, str],
    top_k: int = 20,
) -> Tuple[List[IntraFamilyConfusion], List[IntraFamilyConfusion]]:
    """Identify confusion pairs.

    Returns (intra_family_pairs, top_overall_pairs) sorted by count descending.
    Intra-family pairs are cases where predicted family == true family but
    fine-grained class is wrong — the model knows which family the attack
    belongs to but confuses specific attacks within that family.
    """
    assert len(y_true_fine_names) == len(y_pred_fine_names)

    true_families = label_name_to_family(y_true_fine_names, family_map)
    pred_families = label_name_to_family(y_pred_fine_names, family_map)

    intra_confusions: Dict[Tuple[str, str, str], int] = defaultdict(int)
    overall_confusions: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for tf, pf, tn, pn in zip(true_families, pred_families, y_true_fine_names, y_pred_fine_names):
        if tn == pn:
            continue
        pair_key = (tf, str(tn), str(pn))
        overall_confusions[pair_key] += 1
        if tf == pf:
            intra_confusions[pair_key] += 1

    intra_sorted = sorted(intra_confusions.items(), key=lambda x: x[1], reverse=True)
    overall_sorted = sorted(overall_confusions.items(), key=lambda x: x[1], reverse=True)

    intra_pairs = [
        IntraFamilyConfusion(family=fam, true_class=tn, pred_class=pn, count=cnt)
        for (fam, tn, pn), cnt in intra_sorted[:top_k]
    ]
    overall_pairs = [
        IntraFamilyConfusion(family=fam, true_class=tn, pred_class=pn, count=cnt)
        for (fam, tn, pn), cnt in overall_sorted[:top_k]
    ]

    return intra_pairs, overall_pairs


def run_full_evaluation(
    y_true_ids: np.ndarray,
    y_pred_ids: np.ndarray,
    dataset: str,
    algorithm: str = "Unknown",
    goal: str = "test",
    run_id: int = 0,
    top_k_intra: int = 20,
) -> FullEvalReport:
    """Run both fine-grained and family-level evaluation on raw predictions.

    Parameters
    ----------
    y_true_ids : np.ndarray
        Ground truth label ids (integers, as stored in .npz files).
    y_pred_ids : np.ndarray
        Predicted label ids (integers).
    dataset : str
        One of 'NSLKDD', 'UNSW', 'UAV-NIDD'. Must exist in DATASET_CONFIGS.
    algorithm : str
        Algorithm name for record-keeping.
    goal : str
        Experiment goal label.
    run_id : int
        Run index (for multiple repeats).
    top_k_intra : int
        How many top intra-family confusion pairs to return.

    Returns
    -------
    FullEvalReport
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(DATASET_CONFIGS)}"
        )

    cfg = DATASET_CONFIGS[dataset]
    label_map = cfg["label_map"]
    family_map = cfg["family_map"]

    # Step 1: Convert ids -> names
    y_true_names = label_id_to_name(y_true_ids, label_map)
    y_pred_names = label_id_to_name(y_pred_ids, label_map)

    # Step 2: Fine-grained evaluation
    fine_class_names = sorted(set(family_map.keys()))
    fine_eval = compute_eval_result(y_true_names, y_pred_names, fine_class_names)

    # Step 3: Map to families
    y_true_families = label_name_to_family(y_true_names, family_map)
    y_pred_families = label_name_to_family(y_pred_names, family_map)

    # Step 4: Family-level evaluation
    family_class_names = sorted(set(family_map.values()))
    family_eval = compute_eval_result(y_true_families, y_pred_families, family_class_names)

    # Step 5: Family-fine gap
    acc_gap = family_eval.accuracy - fine_eval.accuracy
    mf1_gap = family_eval.macro_f1 - fine_eval.macro_f1

    # Step 6: Intra-family confusion analysis
    intra_pairs, top_pairs = compute_intra_family_analysis(
        y_true_names, y_pred_names, family_map, top_k=top_k_intra
    )

    return FullEvalReport(
        dataset=dataset,
        algorithm=algorithm,
        goal=goal,
        run_id=run_id,
        fine_grained=fine_eval,
        family_level=family_eval,
        family_fine_accuracy_gap=acc_gap,
        family_fine_macro_f1_gap=mf1_gap,
        intra_family_pairs=intra_pairs,
        top_overall_pairs=top_pairs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization / persistence
# ═══════════════════════════════════════════════════════════════════════════════

def _eval_result_to_dict(r: EvalResult, prefix: str = "") -> Dict[str, Any]:
    return {
        f"{prefix}accuracy": r.accuracy,
        f"{prefix}macro_f1": r.macro_f1,
        f"{prefix}weighted_f1": r.weighted_f1,
        f"{prefix}per_class_precision": r.per_class_precision,
        f"{prefix}per_class_recall": r.per_class_recall,
    }


def _confusion_matrix_to_dict(cm: np.ndarray, class_names: List[str]) -> List[Dict]:
    rows = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if cm[i, j] > 0:
                rows.append({
                    "true": true_name,
                    "predicted": pred_name,
                    "count": int(cm[i, j]),
                })
    return rows


def save_report(
    report: FullEvalReport,
    output_dir: str,
    fmt: str = "json",
) -> Dict[str, str]:
    """Save all evaluation artifacts to disk.

    Produces:
      - fine_grained_metrics.json
      - fine_grained_confusion_matrix.json
      - family_level_metrics.json
      - family_level_confusion_matrix.json
      - family_fine_gap.json
      - intra_family_confusion_pairs.json
      - top_confusion_pairs.json

    Returns a dict mapping artifact name -> file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    # Fine-grained metrics
    fine_metrics = _eval_result_to_dict(report.fine_grained)
    if fmt == "json":
        path = os.path.join(output_dir, "fine_grained_metrics.json")
        with open(path, "w") as f:
            json.dump(fine_metrics, f, indent=2)
        saved["fine_grained_metrics"] = path

    # Fine-grained confusion matrix
    fine_cm = _confusion_matrix_to_dict(report.fine_grained.confusion, report.fine_grained.class_names)
    if fmt == "json":
        path = os.path.join(output_dir, "fine_grained_confusion_matrix.json")
        with open(path, "w") as f:
            json.dump(fine_cm, f, indent=2)
        saved["fine_grained_confusion_matrix"] = path

    # Family-level metrics
    family_metrics = _eval_result_to_dict(report.family_level)
    if fmt == "json":
        path = os.path.join(output_dir, "family_level_metrics.json")
        with open(path, "w") as f:
            json.dump(family_metrics, f, indent=2)
        saved["family_level_metrics"] = path

    # Family-level confusion matrix
    family_cm = _confusion_matrix_to_dict(report.family_level.confusion, report.family_level.class_names)
    if fmt == "json":
        path = os.path.join(output_dir, "family_level_confusion_matrix.json")
        with open(path, "w") as f:
            json.dump(family_cm, f, indent=2)
        saved["family_level_confusion_matrix"] = path

    # Family-fine gap
    gap_data = {
        "family_level_accuracy": report.family_level.accuracy,
        "fine_grained_accuracy": report.fine_grained.accuracy,
        "family_fine_accuracy_gap": report.family_fine_accuracy_gap,
        "family_level_macro_f1": report.family_level.macro_f1,
        "fine_grained_macro_f1": report.fine_grained.macro_f1,
        "family_fine_macro_f1_gap": report.family_fine_macro_f1_gap,
    }
    if fmt == "json":
        path = os.path.join(output_dir, "family_fine_gap.json")
        with open(path, "w") as f:
            json.dump(gap_data, f, indent=2)
        saved["family_fine_gap"] = path

    # Intra-family confusion pairs
    intra_data = [
        {"family": p.family, "true_class": p.true_class,
         "pred_class": p.pred_class, "count": p.count}
        for p in report.intra_family_pairs
    ]
    if fmt == "json":
        path = os.path.join(output_dir, "intra_family_confusion_pairs.json")
        with open(path, "w") as f:
            json.dump(intra_data, f, indent=2)
        saved["intra_family_confusion_pairs"] = path

    # Top overall confusion pairs
    overall_data = [
        {"family": p.family, "true_class": p.true_class,
         "pred_class": p.pred_class, "count": p.count}
        for p in report.top_overall_pairs
    ]
    if fmt == "json":
        path = os.path.join(output_dir, "top_confusion_pairs.json")
        with open(path, "w") as f:
            json.dump(overall_data, f, indent=2)
        saved["top_confusion_pairs"] = path

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Summary / comparison reports
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_text(reports: List[FullEvalReport]) -> str:
    """Generate a human-readable summary across all reports."""
    lines = []
    lines.append("=" * 90)
    lines.append("FAMILY-LEVEL EVALUATION SUMMARY")
    lines.append("=" * 90)

    lines.append(
        f"\n{'Dataset':<14s} {'Method':<14s} {'Fine Acc':>10s} "
        f"{'Family Acc':>11s} {'Acc Gap':>8s} "
        f"{'Fine mF1':>10s} {'Family mF1':>11s} {'mF1 Gap':>8s}"
    )
    lines.append("-" * 90)

    for r in reports:
        lines.append(
            f"{r.dataset:<14s} {r.algorithm:<14s} "
            f"{r.fine_grained.accuracy:>10.4f} {r.family_level.accuracy:>11.4f} "
            f"{r.family_fine_accuracy_gap:>+8.4f} "
            f"{r.fine_grained.macro_f1:>10.4f} {r.family_level.macro_f1:>11.4f} "
            f"{r.family_fine_macro_f1_gap:>+8.4f}"
        )

    lines.append("\n" + "=" * 90)
    lines.append("TOP CONFUSED FINE-GRAINED PAIRS (per dataset/method)")
    lines.append("=" * 90)

    for r in reports:
        if not r.top_overall_pairs:
            continue
        lines.append(f"\n--- {r.dataset} / {r.algorithm} (run {r.run_id}) ---")
        lines.append(f"{'Family':<30s} {'True':<20s} {'Predicted':<20s} "
                     f"{'Count':>8s} {'Intra-Family?':>14s}")
        lines.append("-" * 95)

        total_confusion = 0
        intra_confusion = 0
        for p in r.top_overall_pairs[:10]:
            is_intra = any(
                i.true_class == p.true_class and i.pred_class == p.pred_class
                for i in r.intra_family_pairs
            )
            lines.append(
                f"{p.family:<30s} {p.true_class:<20s} {p.pred_class:<20s} "
                f"{p.count:>8d} {'YES' if is_intra else 'no':>14s}"
            )
            total_confusion += p.count
            if is_intra:
                intra_confusion += p.count

        if total_confusion > 0:
            intra_pct = 100.0 * intra_confusion / total_confusion
            lines.append(
                f"\n  -> Among top-10 confusion pairs: "
                f"{intra_confusion}/{total_confusion} ({intra_pct:.0f}%) "
                f"are intra-family confusions."
            )
            if intra_pct > 50:
                lines.append("  -> Confusion is PRIMARILY within the same attack family.")
            else:
                lines.append("  -> Confusion is spread across different attack families.")

    return "\n".join(lines)


def generate_summary_json(reports: List[FullEvalReport]) -> Dict[str, Any]:
    """Generate a JSON-friendly summary structure."""
    entries = []
    for r in reports:
        entries.append({
            "dataset": r.dataset,
            "algorithm": r.algorithm,
            "goal": r.goal,
            "run_id": r.run_id,
            "fine_grained_accuracy": r.fine_grained.accuracy,
            "family_level_accuracy": r.family_level.accuracy,
            "family_fine_accuracy_gap": r.family_fine_accuracy_gap,
            "fine_grained_macro_f1": r.fine_grained.macro_f1,
            "family_level_macro_f1": r.family_level.macro_f1,
            "family_fine_macro_f1_gap": r.family_fine_macro_f1_gap,
            "num_intra_family_pairs_found": len(r.intra_family_pairs),
            "top_confusion_pairs": [
                {
                    "family": p.family,
                    "true_class": p.true_class,
                    "pred_class": p.pred_class,
                    "count": p.count,
                    "is_intra_family": any(
                        i.true_class == p.true_class and i.pred_class == p.pred_class
                        for i in r.intra_family_pairs
                    ),
                }
                for p in r.top_overall_pairs[:10]
            ],
        })
    return {"summary": entries}
