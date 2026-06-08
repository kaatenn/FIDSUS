"""Post-hoc diagnosis analysis for FIDSUS affinity/top-n experiments.

Reads raw diagnosis logs and produces:
  - topn_diagnosis_by_round.csv (round-level aggregated stats with random baseline)
  - fine_grained_metrics.csv / family_level_metrics.csv
  - family_fine_gap.csv
  - confusion matrices (fine + family level)
  - intra_family_confusion_pairs.csv
  - confusion_pair_similarity_match.csv

Usage:
  uv run python scripts/diagnosis_analysis.py \
      --input-dir audit/fidsus_real_training_diagnosis \
      --audit-dir docs/audit \
      --family-mapping configs/attack_family_mapping.yaml
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

# Add system to path for family_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "system"))
from utils.family_utils import load_family_mapping, map_labels_to_families, get_label_name


def read_client_label_profiles(raw_dir):
    path = raw_dir / "client_label_profile.csv"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return {}

    profiles = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row['client_id'])
            profiles[cid] = row
    return profiles


def read_topn_log(raw_dir):
    path = raw_dir / "topn_selection_log.csv"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []

    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_prediction_log(raw_dir):
    path = raw_dir / "prediction_log_fine.csv"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []

    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_topn_diagnosis(topn_rows, profiles, output_path, num_random=20):
    """Compute per-round per-client aggregated top-n diagnosis stats with random baseline."""
    if not topn_rows or not profiles:
        print("No top-n or profile data, skipping topn_diagnosis.")
        return

    # Group by (round, client_id)
    grouped = defaultdict(list)
    for row in topn_rows:
        key = (int(row['round']), int(row['client_id']))
        grouped[key].append(row)

    # Collect all client profiles for random sampling
    all_client_ids = sorted(profiles.keys())
    profile_map = {}
    for cid in all_client_ids:
        p = profiles[cid]
        profile_map[cid] = {
            'label_entropy': float(p.get('label_entropy', 0)),
            'normalized_label_entropy': float(p.get('normalized_label_entropy', 0)),
            'observed_class_count': int(p.get('observed_class_count', 0)),
            'js_to_global_distribution': float(p.get('js_to_global_distribution', 0)),
            'dominant_label_ratio': float(p.get('dominant_label_ratio', 0)),
            'dominant_label': int(p.get('dominant_label', 0)),
            'dominant_family': p.get('dominant_family', 'Unknown'),
            'dominant_family_ratio': float(p.get('dominant_family_ratio', 0)),
            'label_distribution_json': p.get('label_distribution_json', '{}'),
            'family_distribution_json': p.get('family_distribution_json', '{}'),
            'observed_family_count': int(p.get('observed_family_count', 0)),
            'family_entropy': float(p.get('family_entropy', 0)),
        }

    fieldnames = [
        'dataset', 'method', 'seed', 'round', 'client_id', 'topn_size',
        'mean_neighbor_entropy', 'mean_neighbor_normalized_entropy',
        'mean_neighbor_observed_class_count', 'mean_neighbor_js_to_global',
        'mean_neighbor_dominant_label_ratio', 'same_dominant_label_ratio',
        'same_dominant_family_ratio', 'neighbor_family_coverage',
        'neighbor_class_coverage',
        'label_distribution_similarity_mean',
        'family_distribution_similarity_mean',
        'random_baseline_mean_entropy', 'random_baseline_class_coverage',
        'random_baseline_js_to_global', 'random_baseline_same_dominant_label_ratio',
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (round_num, client_id), neighbor_rows in sorted(grouped.items()):
            first = neighbor_rows[0]
            nids = [int(r['neighbor_id']) for r in neighbor_rows]
            topn_size = len(nids)

            # Compute real top-n neighbor stats
            neigh_entropies = []
            neigh_norm_entropies = []
            neigh_obs_classes = []
            neigh_js_global = []
            neigh_dom_ratios = []
            same_dom_count = 0
            same_fam_count = 0
            neigh_families = set()
            neigh_classes = set()

            client_profile = profile_map.get(client_id, {})
            client_dom_label = client_profile.get('dominant_label', -1)
            client_dom_family = client_profile.get('dominant_family', '')

            for nid in nids:
                np_ = profile_map.get(nid, {})
                neigh_entropies.append(np_.get('label_entropy', 0))
                neigh_norm_entropies.append(np_.get('normalized_label_entropy', 0))
                neigh_obs_classes.append(np_.get('observed_class_count', 0))
                neigh_js_global.append(np_.get('js_to_global_distribution', 0))
                neigh_dom_ratios.append(np_.get('dominant_label_ratio', 0))

                if np_.get('dominant_label', -1) == client_dom_label:
                    same_dom_count += 1
                if np_.get('dominant_family', '') == client_dom_family:
                    same_fam_count += 1

                # Collect family/class coverage
                fam_json = np_.get('family_distribution_json', '{}')
                try:
                    fams = json.loads(fam_json)
                    neigh_families.update(fams.keys())
                except (json.JSONDecodeError, TypeError):
                    pass
                label_json = np_.get('label_distribution_json', '{}')
                try:
                    labs = json.loads(label_json)
                    neigh_classes.update(labs.keys())
                except (json.JSONDecodeError, TypeError):
                    pass

            # Label distribution similarity (JS distance between client and neighbor distributions)
            label_sims = []
            try:
                client_dist = json.loads(client_profile.get('label_distribution_json', '{}'))
            except (json.JSONDecodeError, TypeError):
                client_dist = {}
            for nid in nids:
                np_ = profile_map.get(nid, {})
                try:
                    neigh_dist = json.loads(np_.get('label_distribution_json', '{}'))
                except (json.JSONDecodeError, TypeError):
                    neigh_dist = {}
                all_classes = set(list(client_dist.keys()) + list(neigh_dist.keys()))
                cvec = [float(client_dist.get(c, 0)) for c in sorted(all_classes, key=int)]
                nvec = [float(neigh_dist.get(c, 0)) for c in sorted(all_classes, key=int)]
                csum = sum(cvec)
                nsum = sum(nvec)
                if csum > 0 and nsum > 0:
                    cvec = [v / csum for v in cvec]
                    nvec = [v / nsum for v in nvec]
                    js = _js_divergence(cvec, nvec)
                    label_sims.append(max(0, 1 - math.sqrt(max(js, 0))))

            fam_sims = []
            try:
                client_fam = json.loads(client_profile.get('family_distribution_json', '{}'))
            except (json.JSONDecodeError, TypeError):
                client_fam = {}
            for nid in nids:
                np_ = profile_map.get(nid, {})
                try:
                    neigh_fam = json.loads(np_.get('family_distribution_json', '{}'))
                except (json.JSONDecodeError, TypeError):
                    neigh_fam = {}
                all_fams = sorted(set(list(client_fam.keys()) + list(neigh_fam.keys())))
                cvec = [float(client_fam.get(f, 0)) for f in all_fams]
                nvec = [float(neigh_fam.get(f, 0)) for f in all_fams]
                csum = sum(cvec)
                nsum = sum(nvec)
                if csum > 0 and nsum > 0:
                    cvec = [v / csum for v in cvec]
                    nvec = [v / nsum for v in nvec]
                    js = _js_divergence(cvec, nvec)
                    fam_sims.append(max(0, 1 - math.sqrt(max(js, 0))))

            # Random baseline (average over num_random samples)
            rand_entropies = []
            rand_js = []
            rand_same_dom = []
            rand_all_classes = []
            for _ in range(num_random):
                rids = random.sample(all_client_ids, min(topn_size, len(all_client_ids)))
                r_ent = []
                r_js = []
                r_sd = 0
                r_classes = set()
                for rid in rids:
                    rp = profile_map.get(rid, {})
                    r_ent.append(rp.get('label_entropy', 0))
                    r_js.append(rp.get('js_to_global_distribution', 0))
                    if rp.get('dominant_label', -1) == client_dom_label:
                        r_sd += 1
                    try:
                        labs = json.loads(rp.get('label_distribution_json', '{}'))
                        r_classes.update(labs.keys())
                    except (json.JSONDecodeError, TypeError):
                        pass
                rand_entropies.append(np.mean(r_ent) if r_ent else 0)
                rand_js.append(np.mean(r_js) if r_js else 0)
                rand_same_dom.append(r_sd / max(len(rids), 1))
                rand_all_classes.append(len(r_classes))

            row = {
                'dataset': first.get('dataset', ''),
                'method': first.get('method', ''),
                'seed': first.get('seed', ''),
                'round': round_num,
                'client_id': client_id,
                'topn_size': topn_size,
                'mean_neighbor_entropy': round(np.mean(neigh_entropies) if neigh_entropies else 0, 6),
                'mean_neighbor_normalized_entropy': round(np.mean(neigh_norm_entropies) if neigh_norm_entropies else 0, 6),
                'mean_neighbor_observed_class_count': round(np.mean(neigh_obs_classes) if neigh_obs_classes else 0, 4),
                'mean_neighbor_js_to_global': round(np.mean(neigh_js_global) if neigh_js_global else 0, 8),
                'mean_neighbor_dominant_label_ratio': round(np.mean(neigh_dom_ratios) if neigh_dom_ratios else 0, 6),
                'same_dominant_label_ratio': round(same_dom_count / max(topn_size, 1), 6),
                'same_dominant_family_ratio': round(same_fam_count / max(topn_size, 1), 6),
                'neighbor_family_coverage': len(neigh_families),
                'neighbor_class_coverage': len(neigh_classes),
                'label_distribution_similarity_mean': round(np.mean(label_sims) if label_sims else 0, 6),
                'family_distribution_similarity_mean': round(np.mean(fam_sims) if fam_sims else 0, 6),
                'random_baseline_mean_entropy': round(np.mean(rand_entropies) if rand_entropies else 0, 6),
                'random_baseline_class_coverage': round(np.mean(rand_all_classes) if rand_all_classes else 0, 4),
                'random_baseline_js_to_global': round(np.mean(rand_js) if rand_js else 0, 8),
                'random_baseline_same_dominant_label_ratio': round(np.mean(rand_same_dom) if rand_same_dom else 0, 6),
            }
            writer.writerow(row)

    print(f"Wrote top-n diagnosis: {output_path}")


def _js_divergence(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl1 = sum(pi * math.log(max(pi, 1e-12) / max(mi, 1e-12)) for pi, mi in zip(p, m))
    kl2 = sum(qi * math.log(max(qi, 1e-12) / max(mi, 1e-12)) for qi, mi in zip(q, m))
    return (kl1 + kl2) / 2


def compute_prediction_metrics(pred_rows, family_mapping, processed_dir, metrics_dir):
    """Compute fine-grained and family-level metrics from prediction log."""
    if not pred_rows:
        print("No prediction data, skipping metrics.")
        return

    dataset = pred_rows[0].get('dataset', 'unknown')
    method = pred_rows[0].get('method', 'unknown')
    seed = pred_rows[0].get('seed', '0')

    # Collect by round
    by_round = defaultdict(list)
    for row in pred_rows:
        r = row.get('round', 'final')
        by_round[r].append(row)

    metrics_dir.mkdir(parents=True, exist_ok=True)

    fine_metrics_rows = []
    family_metrics_rows = []
    gap_rows = []
    fine_cm_rows = []
    family_cm_rows = []
    intra_confusion_rows = []

    for round_key, rows in sorted(by_round.items()):
        y_true = [int(r['y_true_id']) for r in rows]
        y_pred = [int(r['y_pred_id']) for r in rows]

        # Fine-grained metrics
        fine_m = compute_multiclass_metrics(y_true, y_pred)

        fine_metrics_rows.append({
            'dataset': dataset, 'method': method, 'seed': seed,
            'round': round_key,
            'accuracy': round(fine_m['accuracy'], 6),
            'macro_f1': round(fine_m['macro_f1'], 6),
            'weighted_f1': round(fine_m['weighted_f1'], 6),
            'per_class_precision': json.dumps({k: round(v, 4) for k, v in fine_m['per_class_precision'].items()}),
            'per_class_recall': json.dumps({k: round(v, 4) for k, v in fine_m['per_class_recall'].items()}),
            'per_class_f1': json.dumps({k: round(v, 4) for k, v in fine_m['per_class_f1'].items()}),
        })

        # Confusion matrix (fine)
        cm = _build_confusion_matrix(y_true, y_pred)
        fine_cm_rows.append({
            'dataset': dataset, 'method': method, 'seed': seed,
            'round': round_key,
            'confusion_matrix': json.dumps(cm),
            'labels': json.dumps(sorted(set(y_true + y_pred))),
        })

        # Family-level metrics
        if family_mapping and dataset in family_mapping:
            fm = family_mapping[dataset]
            ds_fam_map = fm.get('mapping', {})
            y_true_fam = [ds_fam_map.get(y, str(y)) for y in y_true]
            y_pred_fam = [ds_fam_map.get(y, str(y)) for y in y_pred]

            family_m = compute_multiclass_metrics_str(y_true_fam, y_pred_fam)

            family_metrics_rows.append({
                'dataset': dataset, 'method': method, 'seed': seed,
                'round': round_key,
                'accuracy': round(family_m['accuracy'], 6),
                'macro_f1': round(family_m['macro_f1'], 6),
                'weighted_f1': round(family_m['weighted_f1'], 6),
                'per_family_precision': json.dumps({k: round(v, 4) for k, v in family_m['per_class_precision'].items()}),
                'per_family_recall': json.dumps({k: round(v, 4) for k, v in family_m['per_class_recall'].items()}),
                'per_family_f1': json.dumps({k: round(v, 4) for k, v in family_m['per_class_f1'].items()}),
            })

            # Family confusion matrix
            fam_cm = _build_confusion_matrix_str(y_true_fam, y_pred_fam)
            family_cm_rows.append({
                'dataset': dataset, 'method': method, 'seed': seed,
                'round': round_key,
                'confusion_matrix': json.dumps(fam_cm),
                'labels': json.dumps(sorted(set(y_true_fam + y_pred_fam))),
            })

            # Family-fine gap
            gap_rows.append({
                'dataset': dataset, 'method': method, 'seed': seed,
                'round': round_key,
                'fine_accuracy': round(fine_m['accuracy'], 6),
                'family_accuracy': round(family_m['accuracy'], 6),
                'family_fine_accuracy_gap': round(family_m['accuracy'] - fine_m['accuracy'], 6),
                'fine_macro_f1': round(fine_m['macro_f1'], 6),
                'family_macro_f1': round(family_m['macro_f1'], 6),
                'family_fine_macroF1_gap': round(family_m['macro_f1'] - fine_m['macro_f1'], 6),
            })

            # Intra-family confusion analysis
            intra_rows = analyze_intra_family_confusion(
                y_true, y_pred, y_true_fam, y_pred_fam, ds_fam_map, fm)
            for ir in intra_rows:
                ir.update({'dataset': dataset, 'method': method, 'seed': seed, 'round': round_key})
                intra_confusion_rows.append(ir)

    # Write CSVs
    _write_csv(metrics_dir / "fine_grained_metrics.csv",
               ['dataset', 'method', 'seed', 'round', 'accuracy', 'macro_f1',
                'weighted_f1', 'per_class_precision', 'per_class_recall', 'per_class_f1'],
               fine_metrics_rows)

    _write_csv(metrics_dir / "family_level_metrics.csv",
               ['dataset', 'method', 'seed', 'round', 'accuracy', 'macro_f1',
                'weighted_f1', 'per_family_precision', 'per_family_recall', 'per_family_f1'],
               family_metrics_rows)

    _write_csv(metrics_dir / "family_fine_gap.csv",
               ['dataset', 'method', 'seed', 'round', 'fine_accuracy', 'family_accuracy',
                'family_fine_accuracy_gap', 'fine_macro_f1', 'family_macro_f1',
                'family_fine_macroF1_gap'],
               gap_rows)

    _write_csv(metrics_dir / "fine_grained_confusion_matrix.csv",
               ['dataset', 'method', 'seed', 'round', 'confusion_matrix', 'labels'],
               fine_cm_rows)

    _write_csv(metrics_dir / "family_level_confusion_matrix.csv",
               ['dataset', 'method', 'seed', 'round', 'confusion_matrix', 'labels'],
               family_cm_rows)

    _write_csv(metrics_dir / "intra_family_confusion_pairs.csv",
               ['dataset', 'method', 'seed', 'round', 'true_label', 'pred_label',
                'true_family', 'pred_family', 'is_intra_family', 'confusion_count',
                'confusion_rate', 'pair_rank'],
               intra_confusion_rows)

    print(f"Wrote metrics to {metrics_dir}")


def compute_multiclass_metrics(y_true, y_pred):
    """Compute accuracy, macro_f1, weighted_f1, per-class metrics."""
    classes = sorted(set(y_true + y_pred))
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    # Per-class metrics
    per_class = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[c] = {'precision': precision, 'recall': recall, 'f1': f1}

    macro_f1 = np.mean([per_class[c]['f1'] for c in classes]) if classes else 0

    # Weighted F1
    class_counts = Counter(y_true)
    weighted_sum = sum(class_counts.get(c, 0) * per_class[c]['f1'] for c in classes)
    weighted_f1 = weighted_sum / max(total, 1)

    return {
        'accuracy': correct / max(total, 1),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': {str(c): per_class[c]['precision'] for c in classes},
        'per_class_recall': {str(c): per_class[c]['recall'] for c in classes},
        'per_class_f1': {str(c): per_class[c]['f1'] for c in classes},
    }


def compute_multiclass_metrics_str(y_true_str, y_pred_str):
    """Same as compute_multiclass_metrics but for string labels."""
    classes = sorted(set(y_true_str + y_pred_str))
    total = len(y_true_str)
    correct = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == p)

    per_class = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true_str, y_pred_str) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == c and p != c)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[c] = {'precision': precision, 'recall': recall, 'f1': f1}

    macro_f1 = np.mean([per_class[c]['f1'] for c in classes]) if classes else 0
    class_counts = Counter(y_true_str)
    weighted_sum = sum(class_counts.get(c, 0) * per_class[c]['f1'] for c in classes)
    weighted_f1 = weighted_sum / max(total, 1)

    return {
        'accuracy': correct / max(total, 1),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': {c: per_class[c]['precision'] for c in classes},
        'per_class_recall': {c: per_class[c]['recall'] for c in classes},
        'per_class_f1': {c: per_class[c]['f1'] for c in classes},
    }


def _build_confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true + y_pred))
    cm = {}
    for true_c in classes:
        for pred_c in classes:
            count = sum(1 for t, p in zip(y_true, y_pred) if t == true_c and p == pred_c)
            if count > 0:
                cm[f"{true_c}->{pred_c}"] = count
    return cm


def _build_confusion_matrix_str(y_true_str, y_pred_str):
    classes = sorted(set(y_true_str + y_pred_str))
    cm = {}
    for true_c in classes:
        for pred_c in classes:
            count = sum(1 for t, p in zip(y_true_str, y_pred_str) if t == true_c and p == pred_c)
            if count > 0:
                cm[f"{true_c}->{pred_c}"] = count
    return cm


def analyze_intra_family_confusion(y_true, y_pred, y_true_fam, y_pred_fam,
                                    fine_to_family, family_mapping):
    """Analyze intra-family confusion pairs."""
    total = len(y_true)
    # Count confusion pairs
    pair_counts = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            pair_counts[(t, p)] += 1

    if not pair_counts:
        return []

    # Rank by count
    ranked = sorted(pair_counts.items(), key=lambda x: -x[1])
    rows = []
    for rank, ((t, p), count) in enumerate(ranked, 1):
        t_fam = fine_to_family.get(t, str(t))
        p_fam = fine_to_family.get(p, str(p))
        is_intra = (t_fam == p_fam)
        rows.append({
            'true_label': t,
            'pred_label': p,
            'true_family': t_fam,
            'pred_family': p_fam,
            'is_intra_family': int(is_intra),
            'confusion_count': count,
            'confusion_rate': round(count / max(total, 1), 6),
            'pair_rank': rank,
        })

    return rows


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_audit_attack_similarity(audit_dir):
    """Try to read attack similarity data from audit reports."""
    similarity_data = {}
    audit_path = Path(audit_dir)

    # Check for existing similarity CSVs/JSONs
    for pattern in ['*similarity*', '*affinity*', '*attack_matrix*']:
        for f in audit_path.glob(pattern):
            try:
                if f.suffix == '.csv':
                    with open(f, 'r') as fh:
                        reader = csv.DictReader(fh)
                        similarity_data[f.stem] = list(reader)
                elif f.suffix == '.json':
                    with open(f, 'r') as fh:
                        similarity_data[f.stem] = json.load(fh)
            except Exception as e:
                print(f"  Could not read {f}: {e}")

    # Parse from audit TXT files
    for txt_file in audit_path.glob('*.txt'):
        try:
            with open(txt_file, 'r') as fh:
                content = fh.read()
            # Look for similarity matrix sections
            if 'similarity matrix' in content.lower():
                dataset_key = txt_file.stem.replace('_audit_report', '')
                similarity_data[dataset_key] = _parse_similarity_from_txt(content)
        except Exception as e:
            print(f"  Could not parse {txt_file}: {e}")

    return similarity_data


def _parse_similarity_from_txt(content):
    """Simple parser for similarity matrix in audit TXT files."""
    pairs = {}
    in_matrix = False
    header = []
    for line in content.split('\n'):
        line = line.strip()
        if 'Attack similarity matrix' in line:
            in_matrix = True
            continue
        if in_matrix and not line:
            in_matrix = False
            continue
        if in_matrix and line:
            parts = line.split()
            if not parts:
                continue
            if len(parts) > 1 and parts[0] not in ['', ' ']:
                name = parts[0]
                values = parts[1:]
                if header and len(values) == len(header):
                    for h, v in zip(header, values):
                        try:
                            pairs[f"{name}-{h}"] = float(v)
                        except ValueError:
                            pass
                elif not header:
                    # This row might be the header
                    header = parts
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Diagnosis analysis for FIDSUS affinity/top-n")
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to diagnosis output directory')
    parser.add_argument('--audit-dir', type=str, default='docs/audit',
                        help='Path to audit reports directory')
    parser.add_argument('--family-mapping', type=str, default='configs/attack_family_mapping.yaml',
                        help='Path to attack family mapping YAML')
    parser.add_argument('--random-repeats', type=int, default=20,
                        help='Number of random baseline repeats')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    raw_dir = input_dir / "raw"
    processed_dir = input_dir / "processed"
    metrics_dir = input_dir / "metrics"

    print(f"Input: {input_dir}")
    print(f"Audit: {args.audit_dir}")
    print(f"Family mapping: {args.family_mapping}")

    # Load data
    profiles = read_client_label_profiles(raw_dir)
    print(f"Loaded {len(profiles)} client profiles")

    topn_rows = read_topn_log(raw_dir)
    print(f"Loaded {len(topn_rows)} top-n selection records")

    pred_rows = read_prediction_log(raw_dir)
    print(f"Loaded {len(pred_rows)} prediction records")

    # Load family mapping
    try:
        family_mapping = load_family_mapping(args.family_mapping)
    except Exception as e:
        print(f"WARNING: Could not load family mapping: {e}")
        print("Family-level metrics will be skipped.")
        family_mapping = {}

    # Load attack similarity from audit dir
    print(f"\nReading audit data from {args.audit_dir}...")
    similarity_data = read_audit_attack_similarity(args.audit_dir)
    if similarity_data:
        print(f"  Found similarity data for: {list(similarity_data.keys())}")
    else:
        print("  No structured similarity data found in audit dir (will compute from preds)")

    # Compute top-n diagnosis
    print("\nComputing top-n diagnosis...")
    compute_topn_diagnosis(topn_rows, profiles,
                           processed_dir / "topn_diagnosis_by_round.csv",
                           num_random=args.random_repeats)

    # Compute prediction metrics
    print("\nComputing prediction metrics...")
    compute_prediction_metrics(pred_rows, family_mapping, processed_dir, metrics_dir)

    # Compute confusion pair similarity match (if audit similarity available)
    if similarity_data:
        print("\nComputing confusion pair similarity match...")
        confusion_file = metrics_dir / "intra_family_confusion_pairs.csv"
        if confusion_file.exists():
            match_rows = _compute_confusion_pair_similarity_match(
                confusion_file, similarity_data, pred_rows, args.random_repeats)
            _write_csv(metrics_dir / "confusion_pair_similarity_match.csv",
                       ['dataset', 'method', 'seed', 'true_label', 'pred_label',
                        'true_family', 'pred_family', 'is_intra_family',
                        'confusion_count', 'confusion_rate', 'attack_similarity_score',
                        'is_high_similarity_pair', 'pair_rank'],
                       match_rows)
            print(f"  Wrote confusion pair similarity match")

    print("\nDone! Analysis outputs in:", input_dir)


def _compute_confusion_pair_similarity_match(confusion_path, similarity_data,
                                              pred_rows_dict, random_repeats):
    """Match confusion pairs with attack similarity scores from audit data."""
    rows = []
    with open(confusion_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = row['true_label']
            pred_label = row['pred_label']
            score = 0.0
            is_high = False
            # Try to find similarity
            for ds_key, sim_data in similarity_data.items():
                if isinstance(sim_data, dict):
                    for pair_key, sim_val in sim_data.items():
                        if true_label in pair_key and pred_label in pair_key:
                            score = sim_val
                            is_high = score > 0.7
                            break
            row['attack_similarity_score'] = score
            row['is_high_similarity_pair'] = int(is_high)
            rows.append(row)
    return rows


if __name__ == '__main__':
    main()
