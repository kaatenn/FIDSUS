"""Family label mapping utilities for attack classification evaluation.

Loads attack family mapping from YAML config and provides conversion
functions for fine-grained → family label mapping.
Used ONLY for evaluation/diagnosis, NOT for training.
"""

import yaml
from pathlib import Path


def load_family_mapping(config_path):
    """Load attack family mapping from YAML config file.

    Returns dict with structure:
      { 'NSLKDD': { 'mapping': {0: 'Normal', 1: 'DoS', ...},
                    'families': ['Normal', 'DoS', ...],
                    'label_names': {0: 'Normal', 1: 'DoS', ...} },
        'UNSW': {...},
        'UAV': {...} }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Family mapping config not found: {config_path}")

    with open(path, 'r') as f:
        mapping = yaml.safe_load(f)

    if not mapping:
        raise ValueError(f"Empty family mapping config: {config_path}")

    return mapping


def map_to_family(label_int, dataset, mapping):
    """Convert a fine-grained label integer to its family label string.

    Args:
        label_int: Integer label from the model
        dataset: Dataset name (e.g., 'NSLKDD', 'UNSW')
        mapping: Loaded family mapping dict

    Returns:
        Family label string, or str(label_int) if no mapping found
    """
    if dataset not in mapping:
        return str(label_int)
    ds_mapping = mapping[dataset].get('mapping', {})
    return ds_mapping.get(int(label_int), str(label_int))


def get_label_name(label_int, dataset, mapping):
    """Get human-readable label name for a given integer label."""
    if dataset not in mapping:
        return str(label_int)
    label_names = mapping[dataset].get('label_names', {})
    return label_names.get(int(label_int), str(label_int))


def map_labels_to_families(labels, dataset, mapping):
    """Convert array of fine-grained labels to family labels.

    Args:
        labels: Iterable of integer labels
        dataset: Dataset name
        mapping: Loaded family mapping dict

    Returns:
        List of family label strings
    """
    if dataset not in mapping:
        return [str(l) for l in labels]
    ds_mapping = mapping[dataset].get('mapping', {})
    return [ds_mapping.get(int(l), str(l)) for l in labels]


def get_families_list(dataset, mapping):
    """Get sorted list of family names for a dataset."""
    if dataset not in mapping:
        return []
    return sorted(mapping[dataset].get('families', []))


def compute_family_level_metrics(y_true, y_pred, dataset, mapping):
    """Compute accuracy at the family level.

    Args:
        y_true: Array of fine-grained true labels (integers)
        y_pred: Array of fine-grained predicted labels (integers)
        dataset: Dataset name
        mapping: Loaded family mapping dict

    Returns:
        dict with: family_accuracy, family_correct, family_total
    """
    if dataset not in mapping:
        return {'family_accuracy': 0.0, 'family_correct': 0, 'family_total': 0}

    ds_mapping = mapping[dataset].get('mapping', {})

    family_true = [ds_mapping.get(int(y), str(y)) for y in y_true]
    family_pred = [ds_mapping.get(int(y), str(y)) for y in y_pred]
    correct = sum(1 for t, p in zip(family_true, family_pred) if t == p)
    total = len(family_true)

    return {
        'family_accuracy': correct / max(total, 1),
        'family_correct': correct,
        'family_total': total,
        'family_true': family_true,
        'family_pred': family_pred,
    }
