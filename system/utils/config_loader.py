import json
import argparse
import logging

logger = logging.getLogger(__name__)


def dict_to_namespace(d: dict) -> argparse.Namespace:
    """Recursively convert a dict to argparse.Namespace."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return argparse.Namespace(**d)


def load_config(config_path: str) -> list[argparse.Namespace]:
    """Load experiment configurations from a JSON file.

    The JSON file must have an "experiments" array. An optional "defaults"
    object provides shared values that each experiment can override.

    Returns a list of argparse.Namespace, one per experiment.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    defaults = config.get("defaults", {})
    experiments_raw = config.get("experiments", [])

    if not experiments_raw:
        raise ValueError(f"No experiments defined in '{config_path}'.")

    experiments = []
    for exp in experiments_raw:
        merged = {**defaults, **exp}
        if "name" not in merged:
            merged["name"] = f"{merged.get('algorithm', '?')}_{merged.get('dataset', '?')}_{merged.get('goal', '?')}"
        ns = dict_to_namespace(merged)
        experiments.append(ns)

    return experiments
