"""Common YAML configuration loading and section assembly."""

from pathlib import Path

import yaml


def load_yaml(path: str | Path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge_config_sections(*configs):
    merged = {}
    for config in configs:
        overlap = set(merged) & set(config)
        if overlap:
            raise ValueError(f"Configuration sections define duplicate top-level keys: {sorted(overlap)}.")
        merged.update(config)
    return merged