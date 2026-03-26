from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import copy
import yaml


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_config(base_config_path: str | Path) -> Dict[str, Any]:
    base_path = Path(base_config_path).resolve()
    root = base_path.parents[1]  # repo root from configs/base.yaml

    cfg = load_yaml(base_path)

    include_paths = [
        cfg.get("includes", {}).get("task"),
        cfg.get("includes", {}).get("method"),
        cfg.get("includes", {}).get("dataset"),
    ]

    for rel in include_paths:
        if not rel:
            continue
        inc_cfg = load_yaml(root / rel)
        cfg = merge_dicts(cfg, inc_cfg)

    return cfg
