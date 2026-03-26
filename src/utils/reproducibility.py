"""Reproducibility helpers: seed setting and environment info collection."""
from __future__ import annotations

import platform
import random
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for torch, numpy, random, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_info() -> Dict[str, Any]:
    """Get current git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())
    except Exception:
        dirty = False
    return {"git_commit": commit, "git_dirty": dirty}


def collect_env_info(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Collect environment metadata for reproducibility."""
    info: Dict[str, Any] = {}
    info["python_version"] = sys.version.split()[0]
    info["torch_version"] = torch.__version__
    try:
        import monai
        info["monai_version"] = monai.__version__
    except ImportError:
        info["monai_version"] = "not installed"
    info["platform"] = platform.platform()
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    info.update(get_git_info())
    if cfg:
        info["random_seed"] = cfg.get("experiment", {}).get("seed", "not set")
    return info
