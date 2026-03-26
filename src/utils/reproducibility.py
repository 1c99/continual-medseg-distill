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


def set_deterministic_mode(seed: int) -> None:
    """Set seeds and enable deterministic backends where possible."""
    set_seed(seed)
    torch.use_deterministic_algorithms(False)  # True breaks some MONAI ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn for reproducible multi-worker loading.

    Each worker gets ``base_seed + worker_id`` so workers produce
    different-but-deterministic augmentation sequences.
    """
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed if worker_info else 0
    seed = (base_seed + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
