"""OOM fallback guard for GPU training.

Wraps a training step and catches CUDA out-of-memory errors,
logging a warning and clearing the cache. Callers can retry
with a smaller effective batch.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


def oom_guard(fn: Callable[..., T], *args, **kwargs) -> T | None:
    """Call *fn* and catch CUDA OOM errors.

    Returns the result of *fn*, or None if OOM occurred.
    Logs a warning and clears CUDA cache on OOM.
    """
    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "CUDA OOM detected. Clearing cache. "
            "Consider reducing batch_size, patch_size, or enabling AMP."
        )
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(
                "CUDA OOM (RuntimeError). Clearing cache. "
                "Consider reducing batch_size, patch_size, or enabling AMP."
            )
            torch.cuda.empty_cache()
            return None
        raise
