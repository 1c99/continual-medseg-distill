"""Disk-backed cache for precomputed teacher outputs.

Avoids redundant teacher forward passes by caching logits/features
keyed by sample ID + config hash. Useful when the teacher is frozen
and the same samples are seen across multiple epochs.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class TeacherCache:
    """Disk-backed cache for teacher logits and features.

    Config keys (under ``method.kd.cache``):
        enabled: bool (default: false)
        dir: str (default: outputs/.teacher_cache)
    """

    def __init__(self, cache_dir: str | Path, config_hash: str):
        self._dir = Path(cache_dir) / config_hash
        self._dir.mkdir(parents=True, exist_ok=True)
        self._config_hash = config_hash
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(sample_id: str, config_hash: str) -> str:
        """Deterministic cache key from sample ID + config hash."""
        raw = f"{sample_id}::{config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _path(self, sample_id: str) -> Path:
        key = self.make_key(sample_id, self._config_hash)
        return self._dir / f"{key}.pt"

    def get(self, sample_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve cached teacher output. Returns None on miss."""
        path = self._path(sample_id)
        if path.exists():
            self._hits += 1
            return torch.load(path, map_location="cpu", weights_only=False)
        self._misses += 1
        return None

    def put(
        self,
        sample_id: str,
        logits: torch.Tensor,
        features: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Store teacher output to cache."""
        data: Dict[str, Any] = {"logits": logits.detach().cpu()}
        if features:
            data["features"] = {k: v.detach().cpu() for k, v in features.items()}
        path = self._path(sample_id)
        torch.save(data, path)

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "total": self._hits + self._misses}

    def invalidate(self) -> int:
        """Clear all cached entries. Returns number of files removed."""
        count = 0
        if self._dir.exists():
            for f in self._dir.glob("*.pt"):
                f.unlink()
                count += 1
        self._hits = 0
        self._misses = 0
        logger.info(f"Teacher cache invalidated: {count} entries removed")
        return count

    def __len__(self) -> int:
        return sum(1 for _ in self._dir.glob("*.pt")) if self._dir.exists() else 0
