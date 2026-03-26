"""Configurable label remapping utility.

Used to convert non-contiguous label sets (e.g. BraTS {0,1,2,4}) to
contiguous class indices (e.g. {0,1,2,3}) in a dataset-agnostic way.
"""
from __future__ import annotations

from typing import Dict, Set

import numpy as np
import torch


class LabelRemapper:
    """Apply a fixed integer mapping to label arrays.

    Args:
        mapping: dict mapping source label → target label.
                 Labels not in the mapping are left unchanged unless
                 ``strict=True``.
        strict: if True, raise ValueError on unmapped label values.

    Example::

        remap = LabelRemapper({4: 3})
        remapped = remap(seg_array)  # 4 → 3, others unchanged
    """

    def __init__(self, mapping: Dict[int, int], strict: bool = False):
        self.mapping = dict(mapping)
        self.strict = strict

    def __call__(self, labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_torch = isinstance(labels, torch.Tensor)
        if is_torch:
            arr = labels.numpy() if not labels.is_cuda else labels.cpu().numpy()
        else:
            arr = np.array(labels, copy=True)

        if self.strict:
            unique = set(np.unique(arr).tolist())
            expected = set(self.mapping.keys())
            # Values that are also targets are allowed (identity mapping implied)
            all_known = expected | set(self.mapping.values())
            unmapped = unique - all_known
            if unmapped:
                raise ValueError(
                    f"Unmapped label values {unmapped} found. "
                    f"Mapping covers: {expected}"
                )

        out = arr.copy()
        for src, dst in self.mapping.items():
            out[arr == src] = dst
        return torch.from_numpy(out).to(labels.dtype) if is_torch else out

    @property
    def target_domain(self) -> Set[int]:
        """Set of label values after remapping (assuming only mapped values exist)."""
        return set(self.mapping.values())

    def verify_domain(
        self, labels: np.ndarray | torch.Tensor, expected: Set[int]
    ) -> bool:
        """Check that all values in *labels* are within *expected* set."""
        if isinstance(labels, torch.Tensor):
            unique = set(labels.unique().tolist())
        else:
            unique = set(np.unique(labels).tolist())
        unexpected = unique - {int(v) for v in expected}
        return len(unexpected) == 0


def remap_from_config(cfg_remap: dict | None) -> LabelRemapper | None:
    """Build a LabelRemapper from config dict, or return None if not configured.

    Config example::

        data:
          brats21:
            label_remap:
              4: 3
    """
    if not cfg_remap:
        return None
    mapping = {int(k): int(v) for k, v in cfg_remap.items()}
    return LabelRemapper(mapping)
