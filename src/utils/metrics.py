from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
import torch


class CSVMetricsLogger:
    """Minimal CSV logger for epoch-wise metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: Dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of boundary voxels."""
    from scipy.ndimage import binary_erosion

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return mask & ~eroded


def _hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute symmetric 95th percentile Hausdorff distance for binary masks."""
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = int(pred.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        return math.nan

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    # Fallback if erosion removed all voxels (tiny objects).
    if not np.any(pred_surface):
        pred_surface = pred
    if not np.any(target_surface):
        target_surface = target

    dt_target = distance_transform_edt(~target)
    dt_pred = distance_transform_edt(~pred)

    d_pred_to_target = dt_target[pred_surface]
    d_target_to_pred = dt_pred[target_surface]

    if d_pred_to_target.size == 0 or d_target_to_pred.size == 0:
        return math.nan

    return float(
        max(
            np.percentile(d_pred_to_target, 95),
            np.percentile(d_target_to_pred, 95),
        )
    )


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    compute_hd95: bool = True,
) -> Dict[str, Any]:
    """Compute production-grade segmentation metrics from label maps.

    Args:
        pred: Predicted class indices, shape [B, ...].
        target: Ground-truth class indices, shape [B, ...].
        num_classes: Number of semantic classes.
        include_background: Whether to include class 0 in class-wise/mean metrics.
        compute_hd95: Whether to attempt HD95 computation.
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    class_start = 0 if include_background else 1
    class_ids = list(range(class_start, num_classes))

    dice_per_class: Dict[str, float] = {}
    hd95_per_class: Dict[str, float] = {}
    warnings_list: list[str] = []

    hd95_available = True
    if compute_hd95:
        try:
            import scipy  # noqa: F401
        except Exception:
            hd95_available = False
            warnings_list.append("HD95 unavailable (scipy not installed); reporting NaN.")
            warnings.warn(warnings_list[-1])

    for cls in class_ids:
        pred_c = pred_np == cls
        target_c = target_np == cls

        intersection = float(np.logical_and(pred_c, target_c).sum())
        pred_sum = float(pred_c.sum())
        target_sum = float(target_c.sum())
        denom = pred_sum + target_sum

        if denom == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / denom
        dice_per_class[str(cls)] = float(dice)

        hd95 = math.nan
        if compute_hd95 and hd95_available:
            try:
                hd95 = _hd95_binary(pred_c, target_c)
            except Exception as e:  # pragma: no cover - defensive path
                msg = f"HD95 failed for class {cls}: {e}. Reporting NaN."
                warnings_list.append(msg)
                warnings.warn(msg)
        hd95_per_class[str(cls)] = float(hd95)

    return {
        "dice_mean": _nanmean(dice_per_class.values()),
        "dice_per_class": dice_per_class,
        "hd95_mean": _nanmean(hd95_per_class.values()),
        "hd95_per_class": hd95_per_class,
        "warnings": warnings_list,
    }
