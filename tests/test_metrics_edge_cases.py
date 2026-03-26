"""Verification tests for Dice and HD95 metric edge cases."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.metrics import segmentation_metrics, _hd95_binary


def test_dice_identical():
    """Dice of identical non-trivial masks should be 1.0."""
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    pred[0, 4:12, 4:12] = 1
    target = pred.clone()
    result = segmentation_metrics(pred, target, num_classes=2, include_background=False, compute_hd95=False)
    assert result["dice_per_class"]["1"] == 1.0, f"Expected 1.0, got {result['dice_per_class']['1']}"
    print("PASS: Dice(identical) == 1.0")


def test_dice_no_overlap():
    """Dice of non-overlapping masks should be 0.0."""
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    pred[0, 0:4, 0:4] = 1
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[0, 12:16, 12:16] = 1
    result = segmentation_metrics(pred, target, num_classes=2, include_background=False, compute_hd95=False)
    assert result["dice_per_class"]["1"] == 0.0, f"Expected 0.0, got {result['dice_per_class']['1']}"
    print("PASS: Dice(no_overlap) == 0.0")


def test_dice_empty_pred_nonempty_gt():
    """Dice when pred is all background but GT has foreground should be 0.0."""
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[0, 4:12, 4:12] = 1
    result = segmentation_metrics(pred, target, num_classes=2, include_background=False, compute_hd95=False)
    assert result["dice_per_class"]["1"] == 0.0, f"Expected 0.0, got {result['dice_per_class']['1']}"
    print("PASS: Dice(empty_pred, non_empty_gt) == 0.0")


def test_dice_empty_gt_nonempty_pred():
    """Dice when GT is all background but pred has foreground should be 0.0."""
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    pred[0, 4:12, 4:12] = 1
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    result = segmentation_metrics(pred, target, num_classes=2, include_background=False, compute_hd95=False)
    assert result["dice_per_class"]["1"] == 0.0, f"Expected 0.0, got {result['dice_per_class']['1']}"
    print("PASS: Dice(non_empty_pred, empty_gt) == 0.0")


def test_dice_both_empty():
    """Dice when both pred and GT have no foreground should be 1.0 (vacuously correct)."""
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    result = segmentation_metrics(pred, target, num_classes=2, include_background=False, compute_hd95=False)
    assert result["dice_per_class"]["1"] == 1.0, f"Expected 1.0, got {result['dice_per_class']['1']}"
    print("PASS: Dice(both_empty) == 1.0")


def test_dice_multiclass():
    """Dice mean across multiple foreground classes."""
    # 3 classes: 0=bg, 1=class1, 2=class2
    pred = torch.zeros(1, 16, 16, dtype=torch.long)
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    # class 1: perfect match
    pred[0, 0:4, 0:4] = 1
    target[0, 0:4, 0:4] = 1
    # class 2: no overlap
    pred[0, 8:12, 0:4] = 2
    target[0, 8:12, 8:12] = 2
    result = segmentation_metrics(pred, target, num_classes=3, include_background=False, compute_hd95=False)
    assert result["dice_per_class"]["1"] == 1.0
    assert result["dice_per_class"]["2"] == 0.0
    assert result["dice_mean"] == 0.5, f"Expected 0.5, got {result['dice_mean']}"
    print("PASS: Dice multiclass mean == 0.5 (1.0 + 0.0) / 2")


def test_hd95_identical():
    """HD95 of identical masks should be 0.0."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    val = _hd95_binary(mask.astype(np.uint8), mask.astype(np.uint8))
    assert val == 0.0, f"Expected 0.0, got {val}"
    print("PASS: HD95(identical) == 0.0")


def test_hd95_both_empty():
    """HD95 when both masks are empty should be 0.0."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    val = _hd95_binary(mask.astype(np.uint8), mask.astype(np.uint8))
    assert val == 0.0, f"Expected 0.0, got {val}"
    print("PASS: HD95(both_empty) == 0.0")


def test_hd95_one_empty():
    """HD95 when one mask is empty should be NaN."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    empty = np.zeros((16, 16, 16), dtype=bool)

    val1 = _hd95_binary(mask.astype(np.uint8), empty.astype(np.uint8))
    assert math.isnan(val1), f"Expected NaN, got {val1}"

    val2 = _hd95_binary(empty.astype(np.uint8), mask.astype(np.uint8))
    assert math.isnan(val2), f"Expected NaN, got {val2}"
    print("PASS: HD95(one_empty) == NaN")


def test_hd95_known_distance():
    """HD95 for two separated blobs should reflect actual distance."""
    pred = np.zeros((32, 32, 32), dtype=bool)
    target = np.zeros((32, 32, 32), dtype=bool)
    # Two blocks separated by known gap
    pred[2:6, 2:6, 2:6] = True
    target[20:24, 20:24, 20:24] = True
    val = _hd95_binary(pred.astype(np.uint8), target.astype(np.uint8))
    assert val > 0.0, f"Expected positive distance, got {val}"
    assert not math.isnan(val), f"Expected finite distance, got NaN"
    print(f"PASS: HD95(separated_blobs) == {val:.2f} (positive, finite)")


def test_hd95_tiny_object():
    """HD95 for a 1-voxel object should not crash (erosion fallback)."""
    pred = np.zeros((16, 16, 16), dtype=bool)
    target = np.zeros((16, 16, 16), dtype=bool)
    pred[8, 8, 8] = True
    target[8, 8, 8] = True
    val = _hd95_binary(pred.astype(np.uint8), target.astype(np.uint8))
    assert val == 0.0, f"Expected 0.0 for identical single-voxel, got {val}"
    print("PASS: HD95(tiny_object) == 0.0 (erosion fallback works)")


def test_segmentation_metrics_with_hd95():
    """Full segmentation_metrics call with HD95 enabled."""
    pred = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    target = torch.zeros(1, 16, 16, 16, dtype=torch.long)
    pred[0, 4:12, 4:12, 4:12] = 1
    target[0, 4:12, 4:12, 4:12] = 1
    result = segmentation_metrics(pred, target, num_classes=2, include_background=False, compute_hd95=True)
    assert result["dice_per_class"]["1"] == 1.0
    assert result["hd95_per_class"]["1"] == 0.0
    assert result["dice_mean"] == 1.0
    assert result["hd95_mean"] == 0.0
    print("PASS: segmentation_metrics with HD95 (perfect match)")


if __name__ == "__main__":
    tests = [
        test_dice_identical,
        test_dice_no_overlap,
        test_dice_empty_pred_nonempty_gt,
        test_dice_empty_gt_nonempty_pred,
        test_dice_both_empty,
        test_dice_multiclass,
        test_hd95_identical,
        test_hd95_both_empty,
        test_hd95_one_empty,
        test_hd95_known_distance,
        test_hd95_tiny_object,
        test_segmentation_metrics_with_hd95,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
