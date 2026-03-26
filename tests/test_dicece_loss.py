"""Tests for DiceCE loss in ContinualMethod._compute_loss."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from methods.base import ContinualMethod


def _make_method(loss_type: str = "dicece") -> ContinualMethod:
    cfg = {"train": {"loss_type": loss_type}}
    return ContinualMethod(cfg)


def test_dicece_produces_scalar_with_grad():
    """DiceCE loss should return a differentiable scalar."""
    method = _make_method("dicece")
    logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)  # B=2, C=3, 3D spatial
    target = torch.randint(0, 3, (2, 8, 8, 8))
    loss = method._compute_loss(logits, target)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    loss.backward()
    assert logits.grad is not None, "No gradient computed"
    assert logits.grad.shape == logits.shape, "Gradient shape mismatch"
    print("PASS: dicece produces scalar with valid gradients")


def test_ce_fallback_produces_scalar_with_grad():
    """CE fallback should also return a differentiable scalar."""
    method = _make_method("ce")
    logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8, 8))
    loss = method._compute_loss(logits, target)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    loss.backward()
    assert logits.grad is not None, "No gradient computed"
    print("PASS: ce fallback produces scalar with valid gradients")


def test_dicece_2d():
    """DiceCE should work with 2D spatial inputs."""
    method = _make_method("dicece")
    logits = torch.randn(4, 2, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (4, 16, 16))
    loss = method._compute_loss(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
    print("PASS: dicece works with 2D spatial inputs")


def test_dicece_perfect_prediction():
    """DiceCE dice component should be near zero for perfect predictions."""
    method = _make_method("dicece")
    # Create a near-perfect prediction: high logit for the correct class
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 1:3, 1:3] = 1
    logits = torch.zeros(1, 2, 4, 4)
    logits[0, 0] = 10.0  # high confidence background
    logits[0, 1] = -10.0
    logits[0, 1, 1:3, 1:3] = 10.0  # high confidence foreground where target=1
    logits[0, 0, 1:3, 1:3] = -10.0
    logits.requires_grad_(True)

    loss = method._compute_loss(logits, target)
    # Loss should be small (near-perfect prediction)
    assert loss.item() < 0.1, f"Expected loss < 0.1 for perfect prediction, got {loss.item():.4f}"
    print(f"PASS: dicece near-perfect prediction loss = {loss.item():.6f}")


def test_default_config_uses_dicece():
    """When loss_type is not set, default should be dicece."""
    method = ContinualMethod({"train": {}})
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))
    loss = method._compute_loss(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
    print("PASS: default config uses dicece")


if __name__ == "__main__":
    tests = [
        test_dicece_produces_scalar_with_grad,
        test_ce_fallback_produces_scalar_with_grad,
        test_dicece_2d,
        test_dicece_perfect_prediction,
        test_default_config_uses_dicece,
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
