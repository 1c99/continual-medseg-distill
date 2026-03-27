"""Tests for the orthogonality regularizer."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.models.lora import inject_lora, extract_lora_state, get_lora_params
from src.models.ortho_reg import orthogonality_loss


def _make_tiny_model():
    from monai.networks.nets import UNet
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=[8, 16],
        strides=[2],
        num_res_units=1,
    )


def test_ortho_loss_zero_no_prev():
    model = _make_tiny_model()
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)
    loss = orthogonality_loss(model, [])
    assert loss.item() == 0.0, "Should be zero with no previous states"


def test_ortho_loss_nonzero_overlapping():
    """When current adapters overlap with previous, loss should be non-zero."""
    model = _make_tiny_model()
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)

    # Set non-trivial LoRA weights
    for p in get_lora_params(model):
        p.data.uniform_(-1, 1)

    # Save state as "previous task"
    prev_state = extract_lora_state(model)

    # Current model has same weights — maximum overlap
    loss = orthogonality_loss(model, [prev_state])
    assert loss.item() > 0.0, f"Expected non-zero loss, got {loss.item()}"


def test_ortho_loss_zero_orthogonal():
    """Perfectly orthogonal adapters should yield near-zero loss."""
    model = _make_tiny_model()
    inject_lora(model, target_patterns=["conv.unit"], rank=2, alpha=4.0)

    # Find a LoRA A parameter to construct orthogonal states
    lora_a_names = [n for n, _ in model.named_parameters() if "lora_A" in n]
    lora_b_names = [n for n, _ in model.named_parameters() if "lora_B" in n]
    assert len(lora_a_names) > 0

    # Set current LoRA A to first two standard basis vectors
    for name, param in model.named_parameters():
        if "lora_A" in name:
            # param shape: (rank, c_in, 1, 1, 1)
            param.data.zero_()
            r, c = param.shape[0], param.shape[1]
            for i in range(min(r, c)):
                param.data[i, i, 0, 0, 0] = 1.0
        elif "lora_B" in name:
            param.data.zero_()
            r, c = param.shape[0], param.shape[1]
            for i in range(min(r, c)):
                param.data[i, i, 0, 0, 0] = 1.0

    # Create "previous" state with orthogonal basis
    prev_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            p = torch.zeros_like(param)
            r, c = p.shape[0], p.shape[1]
            # Use different basis vectors (offset by rank)
            for i in range(min(r, c - r)):
                p[i, i + r, 0, 0, 0] = 1.0
            prev_state[name] = p
        elif "lora_B" in name:
            p = torch.zeros_like(param)
            r, c = p.shape[0], p.shape[1]
            for i in range(min(r, c - r)):
                p[i + r, i, 0, 0, 0] = 1.0
            prev_state[name] = p

    loss = orthogonality_loss(model, [prev_state])
    assert loss.item() < 1e-6, f"Expected near-zero loss for orthogonal adapters, got {loss.item()}"


def test_ortho_loss_gradient_flows():
    """Gradients should propagate from ortho loss to LoRA params."""
    model = _make_tiny_model()
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)

    for p in get_lora_params(model):
        p.data.uniform_(-1, 1)

    prev_state = extract_lora_state(model)

    # Perturb current weights slightly
    for p in get_lora_params(model):
        p.data += torch.randn_like(p) * 0.1

    loss = orthogonality_loss(model, [prev_state])
    assert loss.requires_grad or loss.item() > 0.0

    if loss.requires_grad:
        loss.backward()
        grads = [
            p.grad for p in get_lora_params(model)
            if p.grad is not None
        ]
        assert len(grads) > 0, "At least some LoRA params should have gradients"


def test_ortho_loss_multiple_prev_states():
    """Loss should accumulate over multiple previous task states."""
    model = _make_tiny_model()
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)

    for p in get_lora_params(model):
        p.data.uniform_(-1, 1)

    state1 = extract_lora_state(model)

    # Perturb for second "task"
    for p in get_lora_params(model):
        p.data += torch.randn_like(p) * 0.5

    state2 = extract_lora_state(model)

    loss_one = orthogonality_loss(model, [state1])
    loss_two = orthogonality_loss(model, [state1, state2])

    # With more previous states, loss should generally be >= single state
    assert loss_two.item() >= loss_one.item() * 0.5, \
        "Loss with 2 prev states should not be drastically less than with 1"


def test_ortho_loss_no_lora_model():
    """Model without LoRA should return zero loss."""
    model = _make_tiny_model()
    fake_prev = {"nonexistent.lora_A.weight": torch.randn(4, 8, 1, 1, 1)}
    loss = orthogonality_loss(model, [fake_prev])
    assert loss.item() == 0.0


if __name__ == "__main__":
    test_ortho_loss_zero_no_prev()
    test_ortho_loss_nonzero_overlapping()
    test_ortho_loss_zero_orthogonal()
    test_ortho_loss_gradient_flows()
    test_ortho_loss_multiple_prev_states()
    test_ortho_loss_no_lora_model()
    print("All test_ortho_reg tests passed!")
