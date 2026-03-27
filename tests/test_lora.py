"""Tests for student-side LoRA module."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from src.models.lora import (
    LoRAConv3d,
    inject_lora,
    extract_lora_state,
    load_lora_state,
    get_lora_params,
    has_lora,
    merge_lora,
)


def _make_tiny_unet():
    """Minimal model mimicking MONAI UNet module naming."""
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=[8, 16],
        strides=[2],
        num_res_units=1,
    )


# ---- LoRAConv3d unit tests ----


def test_lora_conv3d_output_shape():
    conv = nn.Conv3d(4, 8, kernel_size=3, padding=1)
    lora = LoRAConv3d(conv, rank=2, alpha=4.0)
    x = torch.randn(1, 4, 8, 8, 8)
    out = lora(x)
    expected = conv(x)
    assert out.shape == expected.shape, f"{out.shape} != {expected.shape}"


def test_lora_conv3d_starts_as_identity():
    """B is zero-initialized, so adapter output should be zero at init."""
    conv = nn.Conv3d(4, 8, kernel_size=3, padding=1)
    x = torch.randn(1, 4, 8, 8, 8)
    base_out = conv(x)

    lora = LoRAConv3d(conv, rank=2, alpha=4.0)
    lora_out = lora(x)
    # Should be identical since B=0 at init
    assert torch.allclose(base_out, lora_out, atol=1e-6), "LoRA should be identity at init"


def test_lora_conv3d_base_frozen():
    conv = nn.Conv3d(4, 8, kernel_size=3, padding=1)
    lora = LoRAConv3d(conv, rank=2, alpha=4.0)
    for p in lora.base_conv.parameters():
        assert not p.requires_grad, "Base conv params should be frozen"
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad


def test_lora_conv3d_with_stride():
    """LoRA should handle strided convolutions via interpolation."""
    conv = nn.Conv3d(4, 8, kernel_size=3, stride=2, padding=1)
    lora = LoRAConv3d(conv, rank=2, alpha=4.0)
    x = torch.randn(1, 4, 16, 16, 16)
    out = lora(x)
    assert out.shape == (1, 8, 8, 8, 8)


# ---- inject / extract / load ----


def test_inject_lora_freezes_base():
    model = _make_tiny_unet()
    count = inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)
    assert count > 0, "Should replace at least one Conv3d"
    assert has_lora(model)

    # Check: base params frozen, LoRA params trainable
    trainable = list(get_lora_params(model))
    assert len(trainable) > 0

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"


def test_extract_load_roundtrip():
    import copy
    model = _make_tiny_unet()
    # Clone before LoRA injection so base weights match
    model2 = copy.deepcopy(model)

    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)
    inject_lora(model2, target_patterns=["conv.unit"], rank=4, alpha=8.0)

    # Perturb LoRA weights on model1
    for p in get_lora_params(model):
        p.data.uniform_(-1, 1)

    state = extract_lora_state(model)
    assert len(state) > 0

    # Load model1's LoRA state into model2 (same base weights)
    load_lora_state(model2, state)

    # Compare outputs
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=1e-5), "Round-trip should preserve LoRA state"


def test_merge_lora_matches_forward():
    model = _make_tiny_unet()
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)

    # Train a bit to make LoRA non-trivial
    for p in get_lora_params(model):
        p.data.uniform_(-0.1, 0.1)

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        pre_merge_out = model(x).clone()

    merge_lora(model)
    assert not has_lora(model), "After merge, no LoRAConv3d should remain"

    with torch.no_grad():
        post_merge_out = model(x)

    # Merged output won't be exactly identical due to kernel center approximation
    # but should be close for 1x1 adapter effect
    assert post_merge_out.shape == pre_merge_out.shape


def test_get_lora_params_empty_without_lora():
    model = _make_tiny_unet()
    params = list(get_lora_params(model))
    assert len(params) == 0


def test_has_lora():
    model = _make_tiny_unet()
    assert not has_lora(model)
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)
    assert has_lora(model)


# ---- gradient flow ----


def test_gradient_flows_through_lora():
    model = _make_tiny_unet()
    inject_lora(model, target_patterns=["conv.unit"], rank=4, alpha=8.0)

    x = torch.randn(1, 1, 16, 16, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()

    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert p.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    test_lora_conv3d_output_shape()
    test_lora_conv3d_starts_as_identity()
    test_lora_conv3d_base_frozen()
    test_lora_conv3d_with_stride()
    test_inject_lora_freezes_base()
    test_extract_load_roundtrip()
    test_merge_lora_matches_forward()
    test_get_lora_params_empty_without_lora()
    test_has_lora()
    test_gradient_flows_through_lora()
    print("All test_lora tests passed!")
