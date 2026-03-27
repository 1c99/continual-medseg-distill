"""Student-side LoRA for Conv3d layers.

Provides a lightweight low-rank adapter for 3D convolutions, designed for
continual learning on a MONAI UNet student.  The base Conv3d weights are
frozen; only the rank-decomposed A/B matrices are trainable.

Usage::

    from src.models.lora import inject_lora, extract_lora_state, get_lora_params

    model = create_model(cfg)
    inject_lora(model, target_patterns=["conv.unit"], rank=8, alpha=16)
    optimizer = torch.optim.Adam(get_lora_params(model), lr=1e-3)
    # ... train ...
    state = extract_lora_state(model)
    torch.save(state, "lora_taskA.pt")
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRAConv3d(nn.Module):
    """Conv3d with a frozen base weight and a trainable low-rank adapter.

    output = base_conv(x) + (B @ A)(x) * (alpha / rank)

    where A is a 1x1x1 Conv3d projecting c_in → rank and B projects
    rank → c_out.  The base conv retains its original kernel size, stride,
    padding, etc.
    """

    def __init__(
        self,
        base_conv: nn.Conv3d,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.base_conv = base_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        c_in = base_conv.in_channels
        c_out = base_conv.out_channels

        # Low-rank decomposition via 1x1x1 convolutions
        self.lora_A = nn.Conv3d(c_in, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv3d(rank, c_out, kernel_size=1, bias=False)

        # Kaiming init for A, zero init for B (so adapter starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base conv
        for p in self.base_conv.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_conv(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        # Handle spatial size mismatch (base conv may have stride > 1)
        if lora_out.shape[2:] != base_out.shape[2:]:
            lora_out = F.interpolate(
                lora_out, size=base_out.shape[2:], mode="trilinear", align_corners=False
            )
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, "
            f"in={self.base_conv.in_channels}, out={self.base_conv.out_channels}"
        )


def _matches_any(name: str, patterns: List[str]) -> bool:
    """Check if module name contains any of the target patterns."""
    return any(p in name for p in patterns)


def inject_lora(
    model: nn.Module,
    target_patterns: List[str] | None = None,
    rank: int = 8,
    alpha: float = 16.0,
) -> int:
    """Replace matching Conv3d modules with LoRAConv3d wrappers in-place.

    Args:
        model: The model to modify.
        target_patterns: Substring patterns for module names to target.
            Defaults to ``["conv.unit"]`` which matches encoder/decoder
            conv blocks in the MONAI UNet.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Number of Conv3d modules replaced.
    """
    if target_patterns is None:
        target_patterns = ["conv.unit"]

    replacements: List[tuple] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) and _matches_any(name, target_patterns):
            replacements.append((name, module))

    count = 0
    for name, conv in replacements:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1]

        lora_conv = LoRAConv3d(conv, rank=rank, alpha=alpha)
        setattr(parent, attr, lora_conv)
        count += 1

    # Freeze all non-LoRA parameters
    for n, p in model.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad = False

    logger.info(
        f"inject_lora: replaced {count} Conv3d modules (rank={rank}, alpha={alpha})"
    )
    lora_count = sum(p.numel() for p in get_lora_params(model))
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"inject_lora: {lora_count:,} trainable LoRA params / "
        f"{total_count:,} total ({100 * lora_count / max(total_count, 1):.1f}%)"
    )
    return count


def get_lora_params(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only LoRA A/B parameters from the model."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def get_lora_named_params(model: nn.Module) -> Iterator[tuple]:
    """Yield (name, param) for LoRA A/B parameters."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, param


def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters as a state dict."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load saved LoRA parameters into the model."""
    model_state = model.state_dict()
    for key, value in state.items():
        if key in model_state:
            model_state[key] = value
        else:
            logger.warning(f"load_lora_state: key '{key}' not found in model")
    model.load_state_dict(model_state)


def has_lora(model: nn.Module) -> bool:
    """Check if a model has any LoRAConv3d modules."""
    for module in model.modules():
        if isinstance(module, LoRAConv3d):
            return True
    return False


def merge_lora(model: nn.Module) -> None:
    """Fold LoRA adapters into base Conv3d weights for inference.

    After merging, the LoRAConv3d wrappers are replaced with plain Conv3d
    modules containing the merged weights.  This is irreversible.
    """
    replacements: List[tuple] = []

    for name, module in model.named_modules():
        if isinstance(module, LoRAConv3d):
            replacements.append((name, module))

    for name, lora_module in replacements:
        base = lora_module.base_conv
        scaling = lora_module.scaling

        # Compute the LoRA delta: B @ A reshaped to match base kernel
        # A.weight: (rank, c_in, 1, 1, 1), B.weight: (c_out, rank, 1, 1, 1)
        A = lora_module.lora_A.weight.data  # (rank, c_in, 1, 1, 1)
        B = lora_module.lora_B.weight.data  # (c_out, rank, 1, 1, 1)

        # (c_out, rank) @ (rank, c_in) = (c_out, c_in)
        delta = (B.squeeze(-1).squeeze(-1).squeeze(-1) @
                 A.squeeze(-1).squeeze(-1).squeeze(-1))  # (c_out, c_in)

        # Pad to base kernel shape: (c_out, c_in, kD, kH, kW)
        k = base.weight.shape[2:]
        delta_kernel = torch.zeros_like(base.weight.data)
        # Place delta at center of kernel
        center = tuple(s // 2 for s in k)
        delta_kernel[:, :, center[0], center[1], center[2]] = delta * scaling

        # Merge
        merged_conv = nn.Conv3d(
            base.in_channels, base.out_channels,
            kernel_size=base.kernel_size, stride=base.stride,
            padding=base.padding, dilation=base.dilation,
            groups=base.groups, bias=base.bias is not None,
        )
        merged_conv.weight.data = base.weight.data + delta_kernel
        if base.bias is not None:
            merged_conv.bias.data = base.bias.data.clone()

        # Replace in parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], merged_conv)

    logger.info(f"merge_lora: merged {len(replacements)} LoRA adapters into base weights")
