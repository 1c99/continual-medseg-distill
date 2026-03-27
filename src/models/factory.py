from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from monai.networks.nets import UNet

from src.models.lora import inject_lora

logger = logging.getLogger(__name__)


def create_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "monai_unet")

    # Keep backward compatibility with earlier placeholder names
    if name in {"tiny3d_unet", "monai_unet"}:
        model = UNet(
            spatial_dims=3,
            in_channels=mcfg.get("in_channels", 1),
            out_channels=mcfg.get("out_channels", 3),
            channels=tuple(mcfg.get("channels", [16, 32, 64, 128])),
            strides=tuple(mcfg.get("strides", [2, 2, 2])),
            num_res_units=mcfg.get("num_res_units", 2),
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    # Conditionally inject student-side LoRA adapters
    lora_cfg = mcfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        rank = int(lora_cfg.get("rank", 8))
        alpha = float(lora_cfg.get("alpha", 16.0))
        target_modules = list(lora_cfg.get("target_modules", ["conv.unit"]))
        count = inject_lora(model, target_patterns=target_modules, rank=rank, alpha=alpha)
        logger.info(f"Student LoRA: {count} layers, mode={lora_cfg.get('mode', 'standard')}")

    return model


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Alias for script compatibility."""
    return create_model(cfg)
