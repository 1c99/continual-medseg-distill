from __future__ import annotations

from typing import Any, Dict
import torch
from monai.networks.nets import UNet


def create_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "monai_unet")

    # Keep backward compatibility with earlier placeholder names
    if name in {"tiny3d_unet", "monai_unet"}:
        return UNet(
            spatial_dims=3,
            in_channels=mcfg.get("in_channels", 1),
            out_channels=mcfg.get("out_channels", 3),
            channels=tuple(mcfg.get("channels", [16, 32, 64, 128])),
            strides=tuple(mcfg.get("strides", [2, 2, 2])),
            num_res_units=mcfg.get("num_res_units", 2),
        )

    raise ValueError(f"Unsupported model: {name}")


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Alias for script compatibility."""
    return create_model(cfg)
