from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from src.models.lora import inject_lora
from src.models.multi_head import MultiHeadWrapper

logger = logging.getLogger(__name__)


def _build_nnunet(mcfg: Dict[str, Any], out_channels: int) -> nn.Module:
    """Build an nnU-Net PlainConvUNet (TotalSegmentator-style architecture)."""
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    n_stages = mcfg.get("n_stages", 5)
    features = mcfg.get("features_per_stage", [32, 64, 128, 256, 512])
    n_conv = mcfg.get("n_conv_per_stage", [2] * n_stages)
    n_conv_dec = mcfg.get("n_conv_per_stage_decoder", [2] * (n_stages - 1))

    return PlainConvUNet(
        input_channels=mcfg.get("in_channels", 1),
        n_stages=n_stages,
        features_per_stage=features[:n_stages],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
        num_classes=out_channels,
        n_conv_per_stage=n_conv[:n_stages],
        n_conv_per_stage_decoder=n_conv_dec[:n_stages - 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        nonlin=nn.LeakyReLU,
        deep_supervision=False,
    )


class NNUNetMultiHead(nn.Module):
    """Multi-head wrapper for nnU-Net's PlainConvUNet.

    Replaces decoder.seg_layers[-1] with per-task Conv3d heads.
    """

    def __init__(self, nnunet: nn.Module, task_id: str, out_channels: int):
        super().__init__()
        self.model = nnunet

        # Extract the final seg layer (Conv3d(features[0], num_classes, 1))
        orig_seg = nnunet.decoder.seg_layers[-1]
        self._in_features = orig_seg.in_channels

        # Replace with identity — decoder will return raw features
        nnunet.decoder.seg_layers[-1] = nn.Identity()

        # Per-task heads
        self.heads = nn.ModuleDict({task_id: orig_seg})
        self.current_task: str = task_id
        self._head_channels: Dict[str, int] = {task_id: out_channels}

        logger.info(
            f"NNUNetMultiHead: head '{task_id}' ({out_channels} ch) registered, "
            f"in_features={self._in_features}"
        )

    def register_head(self, task_id: str, out_channels: int) -> None:
        if task_id in self.heads:
            return
        self.heads[task_id] = nn.Conv3d(
            self._in_features, out_channels, kernel_size=1
        )
        self._head_channels[task_id] = out_channels
        device = next(self.model.parameters()).device
        self.heads[task_id].to(device)
        logger.info(
            f"NNUNetMultiHead: head '{task_id}' ({out_channels} ch) registered"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Model returns features (decoder.seg_layers[-1] is Identity)
        features = self.model(x)
        return self.heads[self.current_task](features)

    @property
    def head_channels(self) -> Dict[str, int]:
        return dict(self._head_channels)

    @property
    def task_ids(self) -> list:
        return list(self.heads.keys())


def create_model(
    cfg: Dict[str, Any],
    task_id: str | None = None,
) -> torch.nn.Module:
    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "monai_unet")
    out_channels = mcfg.get("out_channels", 3)

    if name in {"tiny3d_unet", "monai_unet"}:
        unet = UNet(
            spatial_dims=3,
            in_channels=mcfg.get("in_channels", 1),
            out_channels=out_channels,
            channels=tuple(mcfg.get("channels", [16, 32, 64, 128])),
            strides=tuple(mcfg.get("strides", [2, 2, 2])),
            num_res_units=mcfg.get("num_res_units", 2),
        )
    elif name == "nnunet":
        unet = _build_nnunet(mcfg, out_channels)
    else:
        raise ValueError(f"Unsupported model: {name}")

    # Conditionally inject student-side LoRA adapters
    lora_cfg = mcfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        rank = int(lora_cfg.get("rank", 8))
        alpha = float(lora_cfg.get("alpha", 16.0))
        target_modules = list(lora_cfg.get("target_modules", ["conv.unit"]))
        count = inject_lora(unet, target_patterns=target_modules, rank=rank, alpha=alpha)
        logger.info(f"Student LoRA: {count} layers, mode={lora_cfg.get('mode', 'standard')}")

    # Wrap in multi-head for continual learning
    if task_id is not None:
        if name == "nnunet":
            model = NNUNetMultiHead(unet, task_id=task_id, out_channels=out_channels)
        else:
            model = MultiHeadWrapper(unet, task_id=task_id, out_channels=out_channels)
    else:
        model = unet

    return model


def build_model(
    cfg: Dict[str, Any],
    task_id: str | None = None,
) -> torch.nn.Module:
    """Alias for script compatibility."""
    return create_model(cfg, task_id=task_id)
