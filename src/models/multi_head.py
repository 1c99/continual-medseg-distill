"""Multi-head output wrapper for task-incremental continual learning.

Wraps a MONAI UNet backbone, replacing the single output block with
per-task output heads. Each task gets its own output projection while
sharing the encoder/decoder backbone.

Usage:
    wrapper = MultiHeadWrapper(unet, task_id="taskA_organs", out_channels=15)
    wrapper.register_head("taskB_muscles", out_channels=11)
    wrapper.current_task = "taskB_muscles"
    logits = wrapper(x)  # Uses taskB head (11 channels)
"""
from __future__ import annotations

import copy
import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiHeadWrapper(nn.Module):
    """Wraps a MONAI UNet with per-task output heads.

    The UNet's final output block (model.model[2]) is extracted and stored
    as the first task's head. The backbone (encoder + decoder without output
    block) is shared. New heads are created by deep-copying the original
    output block architecture and reinitializing weights for different
    output channel counts.
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_id: str,
        out_channels: int,
    ):
        super().__init__()
        self.backbone = backbone

        # Extract the output block from the UNet
        # MONAI UNet structure: model = Sequential(encoder, skip_decoder, output_block)
        self._output_block_template = backbone.model[2]

        # Replace with identity so backbone outputs decoder features
        backbone.model[2] = nn.Identity()

        # Store the original output block as the first task's head
        self.heads = nn.ModuleDict({task_id: self._output_block_template})
        self.current_task: str = task_id

        # Track out_channels per task for diagnostics
        self._head_channels: Dict[str, int] = {task_id: out_channels}

        logger.info(
            f"MultiHeadWrapper: backbone shared, head '{task_id}' "
            f"({out_channels} ch) registered"
        )

    def register_head(self, task_id: str, out_channels: int) -> None:
        """Register a new task-specific output head.

        Creates a fresh output block with the specified number of output
        channels, matching the architecture of the original output block
        but with reinitialized weights.
        """
        if task_id in self.heads:
            logger.info(f"Head '{task_id}' already registered, skipping")
            return

        from monai.networks.blocks import Convolution, ResidualUnit

        # Read input channels from the original output block's ConvTranspose3d
        in_ch = self._output_block_template[0].conv.in_channels
        stride = self._output_block_template[0].conv.stride

        new_head = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=in_ch,
                out_channels=out_channels,
                strides=stride,
                is_transposed=True,
            ),
            ResidualUnit(
                spatial_dims=3,
                in_channels=out_channels,
                out_channels=out_channels,
            ),
        )

        self.heads[task_id] = new_head
        self._head_channels[task_id] = out_channels
        logger.info(
            f"MultiHeadWrapper: head '{task_id}' ({out_channels} ch) registered "
            f"(total heads: {len(self.heads)})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.heads[self.current_task](features)

    @property
    def head_channels(self) -> Dict[str, int]:
        return dict(self._head_channels)

    @property
    def task_ids(self) -> list:
        return list(self.heads.keys())
