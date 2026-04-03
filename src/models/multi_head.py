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

        Creates a fresh output block by deep-copying the original output
        block architecture and patching Conv/Norm layers for the new
        output channel count.
        """
        if task_id in self.heads:
            logger.info(f"Head '{task_id}' already registered, skipping")
            return

        # Deep-copy the original output block template
        new_head = copy.deepcopy(self._output_block_template)

        # Detect original output channel count from the template
        orig_out_ch = self._head_channels[list(self._head_channels.keys())[0]]

        if out_channels != orig_out_ch:
            # Patch Conv3d / ConvTranspose3d layers whose channel counts
            # match the original output channels
            conv_replacements = []
            for parent in new_head.modules():
                for attr_name, child in parent.named_children():
                    if isinstance(child, (nn.Conv3d, nn.ConvTranspose3d)):
                        new_in = child.in_channels
                        new_out = child.out_channels
                        if child.in_channels == orig_out_ch:
                            new_in = out_channels
                        if child.out_channels == orig_out_ch:
                            new_out = out_channels
                        if new_in != child.in_channels or new_out != child.out_channels:
                            kwargs = dict(
                                in_channels=new_in,
                                out_channels=new_out,
                                kernel_size=child.kernel_size,
                                stride=child.stride,
                                padding=child.padding,
                                dilation=child.dilation,
                                groups=child.groups,
                                bias=child.bias is not None,
                                padding_mode=child.padding_mode,
                            )
                            if isinstance(child, nn.ConvTranspose3d):
                                kwargs["output_padding"] = child.output_padding
                            new_conv = child.__class__(**kwargs)
                            conv_replacements.append((parent, attr_name, new_conv))
            for parent, attr_name, new_conv in conv_replacements:
                setattr(parent, attr_name, new_conv)

            # Also patch normalization layers that depend on channel count
            norm_replacements = []
            for parent in new_head.modules():
                for attr_name, child in parent.named_children():
                    if isinstance(child, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                        if child.num_features == orig_out_ch:
                            new_norm = child.__class__(
                                out_channels,
                                eps=child.eps,
                                momentum=child.momentum,
                                affine=child.affine,
                                track_running_stats=child.track_running_stats,
                            )
                            norm_replacements.append((parent, attr_name, new_norm))
                    elif isinstance(child, nn.GroupNorm):
                        if child.num_channels == orig_out_ch:
                            # Adjust num_groups if needed to evenly divide new channel count
                            num_groups = child.num_groups
                            while out_channels % num_groups != 0 and num_groups > 1:
                                num_groups -= 1
                            new_norm = nn.GroupNorm(
                                num_groups=num_groups,
                                num_channels=out_channels,
                                eps=child.eps,
                                affine=child.affine,
                            )
                            norm_replacements.append((parent, attr_name, new_norm))
            for parent, attr_name, new_norm in norm_replacements:
                setattr(parent, attr_name, new_norm)

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
