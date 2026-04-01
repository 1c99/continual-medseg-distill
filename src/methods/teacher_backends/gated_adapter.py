"""Gated Residual Adapter with Prototype Support.

Combines three novel adapter mechanisms for continual foundation model distillation:

1. **Task-Residual Adapter (TRA)**: Frozen shared core + lightweight per-task
   residual projection. The core learns general feature→anatomy mapping on
   Task A and is frozen forever. New tasks only add a small 1x1 Conv3d.

2. **Confidence-Gated Distillation (CGAD)**: A learned gate network predicts
   per-voxel teacher reliability. KD loss is modulated by this gate — strong
   where the teacher is confident, suppressed where uncertain.

3. **Continual Prototype Adapter (CPA)**: Per-class prototypes (mean feature
   vectors) provide an auxiliary soft-label signal that is append-only and
   has zero forgetting by construction.

Usage:
    adapter = GatedResidualAdapter(in_channels=256, out_channels=15)
    # Pre-train on Task A data (trains core + gate + first residual)
    # After pre-training: call adapter.freeze_core()
    # For Task B: call adapter.add_task("taskB_muscles", out_channels=11)
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Conv3d block with skip connection and optional channel change."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.act = nn.GELU()

        # Skip connection with 1x1 conv if channels change
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)


def _build_core(in_channels: int, deep: bool = False) -> nn.Module:
    """Build the shared core with optional depth and skip connections."""
    if deep:
        return nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels // 2),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, in_channels),
            nn.GELU(),
        )


class GatedResidualAdapter(nn.Module):
    """Unified adapter combining TRA + CGAD + CPA for continual KD.

    Architecture:
        features (256) → [shared core (frozen after Task A)] → intermediate (256)
                           ├→ [task residual (per-task 1x1 conv)] → logits
                           └→ [gate head] → confidence gate ∈ [0, 1]

    Prototype bank stores per-class mean feature vectors for auxiliary
    soft-label generation.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 15,
        initial_task_id: str = "task_0",
        gate_hidden: int = 64,
        min_gate: float = 0.1,
        deep: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.min_gate = min_gate
        self._deep = deep

        # ---- TRA: Shared core (frozen after first task) ----
        self.core = _build_core(in_channels, deep=deep)
        self._core_frozen = False

        # Core output channels (deep mode halves channels in last block)
        self._core_out = in_channels // 2 if deep else in_channels

        # ---- TRA: Per-task residual projections ----
        self.residuals = nn.ModuleDict({
            initial_task_id: nn.Conv3d(self._core_out, out_channels, kernel_size=1),
        })
        self.current_task: str = initial_task_id
        self._task_channels: Dict[str, int] = {initial_task_id: out_channels}

        # ---- CGAD: Confidence gate ----
        self.gate_head = nn.Sequential(
            nn.Conv3d(self._core_out, gate_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(gate_hidden, 1, kernel_size=1),
        )

        logger.info(
            f"GatedResidualAdapter: {'deep' if deep else 'shallow'} core, "
            f"core_out={self._core_out}, gate_hidden={gate_hidden}"
        )

        # ---- CPA: Prototype bank ----
        # Maps class_key → (256,) prototype vector
        self._prototypes: Dict[str, torch.Tensor] = {}
        self._prototype_counts: Dict[str, int] = {}

    # ---- Forward ----

    def forward(
        self, features: torch.Tensor, target_shape: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, gate).

        Args:
            features: Backbone features ``(B, 256, D', H', W')``.
            target_shape: Target spatial shape ``(D, H, W)`` for interpolation.

        Returns:
            logits: Class logits ``(B, C, D, H, W)``.
            gate: Per-voxel confidence ``(B, 1, D, H, W)`` in ``[min_gate, 1]``.
        """
        intermediate = self.core(features)

        # Task-specific logits
        logits = self.residuals[self.current_task](intermediate)
        if logits.shape[2:] != target_shape:
            logits = F.interpolate(
                logits, size=target_shape, mode="trilinear", align_corners=False
            )

        # Confidence gate
        gate_raw = self.gate_head(intermediate)
        if gate_raw.shape[2:] != target_shape:
            gate_raw = F.interpolate(
                gate_raw, size=target_shape, mode="trilinear", align_corners=False
            )
        gate = self.min_gate + (1.0 - self.min_gate) * torch.sigmoid(gate_raw)

        return logits, gate

    def forward_logits_only(
        self, features: torch.Tensor, target_shape: tuple
    ) -> torch.Tensor:
        """Forward returning only logits (for compatibility with _OutputAdapter)."""
        logits, _ = self.forward(features, target_shape)
        return logits

    # ---- TRA: Core freezing and task management ----

    def freeze_core(self) -> None:
        """Freeze the shared core after first task pre-training."""
        for p in self.core.parameters():
            p.requires_grad = False
        self._core_frozen = True
        logger.info("GatedResidualAdapter: core frozen")

    def add_task(self, task_id: str, out_channels: int) -> None:
        """Add a new task-specific residual projection."""
        if task_id in self.residuals:
            return
        self.residuals[task_id] = nn.Conv3d(
            self._core_out, out_channels, kernel_size=1
        )
        self._task_channels[task_id] = out_channels
        # Move to same device as core
        device = next(self.core.parameters()).device
        self.residuals[task_id].to(device)
        logger.info(
            f"GatedResidualAdapter: added task '{task_id}' "
            f"({out_channels} ch, core={'frozen' if self._core_frozen else 'trainable'})"
        )

    # ---- CPA: Prototype management ----

    @torch.no_grad()
    def update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: str,
        num_classes: int,
    ) -> None:
        """Accumulate prototype statistics from a batch.

        Call this during adapter pre-training to build prototypes incrementally.

        Args:
            features: Backbone features ``(B, 256, D, H, W)``.
            labels: Ground truth ``(B, D, H, W)`` with class indices.
        """
        B, C, D, H, W = features.shape

        # Resize labels to match feature spatial dims if needed
        if labels.shape[1:] != features.shape[2:]:
            labels = F.interpolate(
                labels.float().unsqueeze(1),
                size=features.shape[2:],
                mode="nearest",
            ).squeeze(1).long()

        for cls in range(1, num_classes):  # skip background
            key = f"{task_id}_cls{cls}"
            mask = (labels == cls)  # (B, D, H, W)
            if mask.sum() == 0:
                continue

            # Extract features at this class's voxels
            mask_expanded = mask.unsqueeze(1).expand_as(features)  # (B, C, D, H, W)
            cls_features = features[mask_expanded].view(C, -1).mean(dim=1)  # (C,)

            count = int(mask.sum())
            if key in self._prototypes:
                # Running mean update (always on CPU)
                old_count = self._prototype_counts[key]
                new_count = old_count + count
                self._prototypes[key] = (
                    self._prototypes[key] * old_count + cls_features.cpu() * count
                ) / new_count
                self._prototype_counts[key] = new_count
            else:
                self._prototypes[key] = cls_features.cpu()
                self._prototype_counts[key] = count

    def prototype_logits(
        self,
        features: torch.Tensor,
        task_id: str,
        num_classes: int,
        temperature: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """Compute soft labels from prototypes via cosine similarity.

        Returns None if no prototypes exist for this task.
        """
        task_protos = {
            k: v for k, v in self._prototypes.items()
            if k.startswith(f"{task_id}_")
        }
        if not task_protos:
            return None

        B, C, D, H, W = features.shape
        device = features.device

        # Normalize features for cosine similarity
        feat_norm = F.normalize(features, dim=1)  # (B, 256, D, H, W)

        logits = torch.zeros(B, num_classes, D, H, W, device=device)
        for cls in range(1, num_classes):
            key = f"{task_id}_cls{cls}"
            if key not in self._prototypes:
                continue
            proto = self._prototypes[key].to(device)
            proto_norm = F.normalize(proto, dim=0)
            # Cosine similarity: dot product of normalized vectors
            sim = (feat_norm * proto_norm.view(1, C, 1, 1, 1)).sum(dim=1)  # (B, D, H, W)
            logits[:, cls] = sim

        return logits / temperature

    @property
    def num_prototypes(self) -> int:
        return len(self._prototypes)

    # ---- State management ----

    def state_dict_full(self) -> Dict[str, any]:
        """Save full state including prototypes."""
        state = self.state_dict()
        state["_prototypes"] = self._prototypes
        state["_prototype_counts"] = self._prototype_counts
        state["_task_channels"] = self._task_channels
        state["_core_frozen"] = self._core_frozen
        return state

    def load_state_dict_full(self, state: Dict[str, any]) -> None:
        """Load full state including prototypes."""
        self._prototypes = state.pop("_prototypes", {})
        self._prototype_counts = state.pop("_prototype_counts", {})
        self._task_channels = state.pop("_task_channels", {})
        was_frozen = state.pop("_core_frozen", False)
        self.load_state_dict(state)
        if was_frozen:
            self.freeze_core()
