"""Slice-wise 2D adapters for foundation model teacher backends.

Processes each depth slice independently with 2D convolutions at the
backbone's native feature resolution, avoiding the resolution loss caused
by 3D stacking and interpolation in the original 3D adapter.

Two variants:
- ``SliceWiseAdapter``: Standard 2D conv projection (drop-in replacement
  for ``_OutputAdapter`` in MedSAM2/MedSAM3 backends).
- ``SliceWiseGRACEAdapter``: GRACE-style adapter with frozen shared core,
  per-task 2D residuals, confidence gate, and CPA prototype bank.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard slice-wise adapter
# ---------------------------------------------------------------------------

class SliceWiseAdapter(nn.Module):
    """Processes backbone features slice-by-slice with 2D convolutions.

    Preserves the backbone's native spatial resolution per-slice,
    only interpolating at the final output stage.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 15,
        deep: bool = False,
    ):
        super().__init__()
        if deep:
            mid = in_channels
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, mid, 3, padding=1),
                nn.GroupNorm(16, mid),
                nn.GELU(),
                nn.Conv2d(mid, mid, 3, padding=1),
                nn.GroupNorm(16, mid),
                nn.GELU(),
                nn.Conv2d(mid, mid // 2, 3, padding=1),
                nn.GroupNorm(8, mid // 2),
                nn.GELU(),
                nn.Conv2d(mid // 2, out_channels, 1),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(16, in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, 1),
            )

    def forward(
        self, features_3d: torch.Tensor, target_shape: tuple
    ) -> torch.Tensor:
        """
        Args:
            features_3d: ``(B, C, D, fH, fW)`` — stacked 2D backbone features.
            target_shape: ``(D, H, W)`` — target spatial dimensions.

        Returns:
            Logits ``(B, out_ch, D, H, W)``.
        """
        B, C, D, fH, fW = features_3d.shape
        target_D, target_H, target_W = target_shape

        slice_logits = []
        for d in range(D):
            feat_2d = features_3d[:, :, d, :, :]  # (B, C, fH, fW)
            logit_2d = self.proj(feat_2d)  # (B, out_ch, fH, fW)
            if logit_2d.shape[-2:] != (target_H, target_W):
                logit_2d = F.interpolate(
                    logit_2d,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )
            slice_logits.append(logit_2d)

        return torch.stack(slice_logits, dim=2)  # (B, out_ch, D, H, W)


# ---------------------------------------------------------------------------
# 2D residual block for the GRACE core
# ---------------------------------------------------------------------------

class _ResidualBlock2d(nn.Module):
    """Conv2d block with skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.act = nn.GELU()
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)


def _build_core_2d(in_channels: int, deep: bool = False) -> nn.Module:
    """Build the shared 2D core."""
    if deep:
        return nn.Sequential(
            _ResidualBlock2d(in_channels, in_channels),
            _ResidualBlock2d(in_channels, in_channels),
            _ResidualBlock2d(in_channels, in_channels // 2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(16, in_channels),
            nn.GELU(),
        )


# ---------------------------------------------------------------------------
# Slice-wise GRACE adapter
# ---------------------------------------------------------------------------

class SliceWiseGRACEAdapter(nn.Module):
    """2D slice-wise GRACE adapter: TRA + CGAD + CPA.

    Architecture (per-slice):
        features (C) -> [shared 2D core (frozen after Task A)] -> intermediate
                          |-> [task residual (per-task 2D 1x1)] -> logits
                          |-> [gate head (2D)] -> confidence gate

    Prototype bank operates in the same feature space as the 3D variant.
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

        # ---- TRA: Shared 2D core (frozen after first task) ----
        self.core = _build_core_2d(in_channels, deep=deep)
        self._core_frozen = False
        self._core_out = in_channels // 2 if deep else in_channels

        # ---- TRA: Per-task residual projections (2D 1x1) ----
        self.residuals = nn.ModuleDict({
            initial_task_id: nn.Conv2d(self._core_out, out_channels, 1),
        })
        self.current_task: str = initial_task_id
        self._task_channels: Dict[str, int] = {initial_task_id: out_channels}

        # ---- CGAD: Confidence gate (2D) ----
        self.gate_head = nn.Sequential(
            nn.Conv2d(self._core_out, gate_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 1, 1),
        )

        # ---- CPA: Prototype bank (same as 3D variant) ----
        self._prototypes: Dict[str, torch.Tensor] = {}
        self._prototype_counts: Dict[str, int] = {}

        logger.info(
            f"SliceWiseGRACEAdapter: {'deep' if deep else 'shallow'} 2D core, "
            f"core_out={self._core_out}, gate_hidden={gate_hidden}"
        )

    # ---- Forward ----

    def forward(
        self, features_3d: torch.Tensor, target_shape: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, gate).

        Args:
            features_3d: ``(B, C, D, fH, fW)`` backbone features.
            target_shape: ``(D, H, W)`` target spatial dimensions.

        Returns:
            logits: ``(B, out_ch, D, H, W)``
            gate: ``(B, 1, D, H, W)`` in ``[min_gate, 1]``
        """
        B, C, D, fH, fW = features_3d.shape
        target_D, target_H, target_W = target_shape

        all_logits = []
        all_gates = []

        for d in range(D):
            feat_2d = features_3d[:, :, d, :, :]  # (B, C, fH, fW)
            intermediate = self.core(feat_2d)

            # Task-specific logits
            logit_2d = self.residuals[self.current_task](intermediate)
            if logit_2d.shape[-2:] != (target_H, target_W):
                logit_2d = F.interpolate(
                    logit_2d,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )

            # Confidence gate
            gate_raw = self.gate_head(intermediate)
            if gate_raw.shape[-2:] != (target_H, target_W):
                gate_raw = F.interpolate(
                    gate_raw,
                    size=(target_H, target_W),
                    mode="bilinear",
                    align_corners=False,
                )
            gate_2d = self.min_gate + (1.0 - self.min_gate) * torch.sigmoid(
                gate_raw
            )

            all_logits.append(logit_2d)
            all_gates.append(gate_2d)

        logits = torch.stack(all_logits, dim=2)  # (B, out_ch, D, H, W)
        gate = torch.stack(all_gates, dim=2)  # (B, 1, D, H, W)
        return logits, gate

    def forward_logits_only(
        self, features_3d: torch.Tensor, target_shape: tuple
    ) -> torch.Tensor:
        """Forward returning only logits."""
        logits, _ = self.forward(features_3d, target_shape)
        return logits

    # ---- TRA: Core freezing and task management ----

    def freeze_core(self) -> None:
        """Freeze the shared core after first task pre-training."""
        for p in self.core.parameters():
            p.requires_grad = False
        self._core_frozen = True
        logger.info("SliceWiseGRACEAdapter: core frozen")

    def add_task(self, task_id: str, out_channels: int) -> None:
        """Add a new task-specific residual projection."""
        if task_id in self.residuals:
            return
        self.residuals[task_id] = nn.Conv2d(self._core_out, out_channels, 1)
        self._task_channels[task_id] = out_channels
        device = next(self.core.parameters()).device
        self.residuals[task_id].to(device)
        logger.info(
            f"SliceWiseGRACEAdapter: added task '{task_id}' "
            f"({out_channels} ch, core={'frozen' if self._core_frozen else 'trainable'})"
        )

    # ---- CPA: Prototype management (same interface as 3D variant) ----

    @torch.no_grad()
    def update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: str,
        num_classes: int,
    ) -> None:
        """Accumulate prototype statistics from a batch.

        Args:
            features: Backbone features ``(B, C, D, H, W)``.
            labels: Ground truth ``(B, D, H, W)`` with class indices.
        """
        B, C, D, H, W = features.shape

        if labels.shape[1:] != features.shape[2:]:
            labels = F.interpolate(
                labels.float().unsqueeze(1),
                size=features.shape[2:],
                mode="nearest",
            ).squeeze(1).long()

        for cls in range(1, num_classes):
            key = f"{task_id}_cls{cls}"
            mask = labels == cls
            if mask.sum() == 0:
                continue
            mask_expanded = mask.unsqueeze(1).expand_as(features)
            cls_features = features[mask_expanded].view(C, -1).mean(dim=1)
            count = int(mask.sum())
            if key in self._prototypes:
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
        """Compute soft labels from prototypes via cosine similarity."""
        task_protos = {
            k: v
            for k, v in self._prototypes.items()
            if k.startswith(f"{task_id}_")
        }
        if not task_protos:
            return None

        B, C, D, H, W = features.shape
        device = features.device
        feat_norm = F.normalize(features, dim=1)
        logits = torch.zeros(B, num_classes, D, H, W, device=device)
        for cls in range(1, num_classes):
            key = f"{task_id}_cls{cls}"
            if key not in self._prototypes:
                continue
            proto = self._prototypes[key].to(device)
            proto_norm = F.normalize(proto, dim=0)
            sim = (feat_norm * proto_norm.view(1, C, 1, 1, 1)).sum(dim=1)
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
