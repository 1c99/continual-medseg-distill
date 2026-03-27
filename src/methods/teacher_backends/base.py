"""Abstract base class for teacher backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class TeacherBackend(ABC):
    """Interface that all teacher backends must implement.

    A backend encapsulates model loading, forward passes, and state
    management for a specific teacher architecture (UNet, SAM3, MedSAM3, etc.).
    """

    @abstractmethod
    def load(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        """Load the model from config (checkpoint path, model variant, etc.)."""

    @abstractmethod
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return output logits with shape ``(B, C_out, D, H, W)``."""

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return intermediate feature dict from a forward pass."""

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Teacher provenance info (model_id, ckpt_hash, etc.)."""

    @abstractmethod
    def to(self, device) -> "TeacherBackend":
        """Move model to *device*."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable state."""

    @abstractmethod
    def eval(self) -> "TeacherBackend":
        """Set model to eval mode."""

    def snapshot(self, model: nn.Module) -> None:
        """Create a frozen copy of *model* as the teacher.

        Only meaningful for UNet-style backends. External backends
        (SAM3, MedSAM3) raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support snapshot()"
        )

    @property
    def has_model(self) -> bool:
        """Return True if a model is loaded and ready for inference."""
        return False

    @property
    def is_external(self) -> bool:
        """Return True for external (non-UNet) backends like SAM3/MedSAM3."""
        return False
