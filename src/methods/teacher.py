"""Teacher model abstraction for knowledge distillation.

Supports two modes:
- ``snapshot``: deepcopy current student as frozen teacher (default, existing behaviour)
- ``checkpoint``: load a pre-trained teacher from a checkpoint file

Feature extraction is optionally available via hook-based intermediate
representation capture for feature-level distillation.

Standardized interface:
- ``forward_logits(x)`` — returns output logits
- ``forward_features(x)`` — returns intermediate feature dict (requires use_features)
- ``metadata`` — teacher provenance: model_id, ckpt_hash, mode, frozen status
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .teacher_backends import UNetBackend, create_backend

logger = logging.getLogger(__name__)


class Teacher:
    """Frozen teacher wrapper with optional feature extraction.

    Config keys (under ``method.kd.teacher``):
        type: snapshot | checkpoint | sam3 | medsam3  (default: snapshot)
        ckpt_path: path to .pt file (required when type=checkpoint/sam3/medsam3)
        model_id: str  (optional, for provenance tracking)
        use_features: bool  (default: false)
        feature_layers: list of layer name prefixes to capture
        output_channels: int  (required for sam3/medsam3)
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        teacher_cfg: Dict[str, Any] | None = None,
        model_template: nn.Module | None = None,
        global_cfg: Dict[str, Any] | None = None,
    ):
        self._cfg = teacher_cfg or {}
        teacher_type = self._cfg.get("type", "snapshot")

        # Create the appropriate backend
        self._backend = create_backend(self._cfg)

        if teacher_type == "checkpoint":
            ckpt_path = self._cfg.get("ckpt_path")
            if not ckpt_path:
                raise ValueError(
                    "method.kd.teacher.ckpt_path is required when teacher.type=checkpoint"
                )
            # Build model_template from global config if not provided
            if model_template is None and global_cfg is not None:
                from src.models.factory import build_model
                model_template = build_model(global_cfg)
            assert isinstance(self._backend, UNetBackend)
            self._backend.load_from_checkpoint(ckpt_path, model_template)
        elif teacher_type in {"sam3", "medsam3"}:
            # External backends handle their own loading in load()
            pass
        elif model is not None:
            self.snapshot(model)

    @property
    def model(self) -> nn.Module | None:
        if isinstance(self._backend, UNetBackend):
            return self._backend.model
        return None

    @property
    def has_model(self) -> bool:
        return self._backend.has_model

    @property
    def is_external(self) -> bool:
        """True for external (non-UNet) backends like SAM3/MedSAM3."""
        return self._backend.is_external

    @property
    def features(self) -> Dict[str, torch.Tensor]:
        """Captured intermediate features from last forward pass."""
        if isinstance(self._backend, UNetBackend):
            return self._backend.features
        return {}

    @property
    def _feature_layers(self) -> list:
        """Expose backend feature_layers for backward compat with distill.py."""
        if isinstance(self._backend, UNetBackend):
            return self._backend._feature_layers
        return []

    @property
    def _use_features(self) -> bool:
        """Expose backend use_features for backward compat with distill.py."""
        if isinstance(self._backend, UNetBackend):
            return self._backend._use_features
        return False

    @property
    def metadata(self) -> Dict[str, Any]:
        """Teacher provenance metadata for reproducibility tracking."""
        return self._backend.metadata

    def snapshot(self, model: nn.Module) -> None:
        """Deepcopy *model* as the frozen teacher."""
        self._backend.snapshot(model)

    def reconfigure_adapter(self, out_channels: int, task_id: str | None = None) -> None:
        """Reconfigure external teacher's output adapter for a new task.

        No-op for snapshot/UNet backends.
        """
        self._backend.reconfigure_adapter(out_channels, task_id=task_id)

    @property
    def backend(self):
        """Direct access to the underlying backend (for replay KD head switching)."""
        return self._backend

    def switch_adapter_task(self, task_id: str) -> str | None:
        """Switch the gated adapter to a specific task, returning the previous task ID.

        Returns None if the backend does not support task switching.
        """
        backend = self._backend
        if not (hasattr(backend, "_gated") and backend._gated):
            return None
        adapter = getattr(backend, "_adapter", None)
        if adapter is None or not hasattr(adapter, "current_task"):
            return None
        if task_id not in adapter.residuals:
            return None
        prev = adapter.current_task
        adapter.current_task = task_id
        return prev

    def restore_adapter_task(self, task_id: str | None) -> None:
        """Restore the gated adapter to a previous task ID."""
        if task_id is None:
            return
        backend = self._backend
        adapter = getattr(backend, "_adapter", None)
        if adapter is not None and hasattr(adapter, "current_task"):
            adapter.current_task = task_id

    def extract_features(self, x: torch.Tensor) -> torch.Tensor | None:
        """Extract backbone features (for prototype KD).

        Returns the 3D feature volume or None if not supported.
        """
        if hasattr(self._backend, "_extract_features_3d"):
            with torch.no_grad():
                return self._backend._extract_features_3d(x)
        return None

    def get_prototype_logits(
        self,
        features: torch.Tensor,
        task_id: str,
        num_classes: int,
        temperature: float = 0.1,
    ) -> torch.Tensor | None:
        """Compute prototype soft labels from stored CPA prototypes.

        Returns None if the adapter has no prototypes for this task.
        """
        adapter = getattr(self._backend, "_adapter", None)
        if adapter is None or not hasattr(adapter, "prototype_logits"):
            return None
        return adapter.prototype_logits(features, task_id, num_classes, temperature)

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (logits, gate). Gate is None for non-gated backends."""
        if hasattr(self._backend, "forward_with_gate"):
            with torch.no_grad():
                return self._backend.forward_with_gate(x)
        return self.forward(x), None

    # ---- forward ----

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run teacher forward pass and return logits.

        If ``use_features=True``, captured features are available via
        ``self.features`` after this call.
        """
        return self._backend.forward_logits(x)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized interface: return output logits."""
        return self.forward(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standardized interface: run forward pass and return captured features.

        Requires ``use_features=True`` and ``feature_layers`` to be set.
        Returns dict mapping layer names to their output tensors.
        """
        return self._backend.forward_features(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # ---- state persistence ----

    def state_dict(self) -> Dict[str, Any]:
        return self._backend.state_dict()

    def load_state_dict(
        self, state: Dict[str, Any], model_template: nn.Module | None = None
    ) -> None:
        if isinstance(self._backend, UNetBackend):
            self._backend.load_state_dict_from_saved(state, model_template)

    def to(self, device) -> "Teacher":
        self._backend.to(device)
        return self
