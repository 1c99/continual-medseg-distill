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

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Teacher:
    """Frozen teacher wrapper with optional feature extraction.

    Config keys (under ``method.kd.teacher``):
        type: snapshot | checkpoint  (default: snapshot)
        ckpt_path: path to .pt file (required when type=checkpoint)
        model_id: str  (optional, for provenance tracking)
        use_features: bool  (default: false)
        feature_layers: list of layer name prefixes to capture
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        teacher_cfg: Dict[str, Any] | None = None,
        model_template: nn.Module | None = None,
    ):
        self._cfg = teacher_cfg or {}
        self._model: nn.Module | None = None
        self._feature_hooks: List[torch.utils.hooks.RemovableHook] = []
        self._features: Dict[str, torch.Tensor] = {}
        self._use_features = bool(self._cfg.get("use_features", False))
        self._feature_layers: List[str] = list(self._cfg.get("feature_layers", []))
        self._ckpt_hash: str | None = None
        self._source_mode: str = "none"  # none | snapshot | checkpoint
        self._model_id: str = self._cfg.get("model_id", "")

        teacher_type = self._cfg.get("type", "snapshot")

        if teacher_type == "checkpoint":
            ckpt_path = self._cfg.get("ckpt_path")
            if not ckpt_path:
                raise ValueError(
                    "method.kd.teacher.ckpt_path is required when teacher.type=checkpoint"
                )
            self._load_from_checkpoint(ckpt_path, model_template)
        elif model is not None:
            self.snapshot(model)

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def features(self) -> Dict[str, torch.Tensor]:
        """Captured intermediate features from last forward pass."""
        return self._features

    @property
    def metadata(self) -> Dict[str, Any]:
        """Teacher provenance metadata for reproducibility tracking."""
        return {
            "model_id": self._model_id or type(self._model).__name__ if self._model else "",
            "ckpt_hash": self._ckpt_hash,
            "source_mode": self._source_mode,
            "frozen": self.has_model and all(
                not p.requires_grad for p in self._model.parameters()
            ) if self.has_model else False,
            "use_features": self._use_features,
            "feature_layers": self._feature_layers,
        }

    def snapshot(self, model: nn.Module) -> None:
        """Deepcopy *model* as the frozen teacher."""
        self._remove_hooks()
        self._model = copy.deepcopy(model).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._source_mode = "snapshot"
        self._ckpt_hash = None
        if self._use_features:
            self._register_hooks()
        logger.debug("Teacher: snapshot created")

    @staticmethod
    def _compute_ckpt_hash(path: Path, nbytes: int = 4096) -> str:
        """SHA256 of first *nbytes* of checkpoint file for lineage tracking."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(nbytes))
        return h.hexdigest()[:16]

    def _load_from_checkpoint(
        self, ckpt_path: str, model_template: nn.Module | None
    ) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {path}. "
                "Provide a valid method.kd.teacher.ckpt_path."
            )
        if model_template is None:
            raise ValueError(
                "model_template is required to load teacher from checkpoint"
            )
        self._ckpt_hash = self._compute_ckpt_hash(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        self._model = copy.deepcopy(model_template)
        self._model.load_state_dict(sd)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._source_mode = "checkpoint"
        if self._use_features:
            self._register_hooks()
        logger.info(f"Teacher: loaded from checkpoint {path} (hash={self._ckpt_hash})")

    # ---- feature hooks ----

    def _register_hooks(self) -> None:
        if self._model is None:
            return
        self._remove_hooks()
        self._features = {}

        for name, module in self._model.named_modules():
            if self._should_hook(name):
                hook = module.register_forward_hook(self._make_hook(name))
                self._feature_hooks.append(hook)

        if not self._feature_hooks:
            logger.warning(
                "Teacher: use_features=True but no layers matched feature_layers=%s. "
                "Available layers: %s",
                self._feature_layers,
                [n for n, _ in self._model.named_modules() if n][:20],
            )

    def _should_hook(self, layer_name: str) -> bool:
        if not self._feature_layers:
            return False
        return any(layer_name.startswith(prefix) for prefix in self._feature_layers)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            self._features[name] = output.detach()
        return hook_fn

    def _remove_hooks(self) -> None:
        for h in self._feature_hooks:
            h.remove()
        self._feature_hooks.clear()
        self._features.clear()

    # ---- forward ----

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run teacher forward pass and return logits.

        If ``use_features=True``, captured features are available via
        ``self.features`` after this call.
        """
        if self._model is None:
            raise RuntimeError("Teacher model is not initialised; call snapshot() first")
        return self._model(x)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized interface: return output logits."""
        return self.forward(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standardized interface: run forward pass and return captured features.

        Requires ``use_features=True`` and ``feature_layers`` to be set.
        Returns dict mapping layer names to their output tensors.
        """
        if not self._use_features:
            raise RuntimeError(
                "forward_features() requires use_features=True in teacher config"
            )
        _ = self.forward(x)
        return dict(self._features)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # ---- state persistence ----

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self._model is not None:
            state["teacher_state_dict"] = self._model.state_dict()
        state["teacher_metadata"] = self.metadata
        return state

    def load_state_dict(
        self, state: Dict[str, Any], model_template: nn.Module | None = None
    ) -> None:
        if "teacher_state_dict" not in state:
            return
        if model_template is None:
            raise ValueError("model_template required to restore teacher state")
        self._remove_hooks()
        self._model = copy.deepcopy(model_template).eval()
        self._model.load_state_dict(state["teacher_state_dict"])
        for p in self._model.parameters():
            p.requires_grad = False
        if self._use_features:
            self._register_hooks()

    def to(self, device) -> "Teacher":
        if self._model is not None:
            self._model.to(device)
        return self
