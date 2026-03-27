"""UNet teacher backend — extracted from the original Teacher class."""
from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import TeacherBackend

logger = logging.getLogger(__name__)


class UNetBackend(TeacherBackend):
    """Backend that wraps a MONAI UNet (or any nn.Module with the same API).

    Supports two modes:
    - **snapshot**: deepcopy a live model as frozen teacher
    - **checkpoint**: load weights from a ``.pt`` file into a model template
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._ckpt_hash: str | None = None
        self._source_mode: str = "none"
        self._model_id: str = ""
        self._use_features: bool = False
        self._feature_layers: List[str] = []
        self._feature_hooks: List[torch.utils.hooks.RemovableHook] = []
        self._features: Dict[str, torch.Tensor] = {}

    # -- TeacherBackend interface --

    def load(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        self._model_id = cfg.get("model_id", "")
        self._use_features = bool(cfg.get("use_features", False))
        self._feature_layers = list(cfg.get("feature_layers", []))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Teacher model is not initialised; call snapshot() first")
        return self._model(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._use_features:
            raise RuntimeError(
                "forward_features() requires use_features=True in teacher config"
            )
        _ = self.forward_logits(x)
        return dict(self._features)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self._model_id or (type(self._model).__name__ if self._model else ""),
            "ckpt_hash": self._ckpt_hash,
            "source_mode": self._source_mode,
            "frozen": (
                all(not p.requires_grad for p in self._model.parameters())
                if self._model is not None
                else False
            ),
            "use_features": self._use_features,
            "feature_layers": self._feature_layers,
        }

    def to(self, device) -> "UNetBackend":
        if self._model is not None:
            self._model.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self._model is not None:
            state["teacher_state_dict"] = self._model.state_dict()
        state["teacher_metadata"] = self.metadata
        return state

    def eval(self) -> "UNetBackend":
        if self._model is not None:
            self._model.eval()
        return self

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def is_external(self) -> bool:
        return False

    # -- UNet-specific --

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
        logger.debug("UNetBackend: snapshot created")

    def load_from_checkpoint(
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
        state = torch.load(path, map_location="cpu")
        sd = state.get("model_state_dict", state)
        self._model = copy.deepcopy(model_template)
        self._model.load_state_dict(sd)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._source_mode = "checkpoint"
        if self._use_features:
            self._register_hooks()
        logger.info(
            f"UNetBackend: loaded from checkpoint {path} (hash={self._ckpt_hash})"
        )

    def load_state_dict_from_saved(
        self, state: Dict[str, Any], model_template: nn.Module | None
    ) -> None:
        """Restore from a previously saved ``state_dict()``."""
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

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @property
    def features(self) -> Dict[str, torch.Tensor]:
        return self._features

    # -- internal helpers --

    @staticmethod
    def _compute_ckpt_hash(path: Path, nbytes: int = 4096) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(nbytes))
        return h.hexdigest()[:16]

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
                "UNetBackend: use_features=True but no layers matched feature_layers=%s. "
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
