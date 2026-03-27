"""SAM3 teacher backend for knowledge distillation.

Wraps the facebookresearch/sam3 model to produce dense segmentation
logits compatible with the ``(B, num_classes, D, H, W)`` format expected
by the distillation pipeline.

Requires:
- ``third_party/sam3/`` to be cloned (see ``scripts/setup_external.sh``)
- A SAM3 checkpoint file
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TeacherBackend

logger = logging.getLogger(__name__)

# Path to the vendored SAM3 repository
_SAM3_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "sam3"


class _OutputAdapter(nn.Module):
    """Projects SAM3 backbone features to dense segmentation logits.

    SAM3 is a detection/segmentation model that produces per-query masks.
    For KD we need dense ``(B, C_out, D, H, W)`` logits, so we extract
    intermediate backbone features and project them with a lightweight head.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 14):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, in_channels),
            nn.GELU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Project features and interpolate to *target_shape* ``(D, H, W)``."""
        out = self.proj(features)
        if out.shape[2:] != target_shape:
            out = F.interpolate(out, size=target_shape, mode="trilinear", align_corners=False)
        return out


class SAM3Backend(TeacherBackend):
    """Teacher backend that wraps a SAM3 image model.

    Config keys (under ``method.kd.teacher``):
        type: sam3
        ckpt_path: path to SAM3 checkpoint
        output_channels: int — number of output classes (must match student)
        model_id: str — provenance tag (optional)
        model_variant: str — reserved for future use (default: vit_h)
        adapter_channels: int — hidden dim for output adapter (default: 256)
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._adapter: _OutputAdapter | None = None
        self._cfg: Dict[str, Any] = {}
        self._ckpt_hash: str | None = None
        self._model_id: str = ""
        self._device: str = "cpu"

    def load(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        self._cfg = cfg
        self._model_id = cfg.get("model_id", "sam3")
        self._device = device

        ckpt_path = cfg.get("ckpt_path")
        output_channels = cfg.get("output_channels")

        # Validate early — actual model instantiation is deferred to first use
        # or explicit load call with a checkpoint path
        if ckpt_path:
            self._ensure_sam3_available()
            self._load_model(ckpt_path, output_channels or 14, device)

    def _ensure_sam3_available(self) -> None:
        if not _SAM3_ROOT.exists():
            raise RuntimeError(
                f"SAM3 repository not found at {_SAM3_ROOT}. "
                "Run: bash scripts/setup_external.sh"
            )
        # Add to sys.path so sam3 can be imported
        sam3_str = str(_SAM3_ROOT)
        if sam3_str not in sys.path:
            sys.path.insert(0, sam3_str)

    def _load_model(
        self, ckpt_path: str, output_channels: int, device: str = "cpu"
    ) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {path}. "
                "Provide a valid method.kd.teacher.ckpt_path."
            )

        self._ckpt_hash = self._compute_ckpt_hash(path)

        from sam3 import build_sam3_image_model

        logger.info(f"SAM3Backend: loading model from {path}")
        self._model = build_sam3_image_model(
            checkpoint_path=str(path),
            device=device,
            eval_mode=True,
            load_from_HF=False,
            enable_segmentation=True,
        )

        # Freeze all SAM3 parameters
        for p in self._model.parameters():
            p.requires_grad = False

        adapter_channels = self._cfg.get("adapter_channels", 256)
        self._adapter = _OutputAdapter(
            in_channels=adapter_channels,
            out_channels=output_channels,
        ).to(device)

        logger.info(
            f"SAM3Backend: loaded (hash={self._ckpt_hash}, "
            f"output_channels={output_channels})"
        )

    @staticmethod
    def _compute_ckpt_hash(path: Path, nbytes: int = 4096) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(nbytes))
        return h.hexdigest()[:16]

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None or self._adapter is None:
            raise RuntimeError(
                "SAM3Backend: model not loaded. Provide ckpt_path in config."
            )
        # x: (B, C_in, D, H, W) — 3D medical volume
        # Extract backbone features from SAM3's visual encoder
        # SAM3 expects 2D images; we process slice-by-slice and aggregate
        B, C_in, D, H, W = x.shape

        # Process each depth slice through SAM3's backbone
        features_3d = self._extract_features_3d(x)

        # Project to segmentation logits
        target_shape = (D, H, W)
        return self._adapter(features_3d, target_shape)

    def _extract_features_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract SAM3 backbone features from a 3D volume.

        Processes depth slices through the visual encoder and stacks
        them into a 3D feature volume.
        """
        B, C_in, D, H, W = x.shape

        # Get the visual backbone (neck + ViT)
        backbone = self._model.backbone

        slice_features = []
        for d in range(D):
            # Extract 2D slice: (B, C_in, H, W)
            img_slice = x[:, :, d, :, :]
            # SAM3 expects 3-channel input; repeat if needed
            if img_slice.shape[1] == 1:
                img_slice = img_slice.repeat(1, 3, 1, 1)
            elif img_slice.shape[1] != 3:
                img_slice = img_slice[:, :3, :, :]

            # Forward through visual backbone
            with torch.no_grad():
                vis_out = backbone.visual(img_slice)
                # vis_out contains feature maps at different scales
                # Use the highest resolution feature map
                if isinstance(vis_out, dict):
                    feat = vis_out.get("feature_maps", [vis_out.get("backbone_fpn", [None])[0]])[0]
                elif isinstance(vis_out, (list, tuple)):
                    feat = vis_out[0] if len(vis_out) > 0 else vis_out
                else:
                    feat = vis_out

            slice_features.append(feat)

        # Stack along depth: (B, C_feat, D, H_feat, W_feat)
        features_3d = torch.stack(slice_features, dim=2)
        return features_3d

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("SAM3Backend: model not loaded")
        features_3d = self._extract_features_3d(x)
        return {"backbone_features": features_3d}

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self._model_id,
            "ckpt_hash": self._ckpt_hash,
            "source_mode": "sam3",
            "frozen": True,
            "use_features": False,
            "feature_layers": [],
        }

    def to(self, device) -> "SAM3Backend":
        self._device = str(device)
        if self._model is not None:
            self._model.to(device)
        if self._adapter is not None:
            self._adapter.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"teacher_metadata": self.metadata}
        if self._adapter is not None:
            state["adapter_state_dict"] = self._adapter.state_dict()
        return state

    def eval(self) -> "SAM3Backend":
        if self._model is not None:
            self._model.eval()
        if self._adapter is not None:
            self._adapter.eval()
        return self

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def is_external(self) -> bool:
        return True
