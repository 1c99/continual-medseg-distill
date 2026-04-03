"""MedSAM2 teacher backend for knowledge distillation.

Wraps the MedSAM2 model (SAM2 fine-tuned for 3D medical segmentation by
bowang-lab) to produce dense segmentation logits compatible with the
``(B, num_classes, D, H, W)`` format expected by the distillation pipeline.

MedSAM2 uses a Hiera image encoder (SAM2 backbone). For KD we extract
image encoder features slice-by-slice from 3D volumes, stack them into a
3D feature volume, and project through a lightweight 3D conv adapter head
to produce per-voxel logits.

Requires:
- ``third_party/medsam2/`` to be cloned (bowang-lab/MedSAM2)
- The MedSAM2 checkpoint (``checkpoints/MedSAM2_latest.pt``)
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TeacherBackend

logger = logging.getLogger(__name__)

_MEDSAM2_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "medsam2"

# MedSAM2 default image size (from sam2.1_hiera_t512.yaml)
_MEDSAM2_IMG_SIZE = 512

# ImageNet normalization constants used by SAM2
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _OutputAdapter(nn.Module):
    """Projects SAM2 backbone features to dense segmentation logits.

    Uses the same ResidualBlock architecture as GatedResidualAdapter
    when deep=True, ensuring fair comparison (only difference between
    standard and GRACE is freeze-vs-rebuild, not architecture).
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 14, deep: bool = False):
        super().__init__()
        if deep:
            # Same ResidualBlock architecture as GRACE deep core + output projection
            from .gated_adapter import ResidualBlock
            self.proj = nn.Sequential(
                ResidualBlock(in_channels, in_channels),
                ResidualBlock(in_channels, in_channels),
                ResidualBlock(in_channels, in_channels // 2),
                nn.Conv3d(in_channels // 2, out_channels, kernel_size=1),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.GroupNorm(16, in_channels),
                nn.GELU(),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )

    def forward(self, features: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        out = self.proj(features)
        if out.shape[2:] != target_shape:
            out = F.interpolate(
                out, size=target_shape, mode="trilinear", align_corners=False
            )
        return out


class MedSAM2Backend(TeacherBackend):
    """Teacher backend wrapping bowang-lab/MedSAM2.

    Extracts Hiera image encoder features slice-by-slice from 3D volumes
    and projects them to dense segmentation logits via a lightweight adapter.

    Config keys (under ``method.kd.teacher``):
        type: medsam2
        ckpt_path: path to MedSAM2 checkpoint (default: third_party/medsam2/checkpoints/MedSAM2_latest.pt)
        config_file: Hydra config name (default: configs/sam2.1_hiera_t512.yaml)
        output_channels: int — number of output classes (must match student)
        adapter_channels: int — hidden dim for output adapter (default: 256)
        model_id: str — provenance tag (optional)
    """

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._adapter: _OutputAdapter | None = None
        self._cfg: Dict[str, Any] = {}
        self._ckpt_hash: str | None = None
        self._model_id: str = ""
        self._device: str = "cpu"
        self._gated: bool = False

    def load(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        self._cfg = cfg
        self._model_id = cfg.get("model_id", "medsam2")
        self._device = device

        self._ensure_medsam2_available()

        ckpt_path = cfg.get(
            "ckpt_path",
            str(_MEDSAM2_ROOT / "checkpoints" / "MedSAM2_latest.pt"),
        )
        config_file = cfg.get("config_file", "configs/sam2.1_hiera_t512.yaml")
        output_channels = cfg.get("output_channels", 14)

        self._load_model(ckpt_path, config_file, output_channels, device)

    def _ensure_medsam2_available(self) -> None:
        if not _MEDSAM2_ROOT.exists():
            raise RuntimeError(
                f"MedSAM2 repository not found at {_MEDSAM2_ROOT}. "
                "Clone it: git clone https://github.com/bowang-lab/MedSAM2.git third_party/medsam2"
            )
        medsam2_str = str(_MEDSAM2_ROOT)
        if medsam2_str not in sys.path:
            sys.path.insert(0, medsam2_str)

    def _load_model(
        self,
        ckpt_path: str,
        config_file: str,
        output_channels: int,
        device: str = "cpu",
    ) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MedSAM2 checkpoint not found: {path}. "
                "Download it via: cd third_party/medsam2 && bash download.sh"
            )

        self._ckpt_hash = self._compute_ckpt_hash(path)

        from sam2.build_sam import build_sam2

        logger.info(f"MedSAM2Backend: loading model from {path}")
        self._model = build_sam2(
            config_file=config_file,
            ckpt_path=str(path),
            device=device,
            mode="eval",
        )

        # Freeze all parameters
        for p in self._model.parameters():
            p.requires_grad = False

        # Detect feature channels from the image encoder neck
        feat_channels = self._model.hidden_dim  # set from image_encoder.neck.d_model
        adapter_channels = cfg_adapter = self._cfg.get("adapter_channels", feat_channels)

        # Choose adapter type and mode
        adapter_type = self._cfg.get("adapter_type", "standard")
        adapter_mode = self._cfg.get("adapter_mode", "3d")  # "3d" or "slice_2d"
        deep_adapter = self._cfg.get("deep_adapter", False)

        if adapter_mode == "slice_2d":
            if adapter_type == "gated_residual":
                from .slice_adapter import SliceWiseGRACEAdapter
                initial_task = self._cfg.get("initial_task_id", "task_0")
                self._adapter = SliceWiseGRACEAdapter(
                    in_channels=adapter_channels,
                    out_channels=output_channels,
                    initial_task_id=initial_task,
                    gate_hidden=self._cfg.get("gate_hidden", 64),
                    min_gate=self._cfg.get("min_gate", 0.1),
                    deep=deep_adapter,
                ).to(device)
                self._gated = True
            else:
                from .slice_adapter import SliceWiseAdapter
                self._adapter = SliceWiseAdapter(
                    in_channels=adapter_channels,
                    out_channels=output_channels,
                    deep=deep_adapter,
                ).to(device)
                self._gated = False
        elif adapter_type == "gated_residual":
            from .gated_adapter import GatedResidualAdapter
            initial_task = self._cfg.get("initial_task_id", "task_0")
            self._adapter = GatedResidualAdapter(
                in_channels=adapter_channels,
                out_channels=output_channels,
                initial_task_id=initial_task,
                gate_hidden=self._cfg.get("gate_hidden", 64),
                min_gate=self._cfg.get("min_gate", 0.1),
                deep=deep_adapter,
            ).to(device)
            self._gated = True
        else:
            self._adapter = _OutputAdapter(
                in_channels=adapter_channels,
                out_channels=output_channels,
                deep=deep_adapter,
            ).to(device)
            self._gated = False

        # Load pre-trained adapter weights if provided
        adapter_ckpt = self._cfg.get("adapter_ckpt_path")
        if adapter_ckpt:
            self._load_adapter_weights(adapter_ckpt)

        logger.info(
            f"MedSAM2Backend: loaded (hash={self._ckpt_hash}, "
            f"feat_dim={feat_channels}, output_channels={output_channels}, "
            f"adapter={'pretrained' if adapter_ckpt else 'random'})"
        )

    def _load_adapter_weights(self, adapter_ckpt: str) -> None:
        """Load pre-trained adapter weights from a checkpoint."""
        path = Path(adapter_ckpt)
        if not path.exists():
            logger.warning(f"MedSAM2Backend: adapter checkpoint not found: {path}, using random init")
            return
        state = torch.load(str(path), map_location=self._device, weights_only=False)
        adapter_sd = state.get("adapter_state_dict", state)
        # Gated adapters save with state_dict_full() (includes prototypes/task
        # metadata). Detect this and use load_state_dict_full() to restore.
        if self._gated and hasattr(self._adapter, "load_state_dict_full"):
            self._adapter.load_state_dict_full(adapter_sd)
        else:
            self._adapter.load_state_dict(adapter_sd)
        logger.info(f"MedSAM2Backend: loaded pre-trained adapter from {path}")

    def reconfigure_adapter(self, out_channels: int, task_id: str | None = None) -> None:
        """Reconfigure adapter for a new task's channel count."""
        if self._adapter is None:
            return
        if self._gated and task_id is not None:
            # Gated adapter: add new task residual, freeze core if first transition
            if not self._adapter._core_frozen:
                self._adapter.freeze_core()
            self._adapter.add_task(task_id, out_channels)
            self._adapter.current_task = task_id
            return
        # Standard adapter: rebuild entirely
        current_out = self._adapter.proj[-1].out_channels
        if current_out == out_channels:
            return
        adapter_channels = self._cfg.get("adapter_channels", 256)
        logger.info(
            f"MedSAM2Backend: reconfiguring adapter {current_out} -> {out_channels} channels"
        )
        self._adapter = _OutputAdapter(
            in_channels=adapter_channels,
            out_channels=out_channels,
        ).to(self._device)

    @staticmethod
    def _compute_ckpt_hash(path: Path, nbytes: int = 4096) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(nbytes))
        return h.hexdigest()[:16]

    def _preprocess_slice(self, img_slice: torch.Tensor) -> torch.Tensor:
        """Prepare a 2D slice for SAM2: grayscale->RGB, resize to 512, normalize."""
        B = img_slice.shape[0]

        # Grayscale to RGB
        if img_slice.shape[1] == 1:
            img_slice = img_slice.repeat(1, 3, 1, 1)
        elif img_slice.shape[1] != 3:
            img_slice = img_slice[:, :3, :, :]

        # Resize to expected input size
        if img_slice.shape[-2] != _MEDSAM2_IMG_SIZE or img_slice.shape[-1] != _MEDSAM2_IMG_SIZE:
            img_slice = F.interpolate(
                img_slice,
                size=(_MEDSAM2_IMG_SIZE, _MEDSAM2_IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet normalization
        mean = _IMG_MEAN.to(img_slice.device, img_slice.dtype)
        std = _IMG_STD.to(img_slice.device, img_slice.dtype)
        img_slice = (img_slice - mean) / std

        return img_slice

    def _extract_features_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image encoder features from a 3D volume slice-by-slice.

        Args:
            x: Input tensor ``(B, C_in, D, H, W)``

        Returns:
            Feature volume ``(B, feat_dim, D, feat_H, feat_W)``
        """
        B, C_in, D, H, W = x.shape

        slice_features = []
        for d in range(D):
            img_slice = x[:, :, d, :, :]  # (B, C_in, H, W)
            img_slice = self._preprocess_slice(img_slice)

            with torch.no_grad():
                backbone_out = self._model.forward_image(img_slice)
                # backbone_fpn is a list of multi-scale features; take the
                # highest-level (last) feature map for KD
                feat = backbone_out["backbone_fpn"][-1]  # (B, C_feat, fH, fW)

            slice_features.append(feat)

        # Stack along depth dimension: (B, C_feat, D, fH, fW)
        features_3d = torch.stack(slice_features, dim=2)
        return features_3d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None or self._adapter is None:
            raise RuntimeError(
                "MedSAM2Backend: model not loaded. Provide ckpt_path in config."
            )
        B, C_in, D, H, W = x.shape
        features_3d = self._extract_features_3d(x)
        target_shape = (D, H, W)
        if self._gated:
            logits, _ = self._adapter(features_3d, target_shape)
            return logits
        return self._adapter(features_3d, target_shape)

    def forward_with_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (logits, gate). Gate is None for standard adapter."""
        if self._model is None or self._adapter is None:
            raise RuntimeError("MedSAM2Backend: model not loaded.")
        B, C_in, D, H, W = x.shape
        features_3d = self._extract_features_3d(x)
        target_shape = (D, H, W)
        if self._gated:
            return self._adapter(features_3d, target_shape)
        return self._adapter(features_3d, target_shape), None

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("MedSAM2Backend: model not loaded")
        features_3d = self._extract_features_3d(x)
        return {"backbone_features": features_3d}

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self._model_id,
            "ckpt_hash": self._ckpt_hash,
            "source_mode": "medsam2",
            "frozen": True,
            "use_features": False,
            "feature_layers": [],
        }

    def to(self, device) -> "MedSAM2Backend":
        self._device = str(device)
        if self._model is not None:
            self._model.to(device)
        if self._adapter is not None:
            self._adapter.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"teacher_metadata": self.metadata}
        if self._adapter is not None:
            if self._gated and hasattr(self._adapter, "state_dict_full"):
                state["adapter_state_dict"] = self._adapter.state_dict_full()
            else:
                state["adapter_state_dict"] = self._adapter.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore adapter weights from a previously saved state."""
        adapter_sd = state.get("adapter_state_dict")
        if adapter_sd is not None and self._adapter is not None:
            if self._gated and hasattr(self._adapter, "load_state_dict_full"):
                self._adapter.load_state_dict_full(adapter_sd)
            else:
                self._adapter.load_state_dict(adapter_sd)

    def eval(self) -> "MedSAM2Backend":
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
