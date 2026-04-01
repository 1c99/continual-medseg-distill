"""MedSAM3 teacher backend for knowledge distillation.

Wraps the MedSAM3 model (SAM3 fine-tuned for medical segmentation) to
produce dense segmentation logits compatible with the
``(B, num_classes, D, H, W)`` format expected by the distillation pipeline.

MedSAM3 uses the same SAM3 base architecture but may include LoRA
fine-tuning weights for medical imaging tasks. It typically produces
denser, more medically-relevant outputs than base SAM3.

Requires:
- ``third_party/medsam3/`` to be cloned (see ``scripts/setup_external.sh``)
- A MedSAM3 checkpoint file
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

# Path to the vendored MedSAM3 repository
_MEDSAM3_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "medsam3"


class _OutputAdapter(nn.Module):
    """Projects MedSAM3 backbone features to dense segmentation logits.

    Identical to the SAM3 adapter — shared architecture since MedSAM3
    uses the same backbone. Kept separate for independent evolution.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 14, deep: bool = False):
        super().__init__()
        if deep:
            mid = in_channels
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels, mid, kernel_size=3, padding=1),
                nn.GroupNorm(16, mid),
                nn.GELU(),
                nn.Conv3d(mid, mid, kernel_size=3, padding=1),
                nn.GroupNorm(16, mid),
                nn.GELU(),
                nn.Conv3d(mid, mid // 2, kernel_size=3, padding=1),
                nn.GroupNorm(8, mid // 2),
                nn.GELU(),
                nn.Conv3d(mid // 2, out_channels, kernel_size=1),
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
            out = F.interpolate(out, size=target_shape, mode="trilinear", align_corners=False)
        return out


class MedSAM3Backend(TeacherBackend):
    """Teacher backend that wraps a MedSAM3 model.

    MedSAM3 is SAM3 fine-tuned for medical segmentation, potentially
    with LoRA adapters. The backend handles both:
    - Full checkpoint (all weights saved)
    - Base SAM3 + LoRA weights (loaded separately)

    Config keys (under ``method.kd.teacher``):
        type: medsam3
        ckpt_path: path to MedSAM3 checkpoint
        output_channels: int — number of output classes (must match student)
        model_id: str — provenance tag (optional)
        lora_path: str — optional path to LoRA weights (if separate from ckpt)
        adapter_channels: int — hidden dim for output adapter (default: 256)
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
        self._model_id = cfg.get("model_id", "medsam3")
        self._device = device

        ckpt_path = cfg.get("ckpt_path")
        output_channels = cfg.get("output_channels")

        if ckpt_path and ckpt_path != "auto":
            self._ensure_medsam3_available()
            self._load_model(ckpt_path, output_channels or 14, device)
        elif ckpt_path == "auto" or cfg.get("load_from_hf", False):
            self._ensure_medsam3_available()
            self._load_model_from_hf(output_channels or 14, device)

    def _ensure_medsam3_available(self) -> None:
        if not _MEDSAM3_ROOT.exists():
            raise RuntimeError(
                f"MedSAM3 repository not found at {_MEDSAM3_ROOT}. "
                "Run: bash scripts/setup_external.sh --medsam3"
            )
        # MedSAM3 bundles its own copy of sam3 — add it to sys.path
        medsam3_str = str(_MEDSAM3_ROOT)
        if medsam3_str not in sys.path:
            sys.path.insert(0, medsam3_str)

    def _load_model(
        self, ckpt_path: str, output_channels: int, device: str = "cpu"
    ) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(
                f"MedSAM3 checkpoint not found: {path}. "
                "Provide a valid method.kd.teacher.ckpt_path."
            )

        self._ckpt_hash = self._compute_ckpt_hash(path)

        # MedSAM3 bundles its own sam3 package
        from sam3 import build_sam3_image_model

        logger.info(f"MedSAM3Backend: loading base model from {path}")
        self._model = build_sam3_image_model(
            checkpoint_path=str(path),
            device=device,
            eval_mode=True,
            load_from_HF=False,
            enable_segmentation=True,
        )

        # Apply LoRA weights if provided separately
        lora_path = self._cfg.get("lora_path")
        if lora_path:
            self._apply_lora(lora_path)

        # Freeze all parameters
        for p in self._model.parameters():
            p.requires_grad = False

        adapter_channels = self._cfg.get("adapter_channels", 256)
        deep_adapter = self._cfg.get("deep_adapter", False)

        adapter_type = self._cfg.get("adapter_type", "standard")
        deep_adapter = self._cfg.get("deep_adapter", False)
        if adapter_type == "gated_residual":
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
            f"MedSAM3Backend: loaded (hash={self._ckpt_hash}, "
            f"output_channels={output_channels}, lora={'yes' if lora_path else 'no'}, "
            f"adapter={'gated' if self._gated else ('pretrained' if adapter_ckpt else 'random')})"
        )

    def _load_model_from_hf(
        self, output_channels: int, device: str = "cpu"
    ) -> None:
        """Load SAM3 base model via HuggingFace auto-download, then apply LoRA."""
        from sam3 import build_sam3_image_model

        logger.info("MedSAM3Backend: loading base model from HuggingFace (auto-download)")
        try:
            self._model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                load_from_HF=True,
                enable_segmentation=True,
            )
        except Exception as e:
            # If HF download fails (e.g. gated repo), build architecture without weights
            logger.warning(
                f"MedSAM3Backend: HF checkpoint download failed ({e}). "
                "Building model architecture without base weights."
            )
            self._model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                load_from_HF=False,
                checkpoint_path=None,
                enable_segmentation=True,
            )

        # Apply LoRA weights if provided
        lora_path = self._cfg.get("lora_path")
        if lora_path:
            self._apply_lora(lora_path)

        # Freeze all parameters
        for p in self._model.parameters():
            p.requires_grad = False

        adapter_channels = self._cfg.get("adapter_channels", 256)
        deep_adapter = self._cfg.get("deep_adapter", False)

        adapter_type = self._cfg.get("adapter_type", "standard")
        if adapter_type == "gated_residual":
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

        logger.info(
            f"MedSAM3Backend: loaded from HF (output_channels={output_channels}, "
            f"lora={'yes' if lora_path else 'no'}, "
            f"adapter={'gated' if self._gated else ('deep' if deep_adapter else 'standard')})"
        )

    def _apply_lora(self, lora_path: str) -> None:
        """Apply LoRA weights from MedSAM3's fine-tuning."""
        lora_p = Path(lora_path)
        if not lora_p.exists():
            raise FileNotFoundError(f"MedSAM3 LoRA weights not found: {lora_p}")

        try:
            sys.path.insert(0, str(_MEDSAM3_ROOT))
            from lora_layers import apply_lora_to_model, LoRAConfig

            # Infer LoRA rank from checkpoint weights
            lora_state = torch.load(str(lora_p), map_location="cpu")
            lora_keys = {k: v for k, v in lora_state.items() if "lora_" in k}

            # Detect rank from first lora_A parameter shape
            inferred_rank = 8  # default
            for k, v in lora_keys.items():
                if "lora_A" in k:
                    inferred_rank = v.shape[1] if v.ndim == 2 else v.shape[0]
                    break
            logger.info(f"MedSAM3Backend: inferred LoRA rank={inferred_rank} from checkpoint")

            # Enable LoRA on all components that appear in the checkpoint
            lora_cfg = LoRAConfig(
                rank=inferred_rank,
                alpha=inferred_rank * 2,
                apply_to_vision_encoder=True,
                apply_to_text_encoder=True,
                apply_to_geometry_encoder=True,
                apply_to_detr_encoder=True,
                apply_to_detr_decoder=True,
                apply_to_mask_decoder=True,
            )
            self._model = apply_lora_to_model(self._model, lora_cfg)

            # Load only LoRA parameters
            model_sd = self._model.state_dict()
            model_sd.update(lora_keys)
            self._model.load_state_dict(model_sd)
            logger.info(f"MedSAM3Backend: applied {len(lora_keys)} LoRA parameters")
        except ImportError:
            logger.warning(
                "MedSAM3 lora_layers not found; loading checkpoint as full model"
            )

    def _load_adapter_weights(self, adapter_ckpt: str) -> None:
        """Load pre-trained adapter weights from a checkpoint."""
        path = Path(adapter_ckpt)
        if not path.exists():
            logger.warning(f"MedSAM3Backend: adapter checkpoint not found: {path}, using random init")
            return
        state = torch.load(str(path), map_location=self._device, weights_only=False)
        adapter_sd = state.get("adapter_state_dict", state)
        self._adapter.load_state_dict(adapter_sd)
        logger.info(f"MedSAM3Backend: loaded pre-trained adapter from {path}")

    def reconfigure_adapter(self, out_channels: int, task_id: str | None = None) -> None:
        """Reconfigure adapter for a new task's channel count."""
        if self._adapter is None:
            return
        if self._gated and task_id is not None:
            if not self._adapter._core_frozen:
                self._adapter.freeze_core()
            self._adapter.add_task(task_id, out_channels)
            self._adapter.current_task = task_id
            return
        current_out = self._adapter.proj[-1].out_channels
        if current_out == out_channels:
            return
        adapter_channels = self._cfg.get("adapter_channels", 256)
        logger.info(
            f"MedSAM3Backend: reconfiguring adapter {current_out} -> {out_channels} channels"
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

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None or self._adapter is None:
            raise RuntimeError(
                "MedSAM3Backend: model not loaded. Provide ckpt_path in config."
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
            raise RuntimeError("MedSAM3Backend: model not loaded.")
        B, C_in, D, H, W = x.shape
        features_3d = self._extract_features_3d(x)
        target_shape = (D, H, W)
        if self._gated:
            return self._adapter(features_3d, target_shape)
        return self._adapter(features_3d, target_shape), None

    def _extract_features_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features from a 3D volume slice-by-slice."""
        B, C_in, D, H, W = x.shape
        backbone = self._model.backbone

        # SAM3 backbone expects 1008x1008 input images
        _SAM3_IMG_SIZE = 1008

        slice_features = []
        for d in range(D):
            img_slice = x[:, :, d, :, :]
            if img_slice.shape[1] == 1:
                img_slice = img_slice.repeat(1, 3, 1, 1)
            elif img_slice.shape[1] != 3:
                img_slice = img_slice[:, :3, :, :]

            # Resize to SAM3's expected spatial resolution
            if img_slice.shape[-2] != _SAM3_IMG_SIZE or img_slice.shape[-1] != _SAM3_IMG_SIZE:
                img_slice = F.interpolate(
                    img_slice, size=(_SAM3_IMG_SIZE, _SAM3_IMG_SIZE),
                    mode="bilinear", align_corners=False,
                )

            with torch.no_grad():
                # SAM3VLBackbone exposes forward_image -> dict with vision_features
                if hasattr(backbone, "forward_image"):
                    vis_out = backbone.forward_image(img_slice)
                elif hasattr(backbone, "vision_backbone"):
                    vis_out = backbone.vision_backbone(img_slice)
                else:
                    vis_out = backbone.visual(img_slice)

                if isinstance(vis_out, dict):
                    feat = vis_out.get(
                        "vision_features",
                        vis_out.get(
                            "feature_maps",
                            [vis_out.get("backbone_fpn", [None])[0]]
                        ),
                    )
                    # vision_features may be a single tensor or list
                    if isinstance(feat, (list, tuple)):
                        feat = feat[0]
                elif isinstance(vis_out, (list, tuple)):
                    feat = vis_out[0] if len(vis_out) > 0 else vis_out
                else:
                    feat = vis_out

            slice_features.append(feat)

        features_3d = torch.stack(slice_features, dim=2)
        return features_3d

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("MedSAM3Backend: model not loaded")
        features_3d = self._extract_features_3d(x)
        return {"backbone_features": features_3d}

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "model_id": self._model_id,
            "ckpt_hash": self._ckpt_hash,
            "source_mode": "medsam3",
            "frozen": True,
            "use_features": False,
            "feature_layers": [],
        }

    def to(self, device) -> "MedSAM3Backend":
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

    def eval(self) -> "MedSAM3Backend":
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
