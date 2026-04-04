"""PLOP: Pooled Local Distillation for continual segmentation.

Implements multi-scale pooled feature distillation from:
  Douillard et al., "PLOP: Learning without Forgetting for Continual
  Semantic Segmentation", CVPR 2021.

Self-distillation: snapshots the model at the end of each task and uses it
as teacher for the next. Distillation acts on intermediate features at
multiple spatial pooling scales, not just output logits.

Also includes pseudo-labeling: the old model's predictions on the new task's
background pixels provide soft targets for old classes.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinualMethod

logger = logging.getLogger(__name__)


class PLOPMethod(ContinualMethod):
    """PLOP: multi-scale pooled local distillation + pseudo-labeling."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        mcfg = cfg.get("method", {})
        plop_cfg = mcfg.get("plop", {})

        self.pod_weight = float(plop_cfg.get("pod_weight", 0.01))
        self.pseudo_weight = float(plop_cfg.get("pseudo_weight", 1.0))
        # pod_scales: spatial subdivision factors. scale=1 → global pool,
        # scale=2 → 2 strips per dim, scale=4 → 4 strips per dim.
        self.pod_scales = list(plop_cfg.get("pod_scales", [1, 2, 4]))
        self.pod_normalize = bool(plop_cfg.get("pod_normalize", True))

        self._old_model: nn.Module | None = None
        self._old_num_classes: int = 0

    # ---- POD: Pooled Output Distillation (Algorithm 1) ----

    def _pod_loss(
        self,
        features_new: List[torch.Tensor],
        features_old: List[torch.Tensor],
    ) -> torch.Tensor:
        """Pooled output distillation across multiple spatial scales.

        For each intermediate feature map and each scale s, divide each
        spatial dimension into s strips, width-pool each strip (sum over that
        dimension), L2-normalize, and compute squared L2 distance between
        old and new representations.

        This is faithful to Algorithm 1 in Douillard et al., CVPR 2021:
        for each feature h of shape (B, C, *spatial), for each scale s,
        for each spatial dimension d:
          - split h into s strips along d
          - for each strip, sum-pool along d → (B, C, *remaining_spatial)
          - L2-normalize along channel dim
          - accumulate ||h_new - h_old||^2
        """
        loss = torch.tensor(0.0, device=features_new[0].device)
        count = 0

        for f_new, f_old in zip(features_new, features_old):
            if f_new.shape != f_old.shape:
                continue

            for scale in self.pod_scales:
                # For each spatial dimension, split into `scale` strips
                for dim in range(2, f_new.ndim):
                    dim_size = f_new.shape[dim]
                    n_strips = min(scale, dim_size)
                    if n_strips < 1:
                        continue

                    # Split into strips along this dim
                    strip_size = dim_size // n_strips
                    for s in range(n_strips):
                        start = s * strip_size
                        end = start + strip_size if s < n_strips - 1 else dim_size

                        # Slice the strip
                        slices = [slice(None)] * f_new.ndim
                        slices[dim] = slice(start, end)

                        strip_new = f_new[tuple(slices)]
                        strip_old = f_old[tuple(slices)]

                        # Width-pool: sum over this dimension
                        p_new = strip_new.sum(dim=dim)
                        p_old = strip_old.sum(dim=dim)

                        # Flatten spatial dims, keep batch and channel
                        p_new = p_new.flatten(2)  # (B, C, -1)
                        p_old = p_old.flatten(2)

                        if self.pod_normalize:
                            p_new = F.normalize(p_new, dim=1)
                            p_old = F.normalize(p_old, dim=1)

                        loss = loss + torch.frobenius_norm(p_new - p_old) ** 2
                        count += 1

        return loss / max(count, 1)

    # ---- Pseudo-labeling ----

    def _pseudo_label_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        old_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Pseudo-labeling: old model predictions on background pixels.

        Where the ground truth is background (class 0) in the new task,
        the old model may have assigned these pixels to old classes.
        Use the old model's soft predictions as the target via KD loss
        on background pixels only.
        """
        bg_mask = target == 0  # (B, *spatial)
        if bg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        old_ch = min(self._old_num_classes, logits.shape[1])
        if old_ch <= 1:
            return torch.tensor(0.0, device=logits.device)

        # Expand bg_mask to cover channel dim: (B, C, *spatial)
        bg_expanded = bg_mask.unsqueeze(1).expand_as(logits[:, :old_ch])

        # Extract background voxels for old class channels
        # Shape: (N_bg, old_ch) where N_bg = number of background voxels
        new_bg = logits[:, :old_ch][bg_expanded].view(-1, old_ch)
        old_bg = old_logits[:, :old_ch][bg_expanded].view(-1, old_ch)

        T = 2.0
        return F.kl_div(
            F.log_softmax(new_bg / T, dim=1),
            F.softmax(old_bg / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

    # ---- Feature extraction via hooks ----

    def _extract_features(
        self, model: nn.Module, x: torch.Tensor, *, no_grad: bool = False
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with hooks to capture intermediate encoder features.

        Returns (logits, list_of_intermediate_features).
        """
        features: List[torch.Tensor] = []
        hooks = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                features.append(output)

        from src.engine.distributed import unwrap_model
        base = unwrap_model(model)

        # Hook into MONAI UNet's downsampling conv layers
        for name, module in base.named_modules():
            if isinstance(module, nn.Conv3d) and "down" in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        # Fallback: hook every other Conv3d (encoder layers)
        if not hooks:
            conv_modules = [m for m in base.modules() if isinstance(m, nn.Conv3d)]
            for m in conv_modules[::2][:4]:
                hooks.append(m.register_forward_hook(hook_fn))

        if no_grad:
            with torch.no_grad():
                logits = model(x)
        else:
            logits = model(x)

        for h in hooks:
            h.remove()

        return logits, features

    # ---- Training loss ----

    def training_loss(
        self, model: nn.Module, batch: Dict, device: str
    ) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        if self._old_model is None:
            logits = model(x)
            return self._compute_loss(logits, y)

        # Single forward pass each, capturing intermediate features
        logits, features_new = self._extract_features(model, x)
        old_logits, features_old = self._extract_features(
            self._old_model, x, no_grad=True
        )

        # Segmentation loss on current task
        seg_loss = self._compute_loss(logits, y)

        # POD: multi-scale pooled distillation on intermediate features
        pod = self._pod_loss(features_new, features_old)

        # Pseudo-labeling: KD on background pixels using old model predictions
        pseudo = self._pseudo_label_loss(logits, y, old_logits)

        return seg_loss + self.pod_weight * pod + self.pseudo_weight * pseudo

    def post_task_update(self, model: nn.Module, **kwargs) -> None:
        from src.engine.distributed import unwrap_model
        self._old_model = copy.deepcopy(unwrap_model(model))
        self._old_model.eval()
        for p in self._old_model.parameters():
            p.requires_grad = False
        # Infer output channels from model's final conv
        for m in reversed(list(self._old_model.modules())):
            if isinstance(m, nn.Conv3d):
                self._old_num_classes = m.out_channels
                break
        logger.info(f"PLOP: snapshot saved (old_classes={self._old_num_classes})")

    def save_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        if self._old_model is not None:
            state["old_model"] = self._old_model.state_dict()
            state["old_num_classes"] = self._old_num_classes
        torch.save(state, path)

    def load_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if "old_model" in state and model_template is not None:
            self._old_model = copy.deepcopy(model_template)
            self._old_model.load_state_dict(state["old_model"])
            self._old_model.eval()
            for p in self._old_model.parameters():
                p.requires_grad = False
            self._old_num_classes = state.get("old_num_classes", 0)
