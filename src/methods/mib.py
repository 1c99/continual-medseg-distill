"""MiB: Modeling the Background for continual semantic segmentation.

Implements:
  Cermelli et al., "Modeling the Background for Incremental Learning
  in Semantic Segmentation", CVPR 2020.

Key idea: In class-incremental segmentation, the "background" class is
overloaded — it contains both true background AND all unseen future classes.
MiB handles this by:
1. Unbiased cross-entropy (Eq. 4-5): For background-labeled pixels, use the
   old model's class probabilities as soft targets instead of hard bg label.
   This prevents the model from being pushed to predict bg on old-class pixels.
2. Knowledge distillation on old class channels.

In task-incremental mode (separate heads per task), the unbiased CE still
helps: during Task B training, any replay/shared representations that feed
Task A's head benefit from the soft background target.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinualMethod

logger = logging.getLogger(__name__)


class MiBMethod(ContinualMethod):
    """Modeling the Background for incremental segmentation."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        mcfg = cfg.get("method", {})
        mib_cfg = mcfg.get("mib", {})

        self.kd_weight = float(mib_cfg.get("kd_weight", 1.0))
        self.temperature = float(mib_cfg.get("temperature", 2.0))

        self._old_model: nn.Module | None = None
        self._old_num_classes: int = 0

    def _unbiased_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        old_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Unbiased cross-entropy per Cermelli et al. Eqs 4-5.

        For pixels labeled as background (class 0), the old model's per-class
        soft predictions are used as targets instead of the hard bg label.
        This prevents the model from collapsing old-class representations
        into background.

        For non-background pixels, standard cross-entropy is used.
        """
        old_ch = min(self._old_num_classes, logits.shape[1])
        bg_mask = target == 0  # (B, *spatial)

        # Non-background pixels: standard CE
        if (~bg_mask).sum() > 0:
            non_bg_ce = F.cross_entropy(logits, target, reduction="none")
            non_bg_loss = non_bg_ce[~bg_mask].mean()
        else:
            non_bg_loss = torch.tensor(0.0, device=logits.device)

        if bg_mask.sum() == 0 or old_ch <= 1:
            return non_bg_loss + F.cross_entropy(logits, target, reduction="none")[bg_mask].mean() \
                if bg_mask.sum() > 0 else non_bg_loss

        # Background pixels: soft CE with old model's distribution as target
        # Old model gives probability distribution over {bg, old_class_1, ..., old_class_N}
        # Use this as the soft target for the new model's first old_ch channels.
        old_probs = F.softmax(old_logits[:, :old_ch], dim=1)  # (B, old_ch, *spatial)

        # New model's log-softmax over all channels (including new classes)
        new_log_probs = F.log_softmax(logits, dim=1)  # (B, C, *spatial)

        # Soft CE on background voxels: -Σ_c p_old(c) * log(p_new(c))
        # Only computed over old class channels (bg + old classes)
        soft_ce = -(old_probs * new_log_probs[:, :old_ch]).sum(dim=1)  # (B, *spatial)
        bg_loss = soft_ce[bg_mask].mean()

        return non_bg_loss + bg_loss

    def _kd_loss(
        self,
        logits: torch.Tensor,
        old_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KD loss on old class channels."""
        old_ch = min(self._old_num_classes, logits.shape[1])
        if old_ch <= 1:
            return torch.tensor(0.0, device=logits.device)

        T = self.temperature
        return F.kl_div(
            F.log_softmax(logits[:, :old_ch] / T, dim=1),
            F.softmax(old_logits[:, :old_ch] / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

    def training_loss(
        self, model: nn.Module, batch: Dict, device: str
    ) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(x)

        if self._old_model is None:
            return self._compute_loss(logits, y)

        with torch.no_grad():
            old_logits = self._old_model(x)

        # Unbiased CE (Eq. 4-5)
        seg_loss = self._unbiased_ce(logits, y, old_logits)

        # KD on old class channels
        kd = self._kd_loss(logits, old_logits)

        return seg_loss + self.kd_weight * kd

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
        logger.info(f"MiB: snapshot saved (old_classes={self._old_num_classes})")

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
