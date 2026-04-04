"""MiB: Modeling the Background for continual semantic segmentation.

Implements:
  Cermelli et al., "Modeling the Background for Incremental Learning
  in Semantic Segmentation", CVPR 2020.

Key idea: In class-incremental segmentation, the "background" class is
overloaded — it contains both true background AND all unseen future classes.
MiB handles this by:
1. Unbiased cross-entropy: redistributes old model's background probability
   to new classes based on the new model's predictions.
2. Knowledge distillation on old class channels.
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

        self.kd_weight = float(mib_cfg.get("kd_weight", 10.0))
        self.unkd_weight = float(mib_cfg.get("unkd_weight", 10.0))
        self.temperature = float(mib_cfg.get("temperature", 2.0))

        self._old_model: nn.Module | None = None
        self._old_num_classes: int = 0

    def _unbiased_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        old_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unbiased cross-entropy that accounts for background shift.

        For new-task training, background (class 0) in the target may contain
        pixels that belong to old classes. Adjust by using old model's
        class probabilities to re-weight the background probability.
        """
        if old_logits is None or self._old_num_classes <= 1:
            return self._compute_loss(logits, target)

        # Standard cross-entropy for non-background
        ce = F.cross_entropy(logits, target, reduction="none")

        # For background pixels, redistribute old model's background probability
        bg_mask = target == 0
        if bg_mask.sum() > 0 and old_logits is not None:
            old_probs = F.softmax(old_logits, dim=1)
            # Old model's non-background probability = probability it assigns to old classes
            # This is the probability that the pixel belongs to an old class
            old_class_prob = 1.0 - old_probs[:, 0:1]  # (B, 1, *spatial)

            # Weighted CE: background pixels with high old-class probability
            # should be down-weighted (they're likely old classes, not true bg)
            bg_weight = old_probs[:, 0:1].squeeze(1)  # true bg probability
            # Flatten spatial dims
            flat_ce = ce[bg_mask]
            flat_weight = bg_weight[bg_mask]
            weighted_bg_ce = (flat_ce * flat_weight).mean()

            non_bg_ce = ce[~bg_mask].mean() if (~bg_mask).sum() > 0 else torch.tensor(0.0, device=ce.device)
            return weighted_bg_ce + non_bg_ce
        return ce.mean()

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
            reduction="mean",
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

        # Unbiased CE
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
