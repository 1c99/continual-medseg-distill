from __future__ import annotations

import logging
from typing import Dict
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContinualMethod:
    """Base continual-learning method interface."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._validate_config()

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_type = self.cfg.get("train", {}).get("loss_type", "dicece")
        if loss_type == "dicece":
            return self._dicece_loss(logits, target)
        else:
            return F.cross_entropy(logits, target)

    def _dicece_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        # One-hot encode target: target shape is (B, *spatial), result is (B, C, *spatial)
        target_one_hot = F.one_hot(target, num_classes).permute(0, -1, *range(1, target.ndim)).float()

        # Dice loss — skip background (class 0)
        smooth = 1e-5
        dice_loss = 0.0
        count = 0
        for c in range(1, num_classes):
            pred_c = probs[:, c]
            target_c = target_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            dice_score = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            dice_loss += 1.0 - dice_score
            count += 1
        if count > 0:
            dice_loss /= count

        ce_loss = F.cross_entropy(logits, target)
        return dice_loss + ce_loss

    def training_loss(self, model: torch.nn.Module, batch: Dict, device: str) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        return self._compute_loss(logits, y)

    def _validate_config(self) -> None:
        """Override in subclasses to validate method-specific hyperparams."""
        pass

    def post_task_update(self, model: torch.nn.Module, **kwargs) -> None:
        # Hook for method-specific memory/teacher/Fisher updates.
        return
