"""Shared loss functions for training and adapter pre-training.

Centralizes loss computation to avoid duplication between the student
training pipeline and the teacher adapter pre-training script.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dicece_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + Cross-Entropy loss for medical segmentation.

    Args:
        logits: Predicted logits ``(B, C, *spatial)``.
        target: Ground truth class indices ``(B, *spatial)``.

    Returns:
        Scalar loss (mean Dice loss over foreground classes + CE).
    """
    if logits.ndim < 3 or target.ndim < 2:
        raise ValueError(
            f"dicece_loss shape error: logits={logits.shape}, target={target.shape}"
        )
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes).permute(
        0, -1, *range(1, target.ndim)
    ).float()

    smooth = 1e-5
    dice_loss = 0.0
    count = 0
    for c in range(1, num_classes):
        pred_c = probs[:, c]
        target_c = target_one_hot[:, c]
        intersection = (pred_c * target_c).sum()
        dice_score = (2.0 * intersection + smooth) / (
            pred_c.sum() + target_c.sum() + smooth
        )
        dice_loss += 1.0 - dice_score
        count += 1
    if count > 0:
        dice_loss /= count

    ce_loss = F.cross_entropy(logits, target)
    return dice_loss + ce_loss
