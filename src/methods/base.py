from __future__ import annotations

from typing import Dict
import torch


class ContinualMethod:
    """Base continual-learning method interface."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def training_loss(self, model: torch.nn.Module, batch: Dict, device: str) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        # Simple CE baseline for scaffold. TODO: swap for DiceCE / task-specific losses.
        return torch.nn.functional.cross_entropy(logits, y)

    def post_task_update(self, model: torch.nn.Module) -> None:
        # TODO: hook for method-specific memory/teacher/Fisher updates.
        return
