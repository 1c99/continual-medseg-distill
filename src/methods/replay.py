from __future__ import annotations

import logging
import random
from typing import Dict, List
import torch
import torch.nn.functional as F

from .base import ContinualMethod

logger = logging.getLogger(__name__)


class ReplayMethod(ContinualMethod):
    """Simple replay baseline with in-memory sample buffer.

    NOTE: This is a practical scaffold, not final research-grade replay policy.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        replay_cfg = cfg.get("method", {}).get("replay", {})
        self.buffer_size = int(replay_cfg.get("buffer_size", 64))
        self.replay_weight = float(replay_cfg.get("weight", 1.0))
        self.memory: List[Dict[str, torch.Tensor]] = []

    def _validate_config(self) -> None:
        mcfg = self.cfg.get("method", {})
        if "replay" not in mcfg:
            logger.warning(
                "ReplayMethod: missing method.replay config section; "
                "using defaults (buffer_size=64, weight=1.0)"
            )
        replay_cfg = mcfg.get("replay", {})
        if "buffer_size" not in replay_cfg:
            logger.warning("ReplayMethod: method.replay.buffer_size not set; defaulting to 64")

    def _push_batch_to_memory(self, batch: Dict[str, torch.Tensor]) -> None:
        x = batch["image"].detach().cpu()
        y = batch["label"].detach().cpu()
        for i in range(x.shape[0]):
            self.memory.append({"image": x[i], "label": y[i]})
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size :]

    def _sample_memory(self, k: int) -> Dict[str, torch.Tensor] | None:
        if len(self.memory) == 0:
            return None
        idx = [random.randrange(len(self.memory)) for _ in range(min(k, len(self.memory)))]
        x = torch.stack([self.memory[i]["image"] for i in idx], dim=0)
        y = torch.stack([self.memory[i]["label"] for i in idx], dim=0)
        return {"image": x, "label": y}

    def training_loss(self, model: torch.nn.Module, batch: Dict, device: str) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss_cur = self._compute_loss(logits, y)

        # replay contribution
        replay = self._sample_memory(k=x.shape[0])
        if replay is not None:
            xr = replay["image"].to(device)
            yr = replay["label"].to(device)
            logits_r = model(xr)
            loss_rep = self._compute_loss(logits_r, yr)
            loss = loss_cur + self.replay_weight * loss_rep
        else:
            loss = loss_cur

        # update memory after computing loss
        self._push_batch_to_memory(batch)
        return loss
