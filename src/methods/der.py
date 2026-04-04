"""DER++: Dark Experience Replay++ for continual learning.

Implements:
  Buzzega et al., "Dark Experience for General Continual Learning:
  a Strong, Simple Baseline", NeurIPS 2020.

DER++ stores both samples AND their logits in the replay buffer.
When replaying, it matches the current model's logits to the stored
logits (dark knowledge), providing a stronger anti-forgetting signal
than raw replay alone.

Two replay objectives:
1. Standard replay CE: current model on stored (x, y) pairs
2. Logit matching: MSE between current logits and stored logits on buffer samples
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinualMethod

logger = logging.getLogger(__name__)


class DERPlusPlusMethod(ContinualMethod):
    """DER++: Dark Experience Replay with logit matching."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        mcfg = cfg.get("method", {})
        der_cfg = mcfg.get("der", {})

        self.buffer_size = int(der_cfg.get("buffer_size", 200))
        self.alpha = float(der_cfg.get("alpha", 0.5))  # weight for logit matching
        self.beta = float(der_cfg.get("beta", 0.5))   # weight for standard replay CE

        # Buffer stores (image, label, logits, task_id)
        self.memory: List[Dict[str, torch.Tensor]] = []
        self._current_task_id: str | None = None

    def set_current_task(self, task_id: str) -> None:
        self._current_task_id = task_id

    def _push_batch_to_memory(
        self,
        batch: Dict[str, torch.Tensor],
        logits: torch.Tensor,
    ) -> None:
        """Store samples WITH their logits (dark experience)."""
        x = batch["image"].detach().cpu()
        y = batch["label"].detach().cpu()
        z = logits.detach().cpu()

        for i in range(x.shape[0]):
            entry = {
                "image": x[i],
                "label": y[i],
                "logits": z[i],
            }
            if self._current_task_id is not None:
                entry["task_id"] = self._current_task_id
            self.memory.append(entry)

        # Reservoir sampling for fixed buffer
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:]

    def _sample_memory(self, k: int) -> List[Dict[str, torch.Tensor]] | None:
        if len(self.memory) == 0:
            return None
        idx = [random.randrange(len(self.memory)) for _ in range(min(k, len(self.memory)))]
        return [self.memory[i] for i in idx]

    def training_loss(
        self, model: nn.Module, batch: Dict, device: str
    ) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        # Current task forward
        logits = model(x)
        seg_loss = self._compute_loss(logits, y)

        # Store current batch with logits
        self._push_batch_to_memory(batch, logits)

        # Replay
        samples = self._sample_memory(k=x.shape[0])
        if samples is None:
            return seg_loss

        # Group replay samples by task (different tasks have different logit channels)
        has_multi_head = hasattr(model, "current_task")
        groups: Dict[str | None, List[Dict]] = {}
        for s in samples:
            tid = s.get("task_id")
            groups.setdefault(tid, []).append(s)

        replay_ce = torch.tensor(0.0, device=device)
        logit_match = torch.tensor(0.0, device=device)
        count = 0

        prev_task = model.current_task if has_multi_head else None

        for tid, group in groups.items():
            xr = torch.stack([s["image"] for s in group]).to(device)
            yr = torch.stack([s["label"] for s in group]).to(device)
            stored = torch.stack([s["logits"] for s in group]).to(device)

            if has_multi_head and tid is not None:
                model.current_task = tid

            current_logits = model(xr)

            # Standard replay CE (beta term)
            if self.beta > 0:
                replay_ce = replay_ce + self._compute_loss(current_logits, yr)

            # Logit matching (alpha term) — DER++ core contribution
            if self.alpha > 0:
                min_ch = min(current_logits.shape[1], stored.shape[1])
                logit_match = logit_match + F.mse_loss(
                    current_logits[:, :min_ch], stored[:, :min_ch]
                )

            count += 1

        if has_multi_head and prev_task is not None:
            model.current_task = prev_task

        replay_ce = replay_ce / max(count, 1)
        logit_match = logit_match / max(count, 1)

        return seg_loss + self.alpha * logit_match + self.beta * replay_ce

    def post_task_update(self, model: nn.Module, **kwargs) -> None:
        logger.info(
            f"DER++: buffer has {len(self.memory)} samples "
            f"after task update"
        )

    def save_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"memory": self.memory}, path)

    def load_state(self, path: Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.memory = state.get("memory", [])
