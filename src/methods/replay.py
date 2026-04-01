from __future__ import annotations

import logging
import random
from pathlib import Path
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
        self._current_task_id: str | None = None

    def set_current_task(self, task_id: str) -> None:
        """Called by multi_task_trainer before each task."""
        self._current_task_id = task_id

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
            entry = {"image": x[i], "label": y[i]}
            if self._current_task_id is not None:
                entry["task_id"] = self._current_task_id
            self.memory.append(entry)
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size :]

    def _sample_memory(self, k: int) -> Dict[str, torch.Tensor | list] | None:
        if len(self.memory) == 0:
            return None
        idx = [random.randrange(len(self.memory)) for _ in range(min(k, len(self.memory)))]
        x = torch.stack([self.memory[i]["image"] for i in idx], dim=0)
        y = torch.stack([self.memory[i]["label"] for i in idx], dim=0)
        task_ids = [self.memory[i].get("task_id") for i in idx]
        return {"image": x, "label": y, "task_ids": task_ids}

    def training_loss(self, model: torch.nn.Module, batch: Dict, device: str) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss_cur = self._compute_loss(logits, y)

        # replay contribution
        replay = self._sample_memory(k=x.shape[0])
        if replay is not None:
            loss_rep = self._replay_loss(model, replay, device)
            loss = loss_cur + self.replay_weight * loss_rep
        else:
            loss = loss_cur

        # update memory after computing loss
        self._push_batch_to_memory(batch)
        return loss

    def _replay_loss(
        self, model: torch.nn.Module, replay: Dict, device: str
    ) -> torch.Tensor:
        """Compute replay loss, switching heads per task if multi-head."""
        xr = replay["image"].to(device)
        yr = replay["label"].to(device)
        task_ids = replay.get("task_ids", [None] * xr.shape[0])

        has_multi_head = hasattr(model, "current_task")
        unique_tasks = set(task_ids)

        # Fast path: all replay samples from same task (common in 2-task setting)
        if len(unique_tasks) == 1 and has_multi_head:
            replay_task = task_ids[0]
            if replay_task is not None:
                prev_task = model.current_task
                model.current_task = replay_task
                logits_r = model(xr)
                loss = self._compute_loss(logits_r, yr)
                model.current_task = prev_task
                return loss

        # No multi-head or no task tags — standard replay
        if not has_multi_head or all(t is None for t in task_ids):
            logits_r = model(xr)
            return self._compute_loss(logits_r, yr)

        # Multi-task replay: group by task, compute loss per group
        prev_task = model.current_task
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        for task_id in unique_tasks:
            mask = [i for i, t in enumerate(task_ids) if t == task_id]
            if task_id is not None:
                model.current_task = task_id
            logits_g = model(xr[mask])
            total_loss = total_loss + self._compute_loss(logits_g, yr[mask])
            count += 1
        model.current_task = prev_task
        return total_loss / max(count, 1)

    def save_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "memory": [
                {k: v for k, v in m.items()}
                for m in self.memory
            ],
        }
        torch.save(state, path)

    def load_state(self, path: Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.memory = state.get("memory", [])
