"""Combined method: distillation + replay + Fisher-based EWC regularization.

Inherits replay buffer from ReplayMethod and adds:
- Teacher-based knowledge distillation (via Teacher abstraction)
- Elastic Weight Consolidation with diagonal Fisher estimation
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay import ReplayMethod
from .teacher import Teacher

logger = logging.getLogger(__name__)


class DistillReplayEWCMethod(ReplayMethod):
    """Combined scaffold: distillation + replay + Fisher-based EWC regularization."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        mcfg = cfg.get("method", {})
        kd_cfg = mcfg.get("kd", {})
        ewc_cfg = mcfg.get("ewc", {})

        self.kd_weight = float(kd_cfg.get("weight", 1.0))
        self.temperature = float(kd_cfg.get("temperature", 2.0))
        self.ewc_weight = float(ewc_cfg.get("weight", 0.1))
        self.fisher_samples = int(ewc_cfg.get("fisher_samples", 64))

        teacher_cfg = kd_cfg.get("teacher", {})
        self.teacher = Teacher(teacher_cfg=teacher_cfg, global_cfg=cfg)

        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

    @property
    def teacher_model(self) -> nn.Module | None:
        """Backward-compatible access."""
        return self.teacher.model

    def _validate_config(self) -> None:
        super()._validate_config()
        mcfg = self.cfg.get("method", {})
        if "kd" not in mcfg:
            logger.warning(
                "DistillReplayEWCMethod: missing method.kd config section; "
                "using defaults (weight=1.0, temperature=2.0)"
            )
        if "ewc" not in mcfg:
            logger.warning(
                "DistillReplayEWCMethod: missing method.ewc config section; "
                "using defaults (weight=0.1)"
            )
        kd_cfg = mcfg.get("kd", {})
        if "weight" not in kd_cfg:
            logger.warning("DistillReplayEWCMethod: method.kd.weight not set; defaulting to 1.0")
        if "temperature" not in kd_cfg:
            logger.warning("DistillReplayEWCMethod: method.kd.temperature not set; defaulting to 2.0")
        ewc_cfg = mcfg.get("ewc", {})
        if "weight" not in ewc_cfg:
            logger.warning("DistillReplayEWCMethod: method.ewc.weight not set; defaulting to 0.1")
        if "fisher_samples" not in ewc_cfg:
            logger.warning("DistillReplayEWCMethod: method.ewc.fisher_samples not set; defaulting to 64")

    def _estimate_fisher(
        self,
        model: nn.Module,
        dataloader,
        device: str,
        n_samples: int = 64,
    ) -> Dict[str, torch.Tensor]:
        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        model.train()
        return fisher

    def _ewc_penalty(self, model: nn.Module, device) -> torch.Tensor:
        if not self.prev_params or not self.fisher:
            return torch.tensor(0.0, device=device)
        penalty = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in self.prev_params and n in self.fisher:
                penalty += (
                    self.fisher[n].to(device)
                    * (p - self.prev_params[n].to(device)).pow(2)
                ).sum()
        return penalty

    def training_loss(
        self, model: nn.Module, batch: Dict, device: str
    ) -> torch.Tensor:
        # base CE + replay from ReplayMethod
        base_loss = super().training_loss(model, batch, device)

        # KD loss (only when teacher exists)
        kd = torch.tensor(0.0, device=base_loss.device)
        if self.teacher.has_model:
            x = batch["image"].to(base_loss.device)
            student_logits = model(x)
            self.teacher.to(base_loss.device)
            teacher_logits = self.teacher(x)
            T = self.temperature
            kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)

        ewc = self._ewc_penalty(model, base_loss.device)
        return base_loss + self.kd_weight * kd + self.ewc_weight * ewc

    def post_task_update(self, model: nn.Module, **kwargs) -> None:
        # Teacher snapshot (skip for external teachers like SAM3/MedSAM3)
        if not self.teacher.is_external:
            self.teacher.snapshot(model)
        # Fisher estimation
        train_loader = kwargs.get("train_loader")
        device = next(model.parameters()).device
        if train_loader is not None:
            self.fisher = self._estimate_fisher(
                model, train_loader, str(device), self.fisher_samples
            )
        # Param snapshot
        self.prev_params = {
            n: p.detach().cpu().clone() for n, p in model.named_parameters()
        }

    def save_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: Dict[str, Any] = {
            "fisher": self.fisher,
            "prev_params": self.prev_params,
            "memory": [
                {"image": m["image"], "label": m["label"]} for m in self.memory
            ],
        }
        state.update(self.teacher.state_dict())
        torch.save(state, path)

    def load_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.fisher = state.get("fisher", {})
        self.prev_params = state.get("prev_params", {})
        self.memory = state.get("memory", [])
        self.teacher.load_state_dict(state, model_template=model_template)
