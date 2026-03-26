from __future__ import annotations

import copy
import logging
from typing import Dict
import torch
import torch.nn.functional as F

from .replay import ReplayMethod

logger = logging.getLogger(__name__)


class DistillReplayEWCMethod(ReplayMethod):
    """Combined scaffold: distillation + replay + EWC-style regularization.

    NOTE: EWC term here is a lightweight placeholder (L2-to-prev-params weighted by lambda).
    Replace with Fisher-based EWC for publication-grade experiments.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        mcfg = cfg.get("method", {})
        kd_cfg = mcfg.get("kd", {})
        ewc_cfg = mcfg.get("ewc", {})

        self.kd_weight = float(kd_cfg.get("weight", 1.0))
        self.temperature = float(kd_cfg.get("temperature", 2.0))
        self.ewc_weight = float(ewc_cfg.get("weight", 0.1))

        self.teacher_model: torch.nn.Module | None = None
        self.prev_params: Dict[str, torch.Tensor] = {}

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

    def _ewc_penalty(self, model: torch.nn.Module, device: str) -> torch.Tensor:
        if not self.prev_params:
            return torch.tensor(0.0, device=device)
        penalty = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in self.prev_params:
                penalty = penalty + (p - self.prev_params[n].to(device)).pow(2).mean()
        return penalty

    def training_loss(self, model: torch.nn.Module, batch: Dict, device: str) -> torch.Tensor:
        # base CE + replay from ReplayMethod
        base_loss = super().training_loss(model, batch, device)

        # KD loss (only when teacher exists, i.e. after first task)
        kd = torch.tensor(0.0, device=base_loss.device)
        if self.teacher_model is not None:
            x = batch["image"].to(base_loss.device)
            student_logits = model(x)
            with torch.no_grad():
                teacher_logits = self.teacher_model(x)
            T = self.temperature
            kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)

        ewc = self._ewc_penalty(model, base_loss.device)
        return base_loss + self.kd_weight * kd + self.ewc_weight * ewc

    def post_task_update(self, model: torch.nn.Module) -> None:
        self.teacher_model = copy.deepcopy(model).eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.prev_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
