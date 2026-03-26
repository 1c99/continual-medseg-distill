from __future__ import annotations

import copy
import logging
from typing import Dict
import torch
import torch.nn.functional as F

from .base import ContinualMethod

logger = logging.getLogger(__name__)


class DistillMethod(ContinualMethod):
    """Logit distillation baseline.

    Flow:
    - Task 1: standard CE training (no teacher yet)
    - After task end: snapshot model as frozen teacher
    - Next tasks: CE + KD(teacher, student)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        kd_cfg = cfg.get("method", {}).get("kd", {})
        self.kd_weight = float(kd_cfg.get("weight", 1.0))
        self.temperature = float(kd_cfg.get("temperature", 2.0))
        self.teacher_model: torch.nn.Module | None = None

    def _validate_config(self) -> None:
        mcfg = self.cfg.get("method", {})
        if "kd" not in mcfg:
            logger.warning(
                "DistillMethod: missing method.kd config section; "
                "using defaults (weight=1.0, temperature=2.0)"
            )
        kd_cfg = mcfg.get("kd", {})
        if "weight" not in kd_cfg:
            logger.warning("DistillMethod: method.kd.weight not set; defaulting to 1.0")
        if "temperature" not in kd_cfg:
            logger.warning("DistillMethod: method.kd.temperature not set; defaulting to 2.0")

    def training_loss(self, model: torch.nn.Module, batch: Dict, device: str) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        student_logits = model(x)
        ce = F.cross_entropy(student_logits, y)

        if self.teacher_model is None:
            return ce

        with torch.no_grad():
            teacher_logits = self.teacher_model(x)

        T = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        return ce + self.kd_weight * kd

    def post_task_update(self, model: torch.nn.Module) -> None:
        self.teacher_model = copy.deepcopy(model).eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
