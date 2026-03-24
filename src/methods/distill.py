from __future__ import annotations

from .base import ContinualMethod


class DistillMethod(ContinualMethod):
    """Distillation skeleton.

    TODO:
    - maintain teacher snapshot after each task
    - compute KL/logit distillation loss on current/replay samples
    """

    pass
