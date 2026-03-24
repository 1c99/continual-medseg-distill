from __future__ import annotations

from .base import ContinualMethod


class DistillReplayEWCMethod(ContinualMethod):
    """Combined method skeleton: distillation + replay + EWC.

    TODO:
    - implement replay buffer strategy
    - implement teacher-based distillation objective
    - estimate Fisher information and apply EWC penalty
    """

    pass
