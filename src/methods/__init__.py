from .base import ContinualMethod
from .finetune import FineTuneMethod
from .replay import ReplayMethod
from .distill import DistillMethod
from .distill_replay_ewc import DistillReplayEWCMethod


def create_method(cfg):
    name = cfg.get("method", {}).get("name", "finetune")
    if name == "finetune":
        return FineTuneMethod(cfg)
    if name == "replay":
        return ReplayMethod(cfg)
    if name == "distill":
        return DistillMethod(cfg)
    if name == "distill_replay_ewc":
        return DistillReplayEWCMethod(cfg)
    raise ValueError(f"Unknown method: {name}")
