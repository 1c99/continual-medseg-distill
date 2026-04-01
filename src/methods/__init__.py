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
    if name == "lwf":
        # LwF is self-distillation: force snapshot teacher regardless of config
        import copy as _copy
        lwf_cfg = _copy.deepcopy(cfg)
        lwf_cfg["method"]["name"] = "distill"
        lwf_cfg["method"].setdefault("kd", {})
        lwf_cfg["method"]["kd"].setdefault("mode", "logit")
        lwf_cfg["method"]["kd"].setdefault("weight", 0.5)
        lwf_cfg["method"]["kd"].setdefault("temperature", 2.0)
        lwf_cfg["method"]["kd"]["teacher"] = {"type": "snapshot"}
        return DistillMethod(lwf_cfg)
    if name == "distill_replay_ewc":
        return DistillReplayEWCMethod(cfg)
    raise ValueError(f"Unknown method: {name}")
