"""Teacher backend factory and public exports."""
from __future__ import annotations

from typing import Any, Dict

from .base import TeacherBackend
from .unet import UNetBackend


def create_backend(cfg: Dict[str, Any]) -> TeacherBackend:
    """Instantiate the appropriate teacher backend from config.

    Args:
        cfg: The ``method.kd.teacher`` config dict.

    Returns:
        A backend, optionally wrapped with LoRA if ``peft.enabled: true``.
    """
    teacher_type = cfg.get("type", "snapshot")

    if teacher_type in {"snapshot", "checkpoint"}:
        backend: TeacherBackend = UNetBackend()
        backend.load(cfg)
    elif teacher_type == "sam3":
        from .sam3 import SAM3Backend
        backend = SAM3Backend()
        backend.load(cfg)
    elif teacher_type == "medsam3":
        from .medsam3 import MedSAM3Backend
        backend = MedSAM3Backend()
        backend.load(cfg)
    elif teacher_type == "medsam2":
        from .medsam2 import MedSAM2Backend
        backend = MedSAM2Backend()
        backend.load(cfg)
    else:
        raise ValueError(
            f"Unknown teacher backend type: '{teacher_type}'. "
            "Valid types: snapshot, checkpoint, sam3, medsam3, medsam2"
        )

    # Conditionally wrap with LoRA (no-op if peft.enabled is false/absent)
    from .lora import maybe_wrap_with_lora
    return maybe_wrap_with_lora(backend, cfg)


__all__ = ["TeacherBackend", "UNetBackend", "create_backend"]
