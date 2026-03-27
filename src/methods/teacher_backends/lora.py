"""LoRA wrapper for teacher backends using Hugging Face PEFT.

This module provides a config-toggleable LoRA decorator that can wrap any
``TeacherBackend`` and apply parameter-efficient fine-tuning adapters to
the underlying teacher model.

Disabled by default (``method.kd.teacher.peft.enabled: false``).
When disabled, the wrapped backend behaves identically to the original.

Requires the ``peft`` package (``pip install peft>=0.7``), which is only
imported when ``peft.enabled: true``.

Config keys (under ``method.kd.teacher.peft``):
    enabled: bool   — master toggle (default: false)
    type: str       — adapter type (default: lora)
    rank: int       — LoRA rank (default: 8)
    alpha: int      — LoRA alpha scaling (default: 16)
    target_modules: list[str] — module name patterns to apply LoRA to
                                (default: ["q_proj", "v_proj"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from .base import TeacherBackend

logger = logging.getLogger(__name__)


def _get_inner_model(backend: TeacherBackend) -> nn.Module | None:
    """Extract the nn.Module from a backend for LoRA patching."""
    # UNetBackend exposes .model directly
    if hasattr(backend, "model") and isinstance(getattr(backend, "model"), nn.Module):
        return backend.model
    # SAM3/MedSAM3 expose _model
    if hasattr(backend, "_model") and isinstance(backend._model, nn.Module):
        return backend._model
    return None


def _set_inner_model(backend: TeacherBackend, model: nn.Module) -> None:
    """Replace the nn.Module inside a backend after LoRA wrapping."""
    if hasattr(backend, "_model"):
        backend._model = model
    else:
        raise AttributeError(
            f"Cannot set inner model on {type(backend).__name__}. "
            "LoRA wrapper only supports backends with a _model attribute."
        )


class LoRATeacherBackend(TeacherBackend):
    """Decorator that adds PEFT LoRA adapters to any TeacherBackend.

    When ``peft.enabled`` is false (default), all calls delegate directly
    to the wrapped backend with zero overhead.

    When enabled, LoRA adapters are injected into the inner model's
    ``target_modules`` after the base backend finishes loading.
    """

    def __init__(self, inner: TeacherBackend, peft_cfg: Dict[str, Any]) -> None:
        self._inner = inner
        self._peft_cfg = peft_cfg
        self._enabled = bool(peft_cfg.get("enabled", False))
        self._lora_applied = False

    # -- TeacherBackend interface (delegated) --

    def load(self, cfg: Dict[str, Any], device: str = "cpu") -> None:
        # Inner backend is already loaded by create_backend; nothing to do
        pass

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner.forward_logits(x)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._inner.forward_features(x)

    @property
    def metadata(self) -> Dict[str, Any]:
        meta = dict(self._inner.metadata)
        if self._enabled:
            meta["peft"] = {
                "type": self._peft_cfg.get("type", "lora"),
                "rank": self._peft_cfg.get("rank", 8),
                "alpha": self._peft_cfg.get("alpha", 16),
                "target_modules": self._peft_cfg.get("target_modules", ["q_proj", "v_proj"]),
                "applied": self._lora_applied,
            }
        return meta

    def to(self, device) -> "LoRATeacherBackend":
        self._inner.to(device)
        return self

    def state_dict(self) -> Dict[str, Any]:
        state = self._inner.state_dict()
        if self._enabled and self._lora_applied:
            # Save only LoRA parameters
            model = _get_inner_model(self._inner)
            if model is not None:
                lora_state = {
                    k: v for k, v in model.state_dict().items() if "lora_" in k
                }
                state["lora_state_dict"] = lora_state
        return state

    def eval(self) -> "LoRATeacherBackend":
        self._inner.eval()
        return self

    def snapshot(self, model: nn.Module) -> None:
        self._inner.snapshot(model)
        if self._enabled:
            self._apply_lora()

    @property
    def has_model(self) -> bool:
        return self._inner.has_model

    @property
    def is_external(self) -> bool:
        return self._inner.is_external

    # -- LoRA-specific --

    def apply_lora_if_enabled(self) -> None:
        """Apply LoRA adapters to the inner model if peft.enabled is true.

        Called after the inner backend has finished loading its model.
        Safe to call multiple times (no-op if already applied or disabled).
        """
        if not self._enabled or self._lora_applied:
            return
        self._apply_lora()

    def _apply_lora(self) -> None:
        """Inject PEFT LoRA adapters into the inner model."""
        model = _get_inner_model(self._inner)
        if model is None:
            logger.warning(
                "LoRATeacherBackend: cannot apply LoRA — inner backend has no model loaded"
            )
            return

        # Lazy import — peft is only needed when enabled
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "peft package is required when method.kd.teacher.peft.enabled=true. "
                "Install with: pip install 'peft>=0.7'"
            )

        peft_type = self._peft_cfg.get("type", "lora")
        if peft_type != "lora":
            raise ValueError(
                f"Unsupported peft.type='{peft_type}'. Currently only 'lora' is supported."
            )

        rank = int(self._peft_cfg.get("rank", 8))
        alpha = int(self._peft_cfg.get("alpha", 16))
        target_modules = list(self._peft_cfg.get("target_modules", ["q_proj", "v_proj"]))

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )

        logger.info(
            f"LoRATeacherBackend: applying LoRA (rank={rank}, alpha={alpha}, "
            f"targets={target_modules})"
        )

        wrapped_model = get_peft_model(model, lora_config)
        _set_inner_model(self._inner, wrapped_model)
        self._lora_applied = True

        # Count LoRA parameters
        lora_params = sum(
            p.numel() for n, p in wrapped_model.named_parameters() if "lora_" in n
        )
        total_params = sum(p.numel() for p in wrapped_model.parameters())
        logger.info(
            f"LoRATeacherBackend: {lora_params:,} LoRA params / {total_params:,} total "
            f"({100 * lora_params / max(total_params, 1):.1f}%)"
        )


def maybe_wrap_with_lora(
    backend: TeacherBackend, teacher_cfg: Dict[str, Any]
) -> TeacherBackend:
    """Conditionally wrap a backend with LoRA if peft config is present and enabled.

    This is the main entry point — called from ``create_backend()``.
    Returns the original backend unchanged if peft is not enabled.
    """
    peft_cfg = teacher_cfg.get("peft", {})
    if not peft_cfg.get("enabled", False):
        return backend

    wrapper = LoRATeacherBackend(backend, peft_cfg)
    wrapper.apply_lora_if_enabled()
    return wrapper
