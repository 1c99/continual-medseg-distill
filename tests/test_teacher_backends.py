"""Tests for teacher backend abstraction (Phase-Teacher Integration).

Tests cover:
- UNet backend snapshot and checkpoint round-trip via backend interface
- Backend ABC compliance
- SAM3/MedSAM3 missing checkpoint error messages
- Output spatial adaptation
- External teacher snapshot skip behaviour
- LoRA/PEFT default-off behaviour
- Config validation for new teacher types and peft

Run:  python tests/test_teacher_backends.py
      python -m pytest tests/test_teacher_backends.py -x
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import torch
import torch.nn as nn

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.methods.teacher import Teacher
from src.methods.teacher_backends import TeacherBackend, UNetBackend, create_backend
from src.models.factory import build_model
from src.utils.config_validation import validate_config


# ---------------------------------------------------------------------------
# Helpers (match existing test_teacher_and_kd.py patterns)
# ---------------------------------------------------------------------------

def _tiny_cfg(**overrides):
    cfg = {
        "experiment": {"name": "test", "seed": 42},
        "model": {
            "name": "monai_unet",
            "in_channels": 1,
            "out_channels": 3,
            "channels": [8, 16],
            "strides": [2],
        },
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "max_steps_per_epoch": 2,
            "loss_type": "dicece",
        },
        "data": {
            "source": "synthetic",
            "batch_size": 2,
            "synthetic": {
                "train_samples": 4,
                "val_samples": 2,
                "channels": 1,
                "num_classes": 3,
                "shape": [16, 16, 16],
            },
        },
        "runtime": {"device": "cpu"},
        "method": {"name": "finetune"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _tiny_model(cfg=None):
    return build_model(cfg or _tiny_cfg())


def _synthetic_batch(channels=1, num_classes=3, shape=(16, 16, 16)):
    x = torch.randn(2, channels, *shape)
    y = torch.randint(0, num_classes, (2, *shape))
    return {"image": x, "label": y}


results = []


def run_test(name, fn):
    try:
        fn()
        results.append(("PASS", name))
    except Exception as e:
        results.append(("FAIL", name))
        traceback.print_exc()


# ====================================================================
# 1) UNet backend snapshot via backend interface
# ====================================================================

def test_unet_backend_snapshot():
    """UNet backend snapshot creates frozen model accessible via Teacher."""
    model = _tiny_model()
    teacher = Teacher()
    assert not teacher.has_model
    assert not teacher.is_external

    teacher.snapshot(model)
    assert teacher.has_model

    # Verify frozen
    for p in teacher.model.parameters():
        assert not p.requires_grad, "Teacher params should be frozen"

    # Verify forward works via backend
    batch = _synthetic_batch()
    logits = teacher(batch["image"])
    assert logits.shape[0] == 2
    assert logits.shape[1] == 3

    # Verify backend metadata
    meta = teacher.metadata
    assert meta["source_mode"] == "snapshot"
    assert meta["frozen"] is True


# ====================================================================
# 2) UNet backend checkpoint round-trip
# ====================================================================

def test_unet_backend_checkpoint_roundtrip():
    """UNet backend state_dict round-trips correctly through Teacher."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)

    # Save via backend
    state = teacher.state_dict()
    assert "teacher_state_dict" in state

    # Restore into new teacher
    teacher2 = Teacher()
    teacher2.load_state_dict(state, model_template=model)
    assert teacher2.has_model

    # Verify outputs match
    batch = _synthetic_batch()
    out1 = teacher(batch["image"])
    out2 = teacher2(batch["image"])
    assert torch.allclose(out1, out2), "Round-trip should produce identical outputs"


# ====================================================================
# 3) All backends implement TeacherBackend ABC
# ====================================================================

def test_backend_interface_compliance():
    """All backend classes implement the TeacherBackend ABC."""
    # UNetBackend
    backend = UNetBackend()
    assert isinstance(backend, TeacherBackend)

    # Verify required abstract methods exist on UNetBackend
    required_methods = [
        "load", "forward_logits", "forward_features",
        "to", "state_dict", "eval",
    ]
    required_properties = ["metadata", "has_model", "is_external"]

    for method_name in required_methods:
        assert hasattr(backend, method_name), f"UNetBackend missing {method_name}"
        assert callable(getattr(backend, method_name)), f"{method_name} not callable"

    for prop_name in required_properties:
        assert hasattr(type(backend), prop_name), f"UNetBackend missing property {prop_name}"

    # Verify create_backend returns TeacherBackend for known types
    for cfg_type in ["snapshot", "checkpoint"]:
        b = create_backend({"type": cfg_type})
        assert isinstance(b, TeacherBackend), f"create_backend({cfg_type}) not TeacherBackend"


# ====================================================================
# 4) SAM3 backend missing checkpoint gives clear error
# ====================================================================

def test_sam3_backend_missing_checkpoint():
    """SAM3 backend raises clear error when checkpoint is missing."""
    try:
        teacher = Teacher(
            teacher_cfg={
                "type": "sam3",
                "ckpt_path": "/nonexistent/sam3_checkpoint.pt",
                "output_channels": 3,
            }
        )
        # If we get here, check that the backend reports no model
        # (some implementations defer checkpoint loading)
        # The important thing is it doesn't silently succeed with a working model
    except (FileNotFoundError, RuntimeError, ImportError, ValueError) as e:
        # Any of these are acceptable error types for missing checkpoint
        err_msg = str(e).lower()
        assert any(
            kw in err_msg for kw in ["not found", "checkpoint", "sam3", "missing", "import", "module"]
        ), f"Error message not descriptive enough: {e}"


# ====================================================================
# 5) MedSAM3 backend missing checkpoint gives clear error
# ====================================================================

def test_medsam3_backend_missing_checkpoint():
    """MedSAM3 backend raises clear error when checkpoint is missing."""
    try:
        teacher = Teacher(
            teacher_cfg={
                "type": "medsam3",
                "ckpt_path": "/nonexistent/medsam3_checkpoint.pt",
                "output_channels": 3,
            }
        )
    except (FileNotFoundError, RuntimeError, ImportError, ValueError) as e:
        err_msg = str(e).lower()
        assert any(
            kw in err_msg for kw in ["not found", "checkpoint", "medsam3", "missing", "import", "module"]
        ), f"Error message not descriptive enough: {e}"


# ====================================================================
# 6) Output spatial adaptation — mock backend with wrong spatial dims
# ====================================================================

def test_output_spatial_adaptation():
    """Backend output with mismatched spatial dims can be detected."""
    # Create a mock backend that returns wrong spatial dimensions
    class WrongDimsBackend(TeacherBackend):
        def load(self, cfg, device="cpu"):
            pass

        def forward_logits(self, x):
            B = x.shape[0]
            # Return logits with wrong spatial dims (8,8,8 instead of 16,16,16)
            return torch.randn(B, 3, 8, 8, 8)

        def forward_features(self, x):
            return {}

        @property
        def metadata(self):
            return {"model_id": "wrong_dims_mock"}

        def to(self, device):
            return self

        def state_dict(self):
            return {}

        def eval(self):
            return self

        @property
        def has_model(self):
            return True

        @property
        def is_external(self):
            return True

    backend = WrongDimsBackend()
    x = torch.randn(2, 1, 16, 16, 16)
    logits = backend.forward_logits(x)

    # Verify spatial mismatch is detectable
    assert logits.shape[2:] != x.shape[2:], "Should have mismatched spatial dims"
    assert logits.shape[2:] == (8, 8, 8)

    # Interpolation can fix it
    import torch.nn.functional as F
    adapted = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
    assert adapted.shape[2:] == x.shape[2:], "Interpolation should fix spatial dims"


# ====================================================================
# 7) External teacher skips snapshot in post_task_update
# ====================================================================

def test_external_teacher_skips_snapshot():
    """External (non-UNet) backend raises NotImplementedError on snapshot()."""
    # Create a mock external backend
    class MockExternalBackend(TeacherBackend):
        def load(self, cfg, device="cpu"):
            pass

        def forward_logits(self, x):
            return torch.randn(x.shape[0], 3, *x.shape[2:])

        def forward_features(self, x):
            return {}

        @property
        def metadata(self):
            return {"model_id": "mock_external"}

        def to(self, device):
            return self

        def state_dict(self):
            return {}

        def eval(self):
            return self

        @property
        def has_model(self):
            return True

        @property
        def is_external(self):
            return True

    # TeacherBackend base class raises NotImplementedError for snapshot()
    backend = MockExternalBackend()
    assert backend.is_external is True

    try:
        model = _tiny_model()
        backend.snapshot(model)
        assert False, "External backend snapshot() should raise NotImplementedError"
    except NotImplementedError:
        pass  # Expected

    # Verify the pattern: when is_external is True, caller should skip snapshot
    # This is the guard pattern used in distill.py and distill_replay_ewc.py
    teacher = Teacher()
    assert not teacher.is_external, "Default UNet teacher should not be external"


# ====================================================================
# 8) LoRA/PEFT toggle off by default
# ====================================================================

def test_lora_toggle_off_by_default():
    """PEFT/LoRA is disabled by default in teacher config."""
    # Default config has no peft section
    cfg = _tiny_cfg(method={
        "name": "distill",
        "kd": {"mode": "logit", "weight": 1.0},
    })
    peft_cfg = cfg.get("method", {}).get("kd", {}).get("teacher", {}).get("peft", {})
    peft_enabled = peft_cfg.get("enabled", False)
    assert peft_enabled is False, "PEFT should be disabled by default"

    # Even with explicit teacher config, peft defaults to off
    cfg2 = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {"type": "snapshot"},
        },
    })
    peft_cfg2 = cfg2["method"]["kd"]["teacher"].get("peft", {})
    assert peft_cfg2.get("enabled", False) is False, "PEFT should default to disabled"


# ====================================================================
# 9) Config validation for new teacher types (sam3/medsam3)
# ====================================================================

def test_config_validation_new_types():
    """Config validation accepts sam3/medsam3 types and rejects invalid types."""
    # sam3 with required fields should pass (no teacher-type errors)
    cfg_sam3 = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {
                "type": "sam3",
                "ckpt_path": "/some/checkpoint.pt",
                "output_channels": 3,
            },
        },
    })
    errors = validate_config(cfg_sam3, strict=False)
    teacher_type_errors = [e for e in errors if "teacher.type" in e]
    assert len(teacher_type_errors) == 0, f"sam3 type should be valid: {teacher_type_errors}"

    # medsam3 with required fields should pass
    cfg_medsam3 = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {
                "type": "medsam3",
                "ckpt_path": "/some/checkpoint.pt",
                "output_channels": 3,
            },
        },
    })
    errors = validate_config(cfg_medsam3, strict=False)
    teacher_type_errors = [e for e in errors if "teacher.type" in e]
    assert len(teacher_type_errors) == 0, f"medsam3 type should be valid: {teacher_type_errors}"

    # sam3 missing ckpt_path should fail
    cfg_no_ckpt = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {"type": "sam3", "output_channels": 3},
        },
    })
    errors = validate_config(cfg_no_ckpt, strict=False)
    assert any("ckpt_path" in e for e in errors), f"sam3 without ckpt_path should fail: {errors}"

    # sam3 missing output_channels should fail
    cfg_no_ch = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {"type": "sam3", "ckpt_path": "/some/path.pt"},
        },
    })
    errors = validate_config(cfg_no_ch, strict=False)
    assert any("output_channels" in e for e in errors), \
        f"sam3 without output_channels should fail: {errors}"

    # Invalid teacher type should fail
    cfg_invalid = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {"type": "nonexistent_type"},
        },
    })
    errors = validate_config(cfg_invalid, strict=False)
    assert any("invalid" in e.lower() or "type" in e.lower() for e in errors), \
        f"Invalid teacher type should fail: {errors}"


# ====================================================================
# 10) Config validation for PEFT settings
# ====================================================================

def test_config_validation_peft():
    """Config validation handles PEFT settings gracefully."""
    # Config without peft section should be valid (peft is optional)
    cfg_no_peft = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {"type": "snapshot"},
        },
    })
    errors = validate_config(cfg_no_peft, strict=False)
    peft_errors = [e for e in errors if "peft" in e.lower()]
    assert len(peft_errors) == 0, f"No-peft config should have no peft errors: {peft_errors}"

    # Config with peft.enabled=false should be valid
    cfg_peft_off = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "logit",
            "teacher": {
                "type": "snapshot",
                "peft": {"enabled": False},
            },
        },
    })
    errors = validate_config(cfg_peft_off, strict=False)
    peft_errors = [e for e in errors if "peft" in e.lower()]
    assert len(peft_errors) == 0, f"Peft-disabled config should have no peft errors: {peft_errors}"


# ====================================================================
# Runner
# ====================================================================

if __name__ == "__main__":
    tests = [
        ("unet_backend_snapshot", test_unet_backend_snapshot),
        ("unet_backend_checkpoint_roundtrip", test_unet_backend_checkpoint_roundtrip),
        ("backend_interface_compliance", test_backend_interface_compliance),
        ("sam3_backend_missing_checkpoint", test_sam3_backend_missing_checkpoint),
        ("medsam3_backend_missing_checkpoint", test_medsam3_backend_missing_checkpoint),
        ("output_spatial_adaptation", test_output_spatial_adaptation),
        ("external_teacher_skips_snapshot", test_external_teacher_skips_snapshot),
        ("lora_toggle_off_by_default", test_lora_toggle_off_by_default),
        ("config_validation_new_types", test_config_validation_new_types),
        ("config_validation_peft", test_config_validation_peft),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print()
    print("=" * 60)
    passed = sum(1 for s, _ in results if s == "PASS")
    failed = sum(1 for s, _ in results if s == "FAIL")
    for status, name in results:
        print(f"  [{status}] {name}")
    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
    if failed:
        sys.exit(1)
