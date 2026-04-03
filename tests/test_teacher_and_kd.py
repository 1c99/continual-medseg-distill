"""Tests for teacher integration, KD mode toggles, state round-trip,
resume mid-task-sequence, and config validation.

Run:  python tests/test_teacher_and_kd.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.methods.teacher import Teacher
from src.methods.distill import DistillMethod
from src.methods.distill_replay_ewc import DistillReplayEWCMethod
from src.methods import create_method
from src.models.factory import build_model
from src.engine.multi_task_trainer import run_task_sequence, _load_progress
from src.engine.evaluator import evaluate
from src.utils.config_validation import validate_config, ConfigError
from src.utils.logging import setup_logger


# ---------------------------------------------------------------------------
# Helpers
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
# A) Teacher load/freeze tests
# ====================================================================

def test_teacher_snapshot_freeze():
    """Teacher.snapshot() creates frozen copy."""
    model = _tiny_model()
    teacher = Teacher()
    assert not teacher.has_model

    teacher.snapshot(model)
    assert teacher.has_model
    for p in teacher.model.parameters():
        assert not p.requires_grad, "Teacher params should be frozen"


def test_teacher_snapshot_independent():
    """Teacher is a deepcopy, not a reference to the original model."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)

    # Modify original model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Teacher should be unchanged
    for (_, p_orig), (_, p_teacher) in zip(
        model.named_parameters(), teacher.model.named_parameters()
    ):
        assert not torch.equal(p_orig, p_teacher), "Teacher should be independent"


def test_teacher_forward_produces_output():
    """Teacher forward pass returns logits with correct shape."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)
    batch = _synthetic_batch()
    logits = teacher(batch["image"])
    assert logits.shape[0] == 2  # batch size
    assert logits.shape[1] == 3  # num classes


def test_teacher_checkpoint_missing_raises():
    """Loading from non-existent checkpoint raises FileNotFoundError."""
    try:
        Teacher(
            teacher_cfg={"type": "checkpoint", "ckpt_path": "/nonexistent/path.pt"},
            model_template=_tiny_model(),
        )
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_teacher_checkpoint_no_template_raises():
    """Loading checkpoint without model_template raises ValueError."""
    try:
        Teacher(
            teacher_cfg={"type": "checkpoint", "ckpt_path": "/nonexistent/path.pt"},
        )
        assert False, "Should have raised ValueError"
    except (ValueError, FileNotFoundError):
        pass


def test_teacher_save_load_roundtrip():
    """Teacher state_dict round-trips correctly."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)

    state = teacher.state_dict()
    assert "teacher_state_dict" in state

    teacher2 = Teacher()
    teacher2.load_state_dict(state, model_template=model)
    assert teacher2.has_model

    batch = _synthetic_batch()
    out1 = teacher(batch["image"])
    out2 = teacher2(batch["image"])
    assert torch.allclose(out1, out2), "Round-trip should produce identical outputs"


def test_teacher_feature_hooks():
    """Teacher with use_features captures intermediate representations."""
    model = _tiny_model()
    # Get a real layer name from the model
    layer_names = [n for n, _ in model.named_modules() if n]
    if not layer_names:
        return  # Skip if no named modules

    prefix = layer_names[0]
    teacher = Teacher(
        teacher_cfg={"use_features": True, "feature_layers": [prefix]}
    )
    teacher.snapshot(model)

    batch = _synthetic_batch()
    _ = teacher(batch["image"])
    assert len(teacher.features) > 0, "Feature hooks should capture features"


# ====================================================================
# B) KD mode toggle tests
# ====================================================================

def test_distill_logit_mode():
    """Logit KD mode produces valid loss."""
    cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": "logit", "weight": 1.0, "temperature": 2.0}})
    method = DistillMethod(cfg)
    model = _tiny_model(cfg)
    batch = _synthetic_batch()

    # No teacher yet — should return CE only
    loss1 = method.training_loss(model, batch, "cpu")
    assert loss1.item() > 0

    # After task update, teacher exists
    method.post_task_update(model)
    loss2 = method.training_loss(model, batch, "cpu")
    assert loss2.item() > 0


def test_distill_weighted_mode():
    """Uncertainty-weighted KD mode runs without error."""
    cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": "weighted", "weight": 1.0, "temperature": 2.0}})
    method = DistillMethod(cfg)
    model = _tiny_model(cfg)
    batch = _synthetic_batch()

    method.post_task_update(model)
    loss = method.training_loss(model, batch, "cpu")
    assert loss.item() > 0


def test_distill_boundary_mode():
    """Boundary-aware KD mode runs without error."""
    cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": "boundary", "weight": 1.0, "temperature": 2.0, "boundary_sigma": 1.0}})
    method = DistillMethod(cfg)
    model = _tiny_model(cfg)
    batch = _synthetic_batch()

    method.post_task_update(model)
    loss = method.training_loss(model, batch, "cpu")
    assert loss.item() > 0


def test_distill_feature_mode():
    """Feature KD mode runs when feature_layers are specified."""
    model = _tiny_model()
    layer_names = [n for n, _ in model.named_modules() if n]
    if not layer_names:
        return

    prefix = layer_names[0]
    cfg = _tiny_cfg(method={
        "name": "distill",
        "kd": {
            "mode": "feature",
            "weight": 1.0,
            "temperature": 2.0,
            "feature_weight": 0.5,
            "teacher": {
                "use_features": True,
                "feature_layers": [prefix],
            },
        },
    })
    method = DistillMethod(cfg)
    batch = _synthetic_batch()

    method.post_task_update(model)
    loss = method.training_loss(model, batch, "cpu")
    assert loss.item() > 0


def test_lwf_creates_distill_method():
    """LwF alias creates a DistillMethod with snapshot teacher."""
    cfg = _tiny_cfg(method={"name": "lwf"})
    method = create_method(cfg)
    assert isinstance(method, DistillMethod), "lwf should create DistillMethod"
    assert not method.teacher.is_external, "LwF teacher must be snapshot (not external)"


def test_lwf_training_loss():
    """LwF produces valid training loss with self-distillation."""
    cfg = _tiny_cfg(method={"name": "lwf", "kd": {"weight": 0.5, "temperature": 2.0}})
    method = create_method(cfg)
    model = _tiny_model(cfg)
    batch = _synthetic_batch()

    # Before any task update, no teacher => CE only
    loss1 = method.training_loss(model, batch, "cpu")
    assert loss1.item() > 0

    # After task update, snapshot teacher exists => CE + KD
    method.post_task_update(model)
    assert method.teacher.has_model, "Teacher should exist after post_task_update"
    loss2 = method.training_loss(model, batch, "cpu")
    assert loss2.item() > 0


def test_lwf_forces_snapshot_teacher():
    """LwF ignores any external teacher config and forces snapshot."""
    cfg = _tiny_cfg(method={
        "name": "lwf",
        "kd": {
            "teacher": {"type": "checkpoint", "ckpt_path": "/fake/path.pt"},
        },
    })
    method = create_method(cfg)
    assert not method.teacher.is_external, "LwF must always use snapshot teacher"


def test_lwf_config_validates():
    """LwF passes config validation."""
    cfg = _tiny_cfg(method={"name": "lwf"})
    errors = validate_config(cfg, strict=False)
    assert len(errors) == 0, f"Unexpected errors: {errors}"


def test_kd_mode_affects_loss():
    """Different KD modes produce different loss values (non-trivially)."""
    model = _tiny_model()
    batch = _synthetic_batch()
    torch.manual_seed(42)

    losses = {}
    for mode in ["logit", "weighted", "boundary"]:
        cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": mode, "weight": 1.0, "temperature": 2.0}})
        method = DistillMethod(cfg)
        method.post_task_update(model)
        losses[mode] = method.training_loss(model, batch, "cpu").item()

    # All modes should produce valid finite losses
    for mode, val in losses.items():
        assert math.isfinite(val) and val > 0, f"{mode} loss invalid: {val}"

    # Different KD modes should produce observably different losses
    unique_losses = set(round(v, 6) for v in losses.values())
    assert len(unique_losses) >= 2, (
        f"Expected at least 2 distinct loss values across KD modes, "
        f"but got: {losses}"
    )


# ====================================================================
# C) Method state round-trip save/load
# ====================================================================

def test_distill_state_roundtrip():
    """DistillMethod save/load preserves teacher state."""
    cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": "logit", "weight": 1.0}})
    method = DistillMethod(cfg)
    model = _tiny_model(cfg)
    method.post_task_update(model)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.pt"
        method.save_state(path, model_template=model)

        method2 = DistillMethod(cfg)
        method2.load_state(path, model_template=model)

        batch = _synthetic_batch()
        out1 = method.teacher(batch["image"])
        out2 = method2.teacher(batch["image"])
        assert torch.allclose(out1, out2), "Teacher should match after round-trip"


def test_distill_replay_ewc_state_roundtrip():
    """DistillReplayEWCMethod save/load preserves teacher+fisher+memory."""
    cfg = _tiny_cfg(method={
        "name": "distill_replay_ewc",
        "kd": {"weight": 0.5, "temperature": 2.0},
        "replay": {"buffer_size": 10, "weight": 1.0},
        "ewc": {"weight": 0.2, "fisher_samples": 2},
    })
    method = DistillReplayEWCMethod(cfg)
    model = _tiny_model(cfg)
    batch = _synthetic_batch()

    # Simulate training to populate memory
    method.training_loss(model, batch, "cpu")

    # Create a simple dataloader for Fisher estimation
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(batch["image"], batch["label"])
    loader = DataLoader(ds)
    # Monkey-patch to return dicts
    orig_iter = loader.__iter__

    class DictLoader:
        def __init__(self, loader):
            self._loader = loader
        def __iter__(self):
            for x, y in self._loader:
                yield {"image": x, "label": y}
        def __len__(self):
            return len(self._loader)

    dict_loader = DictLoader(loader)
    method.post_task_update(model, train_loader=dict_loader)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.pt"
        method.save_state(path, model_template=model)

        method2 = DistillReplayEWCMethod(cfg)
        method2.load_state(path, model_template=model)

        assert method2.teacher.has_model
        assert len(method2.fisher) > 0
        assert len(method2.prev_params) > 0
        assert len(method2.memory) > 0


# ====================================================================
# D) Resume mid-task-sequence
# ====================================================================

def test_resume_from_interrupted_run():
    """Multi-task trainer can resume after interruption."""
    logger = setup_logger("test_resume")
    cfg = _tiny_cfg(method={"name": "finetune"})
    task_configs = [
        {"id": "task_0"},
        {"id": "task_1"},
        {"id": "task_2"},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "multitask"

        # Run first 2 tasks (simulate full run of 2-task sequence to create progress)
        model1 = _tiny_model(cfg)
        method1 = create_method(cfg)
        run_task_sequence(
            model1, method1,
            task_configs[:2],  # Only 2 tasks
            cfg, logger, evaluate, output_dir, dry_run=True,
        )

        # Verify progress file exists
        progress = _load_progress(output_dir)
        assert progress is not None
        assert progress["completed_task_idx"] == 1
        assert progress["last_completed_task_id"] == "task_1"

        # Now resume with full 3-task sequence
        model2 = _tiny_model(cfg)
        method2 = create_method(cfg)
        result = run_task_sequence(
            model2, method2,
            task_configs,  # Full 3 tasks
            cfg, logger, evaluate, output_dir,
            dry_run=True, resume=True,
        )

        assert result["resumed"] is True
        assert len(result["task_order"]) == 3
        assert result["task_order"] == ["task_0", "task_1", "task_2"]
        # Eval history should have all 3 tasks
        assert "task_2" in result["eval_history"]


def test_resume_already_complete():
    """Resume with all tasks done returns immediately."""
    logger = setup_logger("test_resume_done")
    cfg = _tiny_cfg(method={"name": "finetune"})
    task_configs = [{"id": "task_0"}, {"id": "task_1"}]

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "multitask"

        model = _tiny_model(cfg)
        method = create_method(cfg)
        run_task_sequence(
            model, method, task_configs,
            cfg, logger, evaluate, output_dir, dry_run=True,
        )

        # Resume same tasks — should return immediately
        model2 = _tiny_model(cfg)
        method2 = create_method(cfg)
        result = run_task_sequence(
            model2, method2, task_configs,
            cfg, logger, evaluate, output_dir,
            dry_run=True, resume=True,
        )

        assert result["resumed"] is True
        assert len(result["task_order"]) == 2


# ====================================================================
# E) Config validation
# ====================================================================

def test_valid_config_passes():
    """A well-formed config passes validation."""
    cfg = _tiny_cfg(method={"name": "finetune"})
    errors = validate_config(cfg, strict=False)
    assert len(errors) == 0, f"Unexpected errors: {errors}"


def test_missing_out_channels_fails():
    """Missing model.out_channels is caught."""
    cfg = _tiny_cfg()
    del cfg["model"]["out_channels"]
    errors = validate_config(cfg, strict=False)
    assert any("out_channels" in e for e in errors)


def test_invalid_method_name_fails():
    """Invalid method.name is caught."""
    cfg = _tiny_cfg(method={"name": "nonexistent"})
    errors = validate_config(cfg, strict=False)
    assert any("not supported" in e for e in errors)


def test_invalid_kd_mode_fails():
    """Invalid kd.mode is caught."""
    cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": "invalid_mode"}})
    errors = validate_config(cfg, strict=False)
    assert any("invalid" in e.lower() for e in errors)


def test_checkpoint_teacher_needs_path():
    """Teacher type=checkpoint without ckpt_path fails validation."""
    cfg = _tiny_cfg(method={
        "name": "distill",
        "kd": {"mode": "logit", "teacher": {"type": "checkpoint"}},
    })
    errors = validate_config(cfg, strict=False)
    assert any("ckpt_path" in e for e in errors)


def test_feature_mode_needs_layers():
    """Feature KD mode without feature_layers fails validation."""
    cfg = _tiny_cfg(method={
        "name": "distill",
        "kd": {"mode": "feature", "teacher": {"use_features": True}},
    })
    errors = validate_config(cfg, strict=False)
    assert any("feature_layers" in e for e in errors)


def test_strict_mode_raises():
    """Strict validation raises ConfigError."""
    cfg = _tiny_cfg()
    del cfg["model"]["out_channels"]
    try:
        validate_config(cfg, strict=True)
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass


def test_missing_data_source_fails():
    """Missing data.source is caught."""
    cfg = _tiny_cfg()
    del cfg["data"]["source"]
    errors = validate_config(cfg, strict=False)
    assert any("data.source" in e for e in errors)


def test_invalid_loss_type_fails():
    """Invalid train.loss_type is caught."""
    cfg = _tiny_cfg()
    cfg["train"]["loss_type"] = "mse"
    errors = validate_config(cfg, strict=False)
    assert any("loss_type" in e for e in errors)


def test_negative_buffer_size_fails():
    """Negative replay buffer size is caught."""
    cfg = _tiny_cfg(method={"name": "replay", "replay": {"buffer_size": 0}})
    errors = validate_config(cfg, strict=False)
    assert any("buffer_size" in e for e in errors)


# ====================================================================
# Runner
# ====================================================================

if __name__ == "__main__":
    tests = [
        # A) Teacher tests
        ("teacher_snapshot_freeze", test_teacher_snapshot_freeze),
        ("teacher_snapshot_independent", test_teacher_snapshot_independent),
        ("teacher_forward_produces_output", test_teacher_forward_produces_output),
        ("teacher_checkpoint_missing_raises", test_teacher_checkpoint_missing_raises),
        ("teacher_checkpoint_no_template_raises", test_teacher_checkpoint_no_template_raises),
        ("teacher_save_load_roundtrip", test_teacher_save_load_roundtrip),
        ("teacher_feature_hooks", test_teacher_feature_hooks),
        # B) LwF alias tests
        ("lwf_creates_distill_method", test_lwf_creates_distill_method),
        ("lwf_training_loss", test_lwf_training_loss),
        ("lwf_forces_snapshot_teacher", test_lwf_forces_snapshot_teacher),
        ("lwf_config_validates", test_lwf_config_validates),
        # C) KD mode tests
        ("distill_logit_mode", test_distill_logit_mode),
        ("distill_weighted_mode", test_distill_weighted_mode),
        ("distill_boundary_mode", test_distill_boundary_mode),
        ("distill_feature_mode", test_distill_feature_mode),
        ("kd_mode_affects_loss", test_kd_mode_affects_loss),
        # C) State round-trip
        ("distill_state_roundtrip", test_distill_state_roundtrip),
        ("distill_replay_ewc_state_roundtrip", test_distill_replay_ewc_state_roundtrip),
        # D) Resume
        ("resume_from_interrupted_run", test_resume_from_interrupted_run),
        ("resume_already_complete", test_resume_already_complete),
        # E) Config validation
        ("valid_config_passes", test_valid_config_passes),
        ("missing_out_channels_fails", test_missing_out_channels_fails),
        ("invalid_method_name_fails", test_invalid_method_name_fails),
        ("invalid_kd_mode_fails", test_invalid_kd_mode_fails),
        ("checkpoint_teacher_needs_path", test_checkpoint_teacher_needs_path),
        ("feature_mode_needs_layers", test_feature_mode_needs_layers),
        ("strict_mode_raises", test_strict_mode_raises),
        ("missing_data_source_fails", test_missing_data_source_fails),
        ("invalid_loss_type_fails", test_invalid_loss_type_fails),
        ("negative_buffer_size_fails", test_negative_buffer_size_fails),
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
