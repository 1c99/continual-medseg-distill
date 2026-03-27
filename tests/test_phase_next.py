"""Comprehensive tests for Phase-Next features:
- P0-1: Teacher adapter (forward_logits, forward_features, metadata)
- P0-2: Data pipeline (label remap, validate_subject)
- P0-3: Reliability (early stopping, worker seeding, resume lineage)
- P1-4: Metrics (BWT/FWT computation)
- P1-5: Config quality (hash, path checks, resolved config)
- P1-6: Integration (per-method, smoke matrix)

Run: python tests/test_phase_next.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.methods.teacher import Teacher
from src.methods.distill import DistillMethod
from src.methods import create_method
from src.models.factory import build_model
from src.data.label_remap import LabelRemapper, remap_from_config
from src.data.totalseg import TotalSegmentatorDataset
from src.data.brats21 import Brats21Dataset
from src.data.acdc import ACDCDataset
from src.engine.trainer import EarlyStopper
from src.engine.multi_task_trainer import compute_forgetting, run_task_sequence, _load_progress
from src.engine.evaluator import evaluate
from src.utils.reproducibility import worker_init_fn, set_deterministic_mode
from src.utils.config_validation import (
    validate_config, compute_config_hash, save_resolved_config, validate_paths, ConfigError,
)
from src.utils.logging import setup_logger

results = []


def run_test(name, fn):
    try:
        fn()
        results.append(("PASS", name))
    except Exception:
        results.append(("FAIL", name))
        traceback.print_exc()


def _tiny_cfg(**overrides):
    cfg = {
        "experiment": {"name": "test", "seed": 42},
        "model": {
            "name": "monai_unet", "in_channels": 1, "out_channels": 3,
            "channels": [8, 16], "strides": [2],
        },
        "train": {"epochs": 1, "lr": 0.001, "max_steps_per_epoch": 2, "loss_type": "dicece"},
        "data": {
            "source": "synthetic", "batch_size": 2,
            "synthetic": {"train_samples": 4, "val_samples": 2, "channels": 1, "num_classes": 3, "shape": [16, 16, 16]},
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


def _synthetic_batch():
    return {"image": torch.randn(2, 1, 16, 16, 16), "label": torch.randint(0, 3, (2, 16, 16, 16))}


# ====================================================================
# P0-1: Teacher Adapter
# ====================================================================

def test_teacher_forward_logits():
    """forward_logits returns same as forward."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)
    x = torch.randn(1, 1, 16, 16, 16)
    out1 = teacher.forward(x)
    out2 = teacher.forward_logits(x)
    assert torch.equal(out1, out2)


def test_teacher_forward_features_requires_flag():
    """forward_features raises if use_features=False."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)
    try:
        teacher.forward_features(torch.randn(1, 1, 16, 16, 16))
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "use_features" in str(e)


def test_teacher_forward_features_returns_dict():
    """forward_features returns feature dict when hooks are set."""
    model = _tiny_model()
    layers = [n for n, _ in model.named_modules() if n]
    if not layers:
        return
    teacher = Teacher(teacher_cfg={"use_features": True, "feature_layers": [layers[0]]})
    teacher.snapshot(model)
    feats = teacher.forward_features(torch.randn(1, 1, 16, 16, 16))
    assert isinstance(feats, dict)
    assert len(feats) > 0


def test_teacher_metadata_snapshot():
    """Metadata populated after snapshot."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)
    meta = teacher.metadata
    assert meta["source_mode"] == "snapshot"
    assert meta["frozen"] is True
    assert meta["ckpt_hash"] is None


def test_teacher_metadata_checkpoint():
    """Metadata includes ckpt_hash after checkpoint load."""
    model = _tiny_model()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "teacher.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        teacher = Teacher(
            teacher_cfg={"type": "checkpoint", "ckpt_path": str(ckpt_path), "model_id": "test_model"},
            model_template=model,
        )
        meta = teacher.metadata
        assert meta["source_mode"] == "checkpoint"
        assert meta["ckpt_hash"] is not None
        assert len(meta["ckpt_hash"]) == 16
        assert meta["model_id"] == "test_model"


def test_teacher_state_includes_metadata():
    """state_dict includes metadata for persistence."""
    model = _tiny_model()
    teacher = Teacher()
    teacher.snapshot(model)
    state = teacher.state_dict()
    assert "teacher_metadata" in state


# ====================================================================
# P0-2: Data Pipeline
# ====================================================================

def test_label_remapper_basic():
    """LabelRemapper applies mapping correctly."""
    remap = LabelRemapper({4: 3})
    arr = np.array([0, 1, 2, 4, 0])
    result = remap(arr)
    assert list(result) == [0, 1, 2, 3, 0]


def test_label_remapper_torch():
    """LabelRemapper works with torch tensors."""
    remap = LabelRemapper({4: 3})
    t = torch.tensor([0, 1, 2, 4, 4])
    result = remap(t)
    assert result.tolist() == [0, 1, 2, 3, 3]


def test_label_remapper_strict_unmapped():
    """Strict mode raises on unmapped values."""
    remap = LabelRemapper({4: 3}, strict=True)
    arr = np.array([0, 1, 2, 4, 5])  # 5 is unmapped and not a target
    try:
        remap(arr)
        assert False, "Should have raised"
    except ValueError as e:
        assert "Unmapped" in str(e)


def test_label_remapper_verify_domain():
    """verify_domain checks label range."""
    remap = LabelRemapper({4: 3})
    arr = np.array([0, 1, 2, 3])
    assert remap.verify_domain(arr, {0, 1, 2, 3})
    assert not remap.verify_domain(arr, {0, 1})


def test_remap_from_config_none():
    """remap_from_config returns None when not configured."""
    assert remap_from_config(None) is None
    assert remap_from_config({}) is None


def test_remap_from_config_creates():
    """remap_from_config creates remapper from dict."""
    remap = remap_from_config({"4": "3", "2": "1"})
    assert remap is not None
    assert remap.mapping == {4: 3, 2: 1}


def test_totalseg_validate_subject_missing():
    """TotalSegmentatorDataset.validate_subject catches missing dirs."""
    result = TotalSegmentatorDataset.validate_subject("/nonexistent", "s0001")
    assert not result["valid"]
    assert len(result["errors"]) > 0


def test_brats21_validate_subject_missing():
    """Brats21Dataset.validate_subject catches missing files."""
    result = Brats21Dataset.validate_subject("/nonexistent", "BraTS_00000", layout="flat")
    assert not result["valid"]


def test_acdc_validate_subject_bad_format():
    """ACDCDataset.validate_subject catches bad sid format."""
    result = ACDCDataset.validate_subject("/nonexistent", "badformat")
    assert not result["valid"]
    assert "format" in result["errors"][0].lower()


# ====================================================================
# P0-3: Reliability
# ====================================================================

def test_early_stopper_disabled():
    """EarlyStopper with patience=0 never stops."""
    es = EarlyStopper(patience=0)
    assert not es.enabled
    assert not es.step({"dice_mean": 0.5})


def test_early_stopper_triggers():
    """EarlyStopper triggers after patience epochs without improvement."""
    es = EarlyStopper(patience=3, metric="dice_mean", mode="max")
    assert not es.step({"dice_mean": 0.5})
    assert not es.step({"dice_mean": 0.4})  # worse, wait=1
    assert not es.step({"dice_mean": 0.3})  # worse, wait=2
    assert es.step({"dice_mean": 0.2})      # worse, wait=3 → stop


def test_early_stopper_resets_on_improvement():
    """EarlyStopper resets counter on improvement."""
    es = EarlyStopper(patience=2, metric="dice_mean", mode="max")
    es.step({"dice_mean": 0.5})
    es.step({"dice_mean": 0.4})  # wait=1
    es.step({"dice_mean": 0.6})  # improvement, wait=0
    assert not es.step({"dice_mean": 0.5})  # wait=1
    assert es.step({"dice_mean": 0.4})      # wait=2 → stop


def test_worker_init_fn_runs():
    """worker_init_fn doesn't crash."""
    # Can't fully test without DataLoader workers, but verify it doesn't error
    worker_init_fn(0)
    worker_init_fn(1)


def test_set_deterministic_mode():
    """set_deterministic_mode sets seed and flags."""
    set_deterministic_mode(42)
    assert torch.backends.cudnn.deterministic is True


def test_resume_lineage_tracks_count():
    """Resume increments resume_count in progress file."""
    logger = setup_logger("test_lineage")
    cfg = _tiny_cfg(method={"name": "finetune"})
    tasks = [{"id": "t0"}, {"id": "t1"}, {"id": "t2"}]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mt"

        # First run: 2 tasks
        m1 = _tiny_model(cfg)
        me1 = create_method(cfg)
        run_task_sequence(m1, me1, tasks[:2], cfg, logger, evaluate, out, dry_run=True)

        progress = _load_progress(out)
        assert progress["resume_count"] == 0

        # Resume: adds task 3
        m2 = _tiny_model(cfg)
        me2 = create_method(cfg)
        run_task_sequence(m2, me2, tasks, cfg, logger, evaluate, out, dry_run=True, resume=True)

        progress2 = _load_progress(out)
        assert progress2["resume_count"] == 1


# ====================================================================
# P1-4: Metrics (BWT/FWT)
# ====================================================================

def test_bwt_computation():
    """BWT computed correctly from eval history."""
    # Task 0: dice=0.8 on task 0
    # Task 1: dice=0.6 on task 0, 0.7 on task 1
    # BWT(task 0) = R_{1,0} - R_{0,0} = 0.6 - 0.8 = -0.2
    history = {
        "t0": {"t0": {"dice_mean": 0.8}},
        "t1": {"t0": {"dice_mean": 0.6}, "t1": {"dice_mean": 0.7}},
    }
    result = compute_forgetting(history, ["t0", "t1"])
    assert abs(result["mean_bwt"] - (-0.2)) < 1e-6
    assert abs(result["bwt_per_task"]["t0"] - (-0.2)) < 1e-6


def test_fwt_computation():
    """FWT computed correctly from eval history."""
    # Task 0 trained, eval on task 1: dice=0.3 (forward transfer)
    # Task 1 trained, eval on task 1: dice=0.7
    history = {
        "t0": {"t0": {"dice_mean": 0.8}, "t1": {"dice_mean": 0.3}},
        "t1": {"t0": {"dice_mean": 0.6}, "t1": {"dice_mean": 0.7}},
    }
    result = compute_forgetting(history, ["t0", "t1"])
    assert abs(result["mean_fwt"] - 0.3) < 1e-6


def test_forgetting_sign_convention():
    """Positive forgetting means performance dropped."""
    history = {
        "t0": {"t0": {"dice_mean": 0.8}},
        "t1": {"t0": {"dice_mean": 0.6}, "t1": {"dice_mean": 0.7}},
    }
    result = compute_forgetting(history, ["t0", "t1"])
    # forgetting = R_{0,0} - R_{1,0} = 0.8 - 0.6 = 0.2 (positive = perf dropped)
    assert result["per_task"]["t0"] > 0
    # BWT = R_{1,0} - R_{0,0} = -0.2 (negative = forgot)
    assert result["bwt_per_task"]["t0"] < 0


# ====================================================================
# P1-5: Config Quality
# ====================================================================

def test_config_hash_deterministic():
    """Same config produces same hash."""
    cfg = _tiny_cfg()
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 16


def test_config_hash_differs():
    """Different configs produce different hashes."""
    cfg1 = _tiny_cfg()
    cfg2 = _tiny_cfg()
    cfg2["train"]["lr"] = 0.01
    assert compute_config_hash(cfg1) != compute_config_hash(cfg2)


def test_save_resolved_config():
    """save_resolved_config writes YAML + hash files."""
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = save_resolved_config(cfg, tmp)
        assert yaml_path.exists()
        assert (Path(tmp) / "config_hash.txt").exists()
        hash_content = (Path(tmp) / "config_hash.txt").read_text().strip()
        assert hash_content == compute_config_hash(cfg)


def test_validate_paths_missing_root():
    """validate_paths catches missing data root."""
    cfg = _tiny_cfg()
    cfg["data"]["source"] = "totalseg"
    cfg["data"]["totalseg"] = {"root": "/nonexistent/path"}
    errors = validate_paths(cfg)
    assert any("does not exist" in e for e in errors)


def test_validate_paths_synthetic_ok():
    """validate_paths skips synthetic source."""
    cfg = _tiny_cfg()
    errors = validate_paths(cfg)
    assert len(errors) == 0


# ====================================================================
# P1-6: Integration / Smoke Matrix
# ====================================================================

def test_integration_finetune():
    """Finetune method: 1-epoch train produces checkpoint."""
    cfg = _tiny_cfg(method={"name": "finetune"})
    model = _tiny_model(cfg)
    method = create_method(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        cfg["output"] = {"dir": tmp, "best_metric": "voxel_acc"}
        from src.engine.trainer import train
        from src.data.registry import create_loaders
        train_loader, val_loader = create_loaders(cfg)
        train(model, method, train_loader, cfg, setup_logger("ft"), dry_run=True,
              val_loader=val_loader, evaluate_fn=evaluate)
        assert (Path(tmp) / "checkpoints" / "last.pt").exists()
        assert (Path(tmp) / "metrics.csv").exists()


def test_integration_replay():
    """Replay method: training populates memory."""
    cfg = _tiny_cfg(method={"name": "replay", "replay": {"buffer_size": 10, "weight": 1.0}})
    model = _tiny_model(cfg)
    method = create_method(cfg)
    batch = _synthetic_batch()
    method.training_loss(model, batch, "cpu")
    assert len(method.memory) > 0


def test_integration_distill():
    """Distill method: post_task_update creates teacher."""
    cfg = _tiny_cfg(method={"name": "distill", "kd": {"mode": "logit", "weight": 1.0}})
    model = _tiny_model(cfg)
    method = create_method(cfg)
    assert not method.teacher.has_model
    method.post_task_update(model)
    assert method.teacher.has_model


def test_integration_distill_replay_ewc():
    """DistillReplayEWC: post_task_update populates teacher+fisher+memory."""
    cfg = _tiny_cfg(method={
        "name": "distill_replay_ewc",
        "kd": {"weight": 0.5}, "replay": {"buffer_size": 10}, "ewc": {"weight": 0.2, "fisher_samples": 1},
    })
    model = _tiny_model(cfg)
    method = create_method(cfg)
    batch = _synthetic_batch()
    method.training_loss(model, batch, "cpu")

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(batch["image"], batch["label"])
    class DL:
        def __init__(self, ds): self._ds = ds
        def __iter__(self):
            for x, y in DataLoader(self._ds): yield {"image": x, "label": y}
        def __len__(self): return len(self._ds)

    method.post_task_update(model, train_loader=DL(ds))
    assert method.teacher.has_model
    assert len(method.fisher) > 0
    assert len(method.memory) > 0


def test_smoke_matrix_2task_all_methods():
    """Smoke: 4 methods × 2-task synthetic sequence all complete."""
    logger = setup_logger("smoke_matrix")
    for method_name in ["finetune", "replay", "distill", "distill_replay_ewc"]:
        method_cfg = {"name": method_name}
        if method_name == "replay":
            method_cfg["replay"] = {"buffer_size": 10, "weight": 1.0}
        elif method_name == "distill":
            method_cfg["kd"] = {"mode": "logit", "weight": 1.0}
        elif method_name == "distill_replay_ewc":
            method_cfg["kd"] = {"weight": 0.5}
            method_cfg["replay"] = {"buffer_size": 10}
            method_cfg["ewc"] = {"weight": 0.1, "fisher_samples": 1}

        cfg = _tiny_cfg(method=method_cfg)
        model = _tiny_model(cfg)
        method = create_method(cfg)
        tasks = [{"id": "t0"}, {"id": "t1"}]

        with tempfile.TemporaryDirectory() as tmp:
            result = run_task_sequence(
                model, method, tasks, cfg, logger, evaluate, Path(tmp) / "mt", dry_run=True,
            )
            assert len(result["task_order"]) == 2, f"{method_name}: expected 2 tasks"
            assert "t0" in result["eval_history"]["t1"], f"{method_name}: missing eval"
            assert "mean_bwt" in result["forgetting"], f"{method_name}: missing BWT"


def test_smoke_ortho_lora_2task():
    """End-to-end A→B with orthogonal LoRA + distill_replay_ewc on synthetic data."""
    logger = setup_logger("smoke_ortho_lora")
    cfg = _tiny_cfg(
        method={
            "name": "distill_replay_ewc",
            "kd": {"weight": 0.5},
            "replay": {"buffer_size": 10},
            "ewc": {"weight": 0.1, "fisher_samples": 1},
        },
        model={
            "name": "monai_unet", "in_channels": 1, "out_channels": 3,
            "channels": [8, 16], "strides": [2],
            "lora": {
                "enabled": True,
                "mode": "orthogonal",
                "rank": 4,
                "alpha": 8.0,
                "target_modules": ["conv.unit"],
                "ortho_lambda": 0.1,
            },
        },
    )

    model = build_model(cfg)
    method = create_method(cfg)
    tasks = [{"id": "tA"}, {"id": "tB"}]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "ortho_lora"
        result = run_task_sequence(
            model, method, tasks, cfg, logger, evaluate, out_dir, dry_run=True,
        )
        assert len(result["task_order"]) == 2
        assert "mean_bwt" in result["forgetting"]

        # Verify per-task LoRA adapter checkpoints exist
        lora_a = out_dir / "tA" / "checkpoints" / "lora_state_tA.pt"
        lora_b = out_dir / "tB" / "checkpoints" / "lora_state_tB.pt"
        assert lora_a.exists(), f"Missing LoRA checkpoint: {lora_a}"
        assert lora_b.exists(), f"Missing LoRA checkpoint: {lora_b}"

        # Verify saved LoRA states are non-empty dicts
        state_a = torch.load(lora_a, map_location="cpu", weights_only=False)
        state_b = torch.load(lora_b, map_location="cpu", weights_only=False)
        assert len(state_a) > 0, "LoRA state A should be non-empty"
        assert len(state_b) > 0, "LoRA state B should be non-empty"

        # Verify method accumulated prev_lora_states
        assert len(method.prev_lora_states) >= 1, \
            "Method should have at least 1 previous LoRA state after 2 tasks"


# ====================================================================
# Runner
# ====================================================================

if __name__ == "__main__":
    tests = [
        # P0-1: Teacher adapter
        ("teacher_forward_logits", test_teacher_forward_logits),
        ("teacher_forward_features_requires_flag", test_teacher_forward_features_requires_flag),
        ("teacher_forward_features_returns_dict", test_teacher_forward_features_returns_dict),
        ("teacher_metadata_snapshot", test_teacher_metadata_snapshot),
        ("teacher_metadata_checkpoint", test_teacher_metadata_checkpoint),
        ("teacher_state_includes_metadata", test_teacher_state_includes_metadata),
        # P0-2: Data pipeline
        ("label_remapper_basic", test_label_remapper_basic),
        ("label_remapper_torch", test_label_remapper_torch),
        ("label_remapper_strict_unmapped", test_label_remapper_strict_unmapped),
        ("label_remapper_verify_domain", test_label_remapper_verify_domain),
        ("remap_from_config_none", test_remap_from_config_none),
        ("remap_from_config_creates", test_remap_from_config_creates),
        ("totalseg_validate_subject_missing", test_totalseg_validate_subject_missing),
        ("brats21_validate_subject_missing", test_brats21_validate_subject_missing),
        ("acdc_validate_subject_bad_format", test_acdc_validate_subject_bad_format),
        # P0-3: Reliability
        ("early_stopper_disabled", test_early_stopper_disabled),
        ("early_stopper_triggers", test_early_stopper_triggers),
        ("early_stopper_resets_on_improvement", test_early_stopper_resets_on_improvement),
        ("worker_init_fn_runs", test_worker_init_fn_runs),
        ("set_deterministic_mode", test_set_deterministic_mode),
        ("resume_lineage_tracks_count", test_resume_lineage_tracks_count),
        # P1-4: Metrics
        ("bwt_computation", test_bwt_computation),
        ("fwt_computation", test_fwt_computation),
        ("forgetting_sign_convention", test_forgetting_sign_convention),
        # P1-5: Config quality
        ("config_hash_deterministic", test_config_hash_deterministic),
        ("config_hash_differs", test_config_hash_differs),
        ("save_resolved_config", test_save_resolved_config),
        ("validate_paths_missing_root", test_validate_paths_missing_root),
        ("validate_paths_synthetic_ok", test_validate_paths_synthetic_ok),
        # P1-6: Integration
        ("integration_finetune", test_integration_finetune),
        ("integration_replay", test_integration_replay),
        ("integration_distill", test_integration_distill),
        ("integration_distill_replay_ewc", test_integration_distill_replay_ewc),
        ("smoke_matrix_2task_all_methods", test_smoke_matrix_2task_all_methods),
        ("smoke_ortho_lora_2task", test_smoke_ortho_lora_2task),
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
