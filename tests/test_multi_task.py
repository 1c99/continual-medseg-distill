"""Integration tests for multi-task sequential training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engine.multi_task_trainer import compute_forgetting, run_task_sequence
from src.engine.evaluator import evaluate
from src.methods import create_method
from src.models.factory import build_model
from src.utils.logging import setup_logger


def _base_cfg():
    return {
        "experiment": {"name": "test_multitask", "seed": 42},
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
    }


def _task_configs():
    return [{"id": "task_0"}, {"id": "task_1"}]


# --- Multi-task loop tests ---


class TestRunTaskSequence:
    def test_finetune_two_tasks(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {"name": "finetune"}
        model = build_model(cfg)
        method = create_method(cfg)
        logger = setup_logger("test_ft")

        result = run_task_sequence(
            model, method, _task_configs(), cfg, logger,
            evaluate_fn=evaluate, output_dir=tmp_path / "ft_run", dry_run=True,
        )

        assert result["task_order"] == ["task_0", "task_1"]
        assert "task_0" in result["eval_history"]["task_1"]
        assert "task_1" in result["eval_history"]["task_1"]
        assert (tmp_path / "ft_run" / "multi_task_summary.json").exists()
        assert (tmp_path / "ft_run" / "forgetting.json").exists()
        assert (tmp_path / "ft_run" / "task_eval_matrix.csv").exists()

    def test_replay_two_tasks(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {"name": "replay", "replay": {"buffer_size": 8, "weight": 1.0}}
        model = build_model(cfg)
        method = create_method(cfg)
        logger = setup_logger("test_rep")

        result = run_task_sequence(
            model, method, _task_configs(), cfg, logger,
            evaluate_fn=evaluate, output_dir=tmp_path / "rep_run", dry_run=True,
        )

        assert result["task_order"] == ["task_0", "task_1"]
        assert len(method.memory) > 0, "Replay buffer should have samples after training"

    def test_distill_two_tasks(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {"name": "distill", "kd": {"weight": 0.5, "temperature": 2.0}}
        model = build_model(cfg)
        method = create_method(cfg)
        logger = setup_logger("test_distill")

        result = run_task_sequence(
            model, method, _task_configs(), cfg, logger,
            evaluate_fn=evaluate, output_dir=tmp_path / "distill_run", dry_run=True,
        )

        assert method.teacher_model is not None, "Teacher should exist after 2 tasks"

    def test_distill_replay_ewc_two_tasks(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {
            "name": "distill_replay_ewc",
            "kd": {"weight": 0.5, "temperature": 2.0},
            "replay": {"buffer_size": 8, "weight": 1.0},
            "ewc": {"weight": 0.2, "fisher_samples": 2},
        }
        model = build_model(cfg)
        method = create_method(cfg)
        logger = setup_logger("test_dre")

        result = run_task_sequence(
            model, method, _task_configs(), cfg, logger,
            evaluate_fn=evaluate, output_dir=tmp_path / "dre_run", dry_run=True,
        )

        assert method.teacher_model is not None
        assert len(method.fisher) > 0, "Fisher should be computed after training"
        assert len(method.prev_params) > 0
        assert len(method.memory) > 0

    def test_checkpoints_saved(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {"name": "finetune"}
        model = build_model(cfg)
        method = create_method(cfg)
        logger = setup_logger("test_ckpt")

        run_task_sequence(
            model, method, _task_configs(), cfg, logger,
            evaluate_fn=evaluate, output_dir=tmp_path / "ckpt_run", dry_run=True,
        )

        assert (tmp_path / "ckpt_run" / "task_0" / "checkpoints" / "after_task_0.pt").exists()
        assert (tmp_path / "ckpt_run" / "task_1" / "checkpoints" / "after_task_1.pt").exists()

    def test_eval_matrix_csv_content(self, tmp_path):
        import csv

        cfg = _base_cfg()
        cfg["method"] = {"name": "finetune"}
        model = build_model(cfg)
        method = create_method(cfg)
        logger = setup_logger("test_csv")

        run_task_sequence(
            model, method, _task_configs(), cfg, logger,
            evaluate_fn=evaluate, output_dir=tmp_path / "csv_run", dry_run=True,
        )

        csv_path = tmp_path / "csv_run" / "task_eval_matrix.csv"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))

        # After task_0: eval on task_0 (1 row)
        # After task_1: eval on task_0 + task_1 (2 rows)
        assert len(rows) == 3
        assert rows[0]["trained_on"] == "task_0"
        assert rows[0]["evaluated_on"] == "task_0"


# --- Forgetting computation tests ---


class TestComputeForgetting:
    def test_no_forgetting_identical(self):
        history = {
            "t0": {"t0": {"dice_mean": 0.8}},
            "t1": {"t0": {"dice_mean": 0.8}, "t1": {"dice_mean": 0.7}},
        }
        result = compute_forgetting(history, ["t0", "t1"])
        assert result["per_task"]["t0"] == 0.0
        assert result["mean"] == 0.0

    def test_positive_forgetting(self):
        history = {
            "t0": {"t0": {"dice_mean": 0.9}},
            "t1": {"t0": {"dice_mean": 0.6}, "t1": {"dice_mean": 0.8}},
        }
        result = compute_forgetting(history, ["t0", "t1"])
        assert abs(result["per_task"]["t0"] - 0.3) < 1e-6
        assert abs(result["mean"] - 0.3) < 1e-6

    def test_negative_forgetting_backward_transfer(self):
        history = {
            "t0": {"t0": {"dice_mean": 0.5}},
            "t1": {"t0": {"dice_mean": 0.7}, "t1": {"dice_mean": 0.6}},
        }
        result = compute_forgetting(history, ["t0", "t1"])
        assert result["per_task"]["t0"] < 0, "Negative forgetting = positive backward transfer"

    def test_three_tasks(self):
        history = {
            "t0": {"t0": {"dice_mean": 0.9}},
            "t1": {"t0": {"dice_mean": 0.8}, "t1": {"dice_mean": 0.85}},
            "t2": {"t0": {"dice_mean": 0.7}, "t1": {"dice_mean": 0.75}, "t2": {"dice_mean": 0.9}},
        }
        result = compute_forgetting(history, ["t0", "t1", "t2"])
        assert abs(result["per_task"]["t0"] - 0.2) < 1e-6
        assert abs(result["per_task"]["t1"] - 0.1) < 1e-6
        assert abs(result["mean"] - 0.15) < 1e-6


# --- Teacher save/load tests ---


class TestTeacherPersistence:
    def test_distill_save_load_roundtrip(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {"name": "distill", "kd": {"weight": 0.5, "temperature": 2.0}}
        model = build_model(cfg)
        method = create_method(cfg)

        # Simulate post_task_update
        method.post_task_update(model)
        assert method.teacher_model is not None

        # Save
        save_path = tmp_path / "distill_state.pt"
        method.save_state(save_path, model_template=model)
        assert save_path.exists()

        # Create fresh method and load
        method2 = create_method(cfg)
        assert method2.teacher_model is None
        method2.load_state(save_path, model_template=model)
        assert method2.teacher_model is not None

        # Verify teacher weights match
        for (n1, p1), (n2, p2) in zip(
            method.teacher_model.named_parameters(),
            method2.teacher_model.named_parameters(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Teacher param mismatch: {n1}"

    def test_ewc_save_load_roundtrip(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {
            "name": "distill_replay_ewc",
            "kd": {"weight": 0.5, "temperature": 2.0},
            "replay": {"buffer_size": 8, "weight": 1.0},
            "ewc": {"weight": 0.2, "fisher_samples": 2},
        }
        model = build_model(cfg)
        method = create_method(cfg)

        # Add some memory
        method.memory.append({
            "image": torch.randn(1, 16, 16, 16),
            "label": torch.randint(0, 3, (16, 16, 16)),
        })

        # Simulate post_task_update with a loader
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.randn(4, 1, 16, 16, 16)
        y = torch.randint(0, 3, (4, 16, 16, 16))

        class DictLoader:
            def __init__(self, x, y, bs=2):
                self.x, self.y, self.bs = x, y, bs
            def __iter__(self):
                for i in range(0, len(self.x), self.bs):
                    yield {"image": self.x[i:i+self.bs], "label": self.y[i:i+self.bs]}
            def __len__(self):
                return (len(self.x) + self.bs - 1) // self.bs

        loader = DictLoader(x, y)
        method.post_task_update(model, train_loader=loader)

        # Save
        save_path = tmp_path / "ewc_state.pt"
        method.save_state(save_path, model_template=model)

        # Load into fresh method
        method2 = create_method(cfg)
        method2.load_state(save_path, model_template=model)

        # Verify Fisher
        assert len(method2.fisher) == len(method.fisher)
        for name in method.fisher:
            assert torch.equal(method.fisher[name], method2.fisher[name])

        # Verify prev_params
        assert len(method2.prev_params) == len(method.prev_params)
        for name in method.prev_params:
            assert torch.equal(method.prev_params[name], method2.prev_params[name])

        # Verify teacher
        assert method2.teacher_model is not None

        # Verify memory
        assert len(method2.memory) == len(method.memory)

    def test_replay_save_load_roundtrip(self, tmp_path):
        cfg = _base_cfg()
        cfg["method"] = {"name": "replay", "replay": {"buffer_size": 8, "weight": 1.0}}
        method = create_method(cfg)

        method.memory.append({
            "image": torch.randn(1, 16, 16, 16),
            "label": torch.randint(0, 3, (16, 16, 16)),
        })

        save_path = tmp_path / "replay_state.pt"
        method.save_state(save_path)
        assert save_path.exists()

        method2 = create_method(cfg)
        method2.load_state(save_path)
        assert len(method2.memory) == 1
        assert torch.equal(method.memory[0]["image"], method2.memory[0]["image"])


if __name__ == "__main__":
    import tempfile

    tests = [
        ("forgetting_no_forgetting", TestComputeForgetting().test_no_forgetting_identical),
        ("forgetting_positive", TestComputeForgetting().test_positive_forgetting),
        ("forgetting_negative_bt", TestComputeForgetting().test_negative_forgetting_backward_transfer),
        ("forgetting_three_tasks", TestComputeForgetting().test_three_tasks),
    ]

    # Tests that need tmp_path
    tmp_tests = [
        ("finetune_two_tasks", TestRunTaskSequence().test_finetune_two_tasks),
        ("replay_two_tasks", TestRunTaskSequence().test_replay_two_tasks),
        ("distill_two_tasks", TestRunTaskSequence().test_distill_two_tasks),
        ("dre_two_tasks", TestRunTaskSequence().test_distill_replay_ewc_two_tasks),
        ("checkpoints_saved", TestRunTaskSequence().test_checkpoints_saved),
        ("eval_matrix_csv", TestRunTaskSequence().test_eval_matrix_csv_content),
        ("distill_save_load", TestTeacherPersistence().test_distill_save_load_roundtrip),
        ("ewc_save_load", TestTeacherPersistence().test_ewc_save_load_roundtrip),
        ("replay_save_load", TestTeacherPersistence().test_replay_save_load_roundtrip),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            failed += 1

    for name, fn in tmp_tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                fn(Path(td))
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            failed += 1

    total = len(tests) + len(tmp_tests)
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {total} tests")
    if failed > 0:
        sys.exit(1)
