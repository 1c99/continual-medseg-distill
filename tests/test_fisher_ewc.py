"""Tests for Fisher-based EWC in DistillReplayEWCMethod."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.methods.distill_replay_ewc import DistillReplayEWCMethod


def _make_cfg(**overrides):
    cfg = {
        "method": {
            "name": "distill_replay_ewc",
            "kd": {"weight": 0.5, "temperature": 2.0},
            "replay": {"buffer_size": 16, "weight": 1.0},
            "ewc": {"weight": 0.2, "fisher_samples": 4},
        },
    }
    cfg.update(overrides)
    return cfg


def _make_model(in_features=4, num_classes=3):
    return nn.Linear(in_features, num_classes)


def _make_loader(n=8, in_features=4, num_classes=3, batch_size=4):
    x = torch.randn(n, in_features)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    # Wrap in dicts like the training pipeline expects
    class DictDataset:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __iter__(self):
            for x, y in DataLoader(self.ds, batch_size=batch_size):
                yield {"image": x, "label": y}
    return DictDataset(ds)


class TestEstimateFisher:
    def test_shapes_match_params(self):
        model = _make_model()
        loader = _make_loader()
        method = DistillReplayEWCMethod(_make_cfg())
        fisher = method._estimate_fisher(model, loader, "cpu", n_samples=4)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in fisher, f"Missing Fisher entry for {name}"
                assert fisher[name].shape == param.shape, (
                    f"Shape mismatch for {name}: {fisher[name].shape} vs {param.shape}"
                )

    def test_values_non_negative(self):
        model = _make_model()
        loader = _make_loader()
        method = DistillReplayEWCMethod(_make_cfg())
        fisher = method._estimate_fisher(model, loader, "cpu", n_samples=4)

        for name, f in fisher.items():
            assert (f >= 0).all(), f"Negative Fisher values for {name}"

    def test_values_nonzero(self):
        """Fisher should have some nonzero entries (model is not degenerate)."""
        model = _make_model()
        loader = _make_loader()
        method = DistillReplayEWCMethod(_make_cfg())
        fisher = method._estimate_fisher(model, loader, "cpu", n_samples=4)

        total = sum(f.sum().item() for f in fisher.values())
        assert total > 0, "Fisher diagonal is all zeros — something is wrong"


class TestEwcPenalty:
    def test_zero_penalty_unchanged_params(self):
        model = _make_model()
        loader = _make_loader()
        method = DistillReplayEWCMethod(_make_cfg())

        # Simulate post_task_update: store Fisher + params
        method.fisher = method._estimate_fisher(model, loader, "cpu", n_samples=4)
        method.prev_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

        penalty = method._ewc_penalty(model, "cpu")
        assert penalty.item() == 0.0, f"Expected 0 penalty for unchanged params, got {penalty.item()}"

    def test_positive_penalty_perturbed_params(self):
        model = _make_model()
        loader = _make_loader()
        method = DistillReplayEWCMethod(_make_cfg())

        # Store Fisher + params
        method.fisher = method._estimate_fisher(model, loader, "cpu", n_samples=4)
        method.prev_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

        # Perturb model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        penalty = method._ewc_penalty(model, "cpu")
        assert penalty.item() > 0.0, f"Expected positive penalty after perturbation, got {penalty.item()}"

    def test_no_fisher_no_penalty(self):
        """Before any task update, penalty should be 0."""
        model = _make_model()
        method = DistillReplayEWCMethod(_make_cfg())
        penalty = method._ewc_penalty(model, "cpu")
        assert penalty.item() == 0.0


class TestPostTaskUpdate:
    def test_stores_fisher_and_params(self):
        model = _make_model()
        loader = _make_loader()
        method = DistillReplayEWCMethod(_make_cfg())

        method.post_task_update(model, train_loader=loader)

        assert len(method.fisher) > 0, "Fisher dict is empty after post_task_update"
        assert len(method.prev_params) > 0, "prev_params dict is empty after post_task_update"
        assert method.teacher_model is not None, "teacher_model is None after post_task_update"

    def test_no_loader_skips_fisher(self):
        model = _make_model()
        method = DistillReplayEWCMethod(_make_cfg())

        method.post_task_update(model)

        assert len(method.fisher) == 0, "Fisher should be empty when no train_loader provided"
        assert len(method.prev_params) > 0, "prev_params should still be stored"


if __name__ == "__main__":
    tests = [
        ("fisher_shapes", TestEstimateFisher().test_shapes_match_params),
        ("fisher_non_negative", TestEstimateFisher().test_values_non_negative),
        ("fisher_nonzero", TestEstimateFisher().test_values_nonzero),
        ("zero_penalty_unchanged", TestEwcPenalty().test_zero_penalty_unchanged_params),
        ("positive_penalty_perturbed", TestEwcPenalty().test_positive_penalty_perturbed_params),
        ("no_fisher_no_penalty", TestEwcPenalty().test_no_fisher_no_penalty),
        ("post_task_stores", TestPostTaskUpdate().test_stores_fisher_and_params),
        ("no_loader_skips_fisher", TestPostTaskUpdate().test_no_loader_skips_fisher),
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
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
