"""Tests for src.utils.reproducibility."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.reproducibility import collect_env_info, get_git_info, set_seed


def test_set_seed_deterministic():
    """set_seed should make torch.randn reproducible."""
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b), f"Expected identical tensors, got {a} vs {b}"


def test_get_git_info_keys():
    """get_git_info should return dict with git_commit and git_dirty."""
    info = get_git_info()
    assert isinstance(info, dict)
    assert "git_commit" in info
    assert "git_dirty" in info


def test_collect_env_info_required_keys():
    """collect_env_info should return all required metadata keys."""
    info = collect_env_info()
    required = {"python_version", "torch_version", "monai_version", "platform", "cuda_available", "git_commit", "git_dirty"}
    assert required.issubset(info.keys()), f"Missing keys: {required - info.keys()}"


def test_collect_env_info_with_cfg():
    """collect_env_info should include random_seed when cfg is provided."""
    info = collect_env_info({"experiment": {"seed": 99}})
    assert info["random_seed"] == 99


def test_collect_env_info_seed_not_set():
    """collect_env_info should report 'not set' when seed is missing from cfg."""
    info = collect_env_info({"experiment": {}})
    assert info["random_seed"] == "not set"


if __name__ == "__main__":
    tests = [
        test_set_seed_deterministic,
        test_get_git_info_keys,
        test_collect_env_info_required_keys,
        test_collect_env_info_with_cfg,
        test_collect_env_info_seed_not_set,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
