"""Tests for Code-Only Sprint: DDP, patches, cache, matrix, registry.

Run: python tests/test_sprint_code.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.engine.distributed import DistributedContext, setup_ddp
from src.data.patch_sampler import compute_patch_coords, extract_patch, reconstruct_volume
from src.methods.teacher_cache import TeacherCache
from src.utils.memory_guard import oom_guard

results = []


def run_test(name, fn):
    try:
        fn()
        results.append(("PASS", name))
    except Exception:
        results.append(("FAIL", name))
        traceback.print_exc()


# ====================================================================
# Task 1: DDP
# ====================================================================

def test_ddp_disabled_context():
    ctx = DistributedContext(enabled=False)
    assert ctx.is_main_process()
    assert ctx.rank == 0
    assert ctx.world_size == 1


def test_ddp_setup_from_config():
    cfg = {"runtime": {"distributed": {"enabled": False, "backend": "gloo", "grad_accum_steps": 4}}}
    ctx = setup_ddp(cfg)
    assert not ctx.enabled
    assert ctx.grad_accum_steps == 4


def test_ddp_wrap_model_noop():
    ctx = DistributedContext(enabled=False)
    model = torch.nn.Linear(10, 5)
    wrapped = ctx.wrap_model(model)
    assert wrapped is model  # no wrapping when disabled


def test_ddp_sampler_none_when_disabled():
    ctx = DistributedContext(enabled=False)
    ds = torch.utils.data.TensorDataset(torch.randn(10, 3))
    assert ctx.make_sampler(ds) is None


def test_ddp_grad_accum():
    ctx = DistributedContext(enabled=False, grad_accum_steps=4)
    assert ctx.should_accumulate(1)  # step 1: accumulate
    assert ctx.should_accumulate(2)  # step 2: accumulate
    assert ctx.should_accumulate(3)  # step 3: accumulate
    assert not ctx.should_accumulate(4)  # step 4: update


def test_ddp_reduce_noop():
    ctx = DistributedContext(enabled=False)
    t = torch.tensor([1.0, 2.0])
    out = ctx.reduce_tensor(t)
    assert torch.equal(out, t)


# ====================================================================
# Task 2: Patch Sampler + Memory
# ====================================================================

def test_patch_coords_full_coverage():
    coords = compute_patch_coords((64, 64, 64), (32, 32, 32), (32, 32, 32))
    assert len(coords) == 8  # 2x2x2 non-overlapping


def test_patch_coords_with_overlap():
    coords = compute_patch_coords((64, 64, 64), (32, 32, 32), (16, 16, 16))
    assert len(coords) > 8  # overlapping


def test_patch_coords_boundary():
    """Patches cover boundary even with non-divisible sizes."""
    coords = compute_patch_coords((50, 50, 50), (32, 32, 32), (32, 32, 32))
    # Last patch should be at 18 (50-32) to cover the edge
    max_d = max(c[0] for c in coords)
    assert max_d + 32 >= 50


def test_extract_patch():
    vol = torch.arange(64*64*64, dtype=torch.float32).reshape(1, 64, 64, 64)
    patch = extract_patch(vol, (0, 0, 0), (32, 32, 32))
    assert patch.shape == (1, 32, 32, 32)
    assert torch.equal(patch, vol[:, :32, :32, :32])


def test_reconstruct_volume_identity():
    """Non-overlapping patches reconstruct exactly."""
    vol = torch.randn(1, 64, 64, 64)
    coords = compute_patch_coords((64, 64, 64), (32, 32, 32), (32, 32, 32))
    patches = [extract_patch(vol, c, (32, 32, 32)) for c in coords]
    recon = reconstruct_volume(patches, coords, (64, 64, 64), num_channels=1)
    assert torch.allclose(recon, vol, atol=1e-6)


def test_oom_guard_normal():
    def ok_fn():
        return torch.tensor(42.0)
    result = oom_guard(ok_fn)
    assert result is not None
    assert result.item() == 42.0


def test_oom_guard_catches():
    def oom_fn():
        raise RuntimeError("CUDA out of memory. Tried to allocate ...")
    result = oom_guard(oom_fn)
    assert result is None


# ====================================================================
# Task 3: Teacher Cache
# ====================================================================

def test_cache_put_get():
    with tempfile.TemporaryDirectory() as tmp:
        cache = TeacherCache(tmp, "hash123")
        logits = torch.randn(2, 3, 16, 16, 16)
        cache.put("sample_0", logits)
        result = cache.get("sample_0")
        assert result is not None
        assert torch.allclose(result["logits"], logits)


def test_cache_miss():
    with tempfile.TemporaryDirectory() as tmp:
        cache = TeacherCache(tmp, "hash123")
        assert cache.get("nonexistent") is None


def test_cache_stats():
    with tempfile.TemporaryDirectory() as tmp:
        cache = TeacherCache(tmp, "hash123")
        cache.put("s1", torch.tensor(1.0))
        cache.get("s1")  # hit
        cache.get("s2")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1


def test_cache_invalidate():
    with tempfile.TemporaryDirectory() as tmp:
        cache = TeacherCache(tmp, "hash123")
        cache.put("s1", torch.tensor(1.0))
        cache.put("s2", torch.tensor(2.0))
        assert len(cache) == 2
        removed = cache.invalidate()
        assert removed == 2
        assert len(cache) == 0


def test_cache_key_deterministic():
    k1 = TeacherCache.make_key("sample_0", "config_abc")
    k2 = TeacherCache.make_key("sample_0", "config_abc")
    k3 = TeacherCache.make_key("sample_0", "config_xyz")
    assert k1 == k2
    assert k1 != k3


# ====================================================================
# Task 4: Ablation Matrix
# ====================================================================

def test_ablation_matrix_compile():
    import csv
    with tempfile.TemporaryDirectory() as tmp:
        # Create synthetic ablation structure
        for method in ["finetune", "replay"]:
            run_dir = Path(tmp) / "run_001" / method
            run_dir.mkdir(parents=True)
            with (run_dir / "metrics.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "val_dice_mean"])
                w.writeheader()
                w.writerow({"epoch": 1, "val_dice_mean": 0.5 if method == "finetune" else 0.6})

        from scripts.compile_ablation_matrix import compile_matrix
        result = compile_matrix(Path(tmp))
        assert len(result["methods"]) == 2
        assert "finetune" in result["methods"]
        assert len(result["matrix"]) == 2


def test_ablation_matrix_missing_graceful():
    with tempfile.TemporaryDirectory() as tmp:
        from scripts.compile_ablation_matrix import compile_matrix
        result = compile_matrix(Path(tmp))
        assert len(result["warnings"]) > 0
        assert len(result["matrix"]) == 0


# ====================================================================
# Task 6: Experiment Registry
# ====================================================================

def test_registry_load():
    import yaml
    registry = yaml.safe_load(
        (Path(_REPO_ROOT) / "experiments" / "registry.yaml").read_text()
    )
    experiments = registry.get("experiments", [])
    assert len(experiments) > 0
    assert all("id" in e for e in experiments)
    assert all("status" in e for e in experiments)


def test_registry_validate_refs():
    from scripts.experiment_status import load_registry, validate_config_refs
    registry = load_registry()
    warnings = validate_config_refs(registry.get("experiments", []))
    # Some refs might not exist (workstation-specific), just check it runs
    assert isinstance(warnings, list)


def test_registry_status_update():
    import yaml
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.yaml"
        reg_path.write_text(yaml.dump({
            "experiments": [{"id": "test_exp", "status": "pending"}]
        }))
        from scripts.experiment_status import load_registry, update_status, save_registry
        # Monkey-patch the path
        import scripts.experiment_status as mod
        orig = mod.REGISTRY_PATH
        mod.REGISTRY_PATH = reg_path
        try:
            registry = load_registry(reg_path)
            exps = registry["experiments"]
            assert update_status(exps, "test_exp", "done")
            assert exps[0]["status"] == "done"
        finally:
            mod.REGISTRY_PATH = orig


# ====================================================================
# Runner
# ====================================================================

if __name__ == "__main__":
    tests = [
        # Task 1: DDP
        ("ddp_disabled_context", test_ddp_disabled_context),
        ("ddp_setup_from_config", test_ddp_setup_from_config),
        ("ddp_wrap_model_noop", test_ddp_wrap_model_noop),
        ("ddp_sampler_none_when_disabled", test_ddp_sampler_none_when_disabled),
        ("ddp_grad_accum", test_ddp_grad_accum),
        ("ddp_reduce_noop", test_ddp_reduce_noop),
        # Task 2: Patches + Memory
        ("patch_coords_full_coverage", test_patch_coords_full_coverage),
        ("patch_coords_with_overlap", test_patch_coords_with_overlap),
        ("patch_coords_boundary", test_patch_coords_boundary),
        ("extract_patch", test_extract_patch),
        ("reconstruct_volume_identity", test_reconstruct_volume_identity),
        ("oom_guard_normal", test_oom_guard_normal),
        ("oom_guard_catches", test_oom_guard_catches),
        # Task 3: Teacher Cache
        ("cache_put_get", test_cache_put_get),
        ("cache_miss", test_cache_miss),
        ("cache_stats", test_cache_stats),
        ("cache_invalidate", test_cache_invalidate),
        ("cache_key_deterministic", test_cache_key_deterministic),
        # Task 4: Ablation Matrix
        ("ablation_matrix_compile", test_ablation_matrix_compile),
        ("ablation_matrix_missing_graceful", test_ablation_matrix_missing_graceful),
        # Task 6: Registry
        ("registry_load", test_registry_load),
        ("registry_validate_refs", test_registry_validate_refs),
        ("registry_status_update", test_registry_status_update),
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
