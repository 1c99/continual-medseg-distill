"""Tests for infrastructure scripts (no training runs).

Covers core parsing/validation logic for:
- check_experiment_fairness.py
- stats_report.py
- build_failure_panel.py
- compute_report.py
- validate_freeze_spec.py

Run: python tests/test_infra_scripts.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

results = []


def run_test(name, fn):
    import traceback
    try:
        fn()
        results.append(("PASS", name))
    except Exception:
        results.append(("FAIL", name))
        traceback.print_exc()


# ====================================================================
# check_experiment_fairness tests
# ====================================================================

def test_fairness_check_parity_pass():
    from scripts.check_experiment_fairness import check_parity
    configs = {
        "a": {"train": {"epochs": 50, "lr": 0.001}, "data": {"batch_size": 2}},
        "b": {"train": {"epochs": 50, "lr": 0.001}, "data": {"batch_size": 2}},
    }
    results_ = check_parity(configs, [("train", "epochs"), ("train", "lr"), ("data", "batch_size")])
    statuses = [r["status"] for r in results_]
    assert all(s == "PASS" for s in statuses), f"Expected all PASS, got {statuses}"


def test_fairness_check_parity_fail():
    from scripts.check_experiment_fairness import check_parity
    configs = {
        "a": {"train": {"epochs": 50}},
        "b": {"train": {"epochs": 100}},
    }
    results_ = check_parity(configs, [("train", "epochs")])
    assert results_[0]["status"] == "FAIL"


def test_fairness_check_parity_skip():
    from scripts.check_experiment_fairness import check_parity
    configs = {
        "a": {"train": {}},
        "b": {"train": {}},
    }
    results_ = check_parity(configs, [("train", "amp")])
    assert results_[0]["status"] == "SKIP"


def test_fairness_markdown_report():
    from scripts.check_experiment_fairness import check_parity, format_markdown_report
    configs = {
        "ft": {"train": {"epochs": 50}},
        "kd": {"train": {"epochs": 50}},
    }
    results_ = check_parity(configs, [("train", "epochs")])
    md = format_markdown_report(results_, ["ft", "kd"])
    assert "PASS" in md
    assert "Fairness" in md


def test_fairness_resolve_method_config():
    from scripts.check_experiment_fairness import resolve_method_config
    repo = Path(_REPO_ROOT)
    cfg = resolve_method_config(
        repo / "configs" / "base.yaml",
        repo / "configs" / "methods" / "finetune.yaml",
    )
    assert cfg.get("method", {}).get("name") == "finetune"
    assert "model" in cfg


def test_fairness_real_configs_pass():
    """Finetune and replay should agree on training params."""
    from scripts.check_experiment_fairness import resolve_method_config, check_parity
    repo = Path(_REPO_ROOT)
    ft = resolve_method_config(repo / "configs/base.yaml", repo / "configs/methods/finetune.yaml")
    rp = resolve_method_config(repo / "configs/base.yaml", repo / "configs/methods/replay.yaml")
    results_ = check_parity(
        {"finetune": ft, "replay": rp},
        [("train", "epochs"), ("train", "lr"), ("data", "batch_size")],
    )
    for r in results_:
        assert r["status"] in ("PASS", "SKIP"), f"Unexpected {r['status']} for {r['key']}"


# ====================================================================
# stats_report tests
# ====================================================================

def test_stats_aggregate_metrics():
    from scripts.stats_report import aggregate_metrics
    per_run = [
        {"val_dice_mean": 0.8, "val_hd95_mean": 10.0},
        {"val_dice_mean": 0.9, "val_hd95_mean": 8.0},
        {"val_dice_mean": 0.85, "val_hd95_mean": 9.0},
    ]
    agg = aggregate_metrics(per_run, ["val_dice_mean", "val_hd95_mean"])
    assert abs(agg["val_dice_mean"]["mean"] - 0.85) < 1e-6
    assert agg["val_dice_mean"]["count"] == 3
    assert agg["val_dice_mean"]["std"] > 0


def test_stats_aggregate_empty():
    from scripts.stats_report import aggregate_metrics
    agg = aggregate_metrics([], None)
    assert agg == {}


def test_stats_bootstrap_ci_sufficient():
    from scripts.stats_report import bootstrap_ci
    values = [0.8, 0.85, 0.9, 0.82, 0.88]
    ci = bootstrap_ci(values, n_bootstrap=500, ci=0.95)
    assert "lower" in ci
    assert "upper" in ci
    assert ci["lower"] <= ci["upper"]


def test_stats_bootstrap_ci_insufficient():
    from scripts.stats_report import bootstrap_ci
    ci = bootstrap_ci([0.8, 0.9], n_bootstrap=100)
    assert "caveat" in ci


def test_stats_paired_comparison():
    from scripts.stats_report import paired_comparison
    comp = paired_comparison([0.8, 0.85, 0.9], [0.82, 0.88, 0.92], "A", "B")
    assert comp["n_a"] == 3
    assert comp["n_b"] == 3
    assert comp["delta"] > 0  # B is slightly better


def test_stats_paired_comparison_insufficient():
    from scripts.stats_report import paired_comparison
    comp = paired_comparison([0.8], [0.9], "A", "B")
    assert "caveat" in comp


def test_stats_read_metrics_csv():
    from scripts.stats_report import read_metrics_csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_dice_mean"])
        w.writerow([1, 1.5, 0.8])
        w.writerow([2, 1.2, 0.85])
        f.flush()
        rows = read_metrics_csv(Path(f.name))
    assert len(rows) == 2
    assert rows[1]["val_dice_mean"] == 0.85
    assert rows[0]["epoch"] == 1


def test_stats_read_nonexistent():
    from scripts.stats_report import read_metrics_csv
    rows = read_metrics_csv(Path("/nonexistent/metrics.csv"))
    assert rows == []


def test_stats_format_markdown():
    from scripts.stats_report import format_markdown
    agg = {
        "cond1": {
            "val_dice_mean": {"mean": 0.85, "std": 0.05, "min": 0.8, "max": 0.9, "count": 3},
        },
    }
    md = format_markdown(agg, caveats=["test caveat"])
    assert "cond1" in md
    assert "0.85" in md
    assert "test caveat" in md


# ====================================================================
# build_failure_panel tests
# ====================================================================

def test_panel_rank_cases_worst():
    from scripts.build_failure_panel import rank_cases
    rows = [
        {"val_dice_mean": 0.9, "subject": "s1"},
        {"val_dice_mean": 0.3, "subject": "s2"},
        {"val_dice_mean": 0.7, "subject": "s3"},
        {"val_dice_mean": 0.1, "subject": "s4"},
    ]
    ranked = rank_cases(rows, "val_dice_mean", top_k=2, direction="worst")
    assert len(ranked) == 2
    assert ranked[0]["metric_value"] == 0.1  # worst Dice
    assert ranked[1]["metric_value"] == 0.3


def test_panel_rank_cases_best():
    from scripts.build_failure_panel import rank_cases
    rows = [
        {"val_dice_mean": 0.9}, {"val_dice_mean": 0.3}, {"val_dice_mean": 0.7},
    ]
    ranked = rank_cases(rows, "val_dice_mean", top_k=2, direction="best")
    assert ranked[0]["metric_value"] == 0.9


def test_panel_rank_hd95_worst():
    from scripts.build_failure_panel import rank_cases
    rows = [
        {"val_hd95_mean": 5.0}, {"val_hd95_mean": 50.0}, {"val_hd95_mean": 20.0},
    ]
    ranked = rank_cases(rows, "val_hd95_mean", top_k=2, direction="worst")
    # Worst HD95 = highest
    assert ranked[0]["metric_value"] == 50.0


def test_panel_rank_missing_metric():
    from scripts.build_failure_panel import rank_cases
    rows = [{"other_metric": 0.5}]
    ranked = rank_cases(rows, "val_dice_mean", top_k=5)
    assert len(ranked) == 0


def test_panel_build_manifest():
    from scripts.build_failure_panel import rank_cases, build_manifest
    rows = [{"val_dice_mean": 0.3}, {"val_dice_mean": 0.9}]
    ranked = rank_cases(rows, "val_dice_mean", top_k=1, direction="worst")
    manifest = build_manifest(ranked, Path("/test"), "val_dice_mean", "worst")
    assert manifest["num_cases"] == 1
    assert manifest["metric_key"] == "val_dice_mean"


def test_panel_markdown():
    from scripts.build_failure_panel import rank_cases, build_manifest, format_markdown_panel
    rows = [{"val_dice_mean": 0.2, "trained_on": "tA", "evaluated_on": "tB"}]
    ranked = rank_cases(rows, "val_dice_mean", top_k=1)
    manifest = build_manifest(ranked, Path("/test"), "val_dice_mean", "worst")
    md = format_markdown_panel(manifest)
    assert "Failure Case Panel" in md
    assert "0.2000" in md


def test_panel_from_real_data():
    """Test against actual task_eval_matrix.csv if it exists."""
    from scripts.build_failure_panel import read_csv, rank_cases
    csv_path = Path(_REPO_ROOT) / "outputs" / "smoke_AB_real" / "task_eval_matrix.csv"
    if csv_path.exists():
        rows = read_csv(csv_path)
        assert len(rows) > 0
        ranked = rank_cases(rows, "val_dice_mean", top_k=3, direction="worst")
        assert len(ranked) > 0
    # else: skip silently


# ====================================================================
# compute_report tests
# ====================================================================

def test_compute_count_params():
    from scripts.compute_report import count_model_params
    cfg = {
        "model": {
            "name": "monai_unet", "in_channels": 1, "out_channels": 3,
            "channels": [8, 16], "strides": [2], "num_res_units": 1,
        },
    }
    info = count_model_params(cfg)
    assert info["total_params"] > 0
    assert info["source"] in ("instantiated", "estimated")


def test_compute_parse_metrics():
    from scripts.compute_report import parse_metrics_csv
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss"])
        w.writerow([1, 2.5])
        w.writerow([2, 1.8])
        f.flush()
        info = parse_metrics_csv(Path(f.name))
    assert info["num_epochs"] == 2
    assert info["final_train_loss"] == 1.8


def test_compute_parse_manifest():
    from scripts.compute_report import parse_run_manifest
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"environment": {"gpu_name": "RTX 3090", "torch_version": "2.0"}}, f)
        f.flush()
        info = parse_run_manifest(Path(f.name))
    assert info["gpu_name"] == "RTX 3090"


def test_compute_markdown_report():
    from scripts.compute_report import format_markdown_report
    runs = [
        {"name": "finetune", "method": "finetune", "total_params": 1000000,
         "trainable_params": 1000000, "lora_params": 0, "num_epochs": 50,
         "amp_enabled": False},
    ]
    md = format_markdown_report(runs)
    assert "Compute Efficiency" in md
    assert "finetune" in md
    assert "1,000,000" in md


def test_compute_csv_report():
    from scripts.compute_report import format_csv_report
    runs = [{"name": "test", "method": "finetune", "total_params": 100}]
    csv_out = format_csv_report(runs)
    assert "name" in csv_out
    assert "test" in csv_out


# ====================================================================
# validate_freeze_spec tests
# ====================================================================

def test_freeze_validate_condition_pass():
    from scripts.validate_freeze_spec import validate_condition_against_spec
    cfg = {"train": {"epochs": 50, "lr": 0.001}, "data": {"batch_size": 2},
           "model": {"name": "monai_unet", "in_channels": 1, "out_channels": 6}}
    spec = {
        "training": {"epochs": 50, "lr": 0.001, "batch_size": 2},
        "model": {"name": "monai_unet", "in_channels": 1, "out_channels": 6},
    }
    results_ = validate_condition_against_spec("test", cfg, spec)
    fails = [r for r in results_ if r["status"] == "FAIL"]
    assert len(fails) == 0, f"Unexpected failures: {fails}"


def test_freeze_validate_condition_fail():
    from scripts.validate_freeze_spec import validate_condition_against_spec
    cfg = {"train": {"epochs": 10}}
    spec = {"training": {"epochs": 50}, "model": {}}
    results_ = validate_condition_against_spec("test", cfg, spec)
    fails = [r for r in results_ if r["status"] == "FAIL"]
    assert len(fails) == 1
    assert fails[0]["key"] == "train.epochs"


def test_freeze_spec_completeness():
    from scripts.validate_freeze_spec import validate_spec_completeness
    spec = {"conditions": [{"name": "a", "method_config": "nonexistent.yaml"}]}
    warnings = validate_spec_completeness(spec)
    assert any("seeds" in w for w in warnings)
    assert any("nonexistent" in w for w in warnings)


def test_freeze_spec_loads():
    from scripts.validate_freeze_spec import load_freeze_spec
    spec_path = Path(_REPO_ROOT) / "configs" / "experiments" / "paper_ablation_freeze.yaml"
    if spec_path.exists():
        spec = load_freeze_spec(spec_path)
        assert "conditions" in spec
        assert len(spec["conditions"]) >= 4
        assert "seeds" in spec
        assert len(spec["seeds"]) == 3


def test_freeze_format_report():
    from scripts.validate_freeze_spec import format_report
    results_ = [
        {"condition": "a", "key": "train.epochs", "status": "PASS",
         "expected": 50, "actual": 50, "reason": "ok"},
        {"condition": "b", "key": "train.epochs", "status": "FAIL",
         "expected": 50, "actual": 10, "reason": "drift"},
    ]
    spec = {"spec_version": "1.0", "frozen_at": "2026-03-27"}
    md = format_report(results_, [], spec)
    assert "FAIL" in md
    assert "Drift" in md


# ====================================================================
# Runner
# ====================================================================

if __name__ == "__main__":
    tests = [
        # Fairness
        ("fairness_parity_pass", test_fairness_check_parity_pass),
        ("fairness_parity_fail", test_fairness_check_parity_fail),
        ("fairness_parity_skip", test_fairness_check_parity_skip),
        ("fairness_markdown_report", test_fairness_markdown_report),
        ("fairness_resolve_method_config", test_fairness_resolve_method_config),
        ("fairness_real_configs_pass", test_fairness_real_configs_pass),
        # Stats
        ("stats_aggregate_metrics", test_stats_aggregate_metrics),
        ("stats_aggregate_empty", test_stats_aggregate_empty),
        ("stats_bootstrap_ci_sufficient", test_stats_bootstrap_ci_sufficient),
        ("stats_bootstrap_ci_insufficient", test_stats_bootstrap_ci_insufficient),
        ("stats_paired_comparison", test_stats_paired_comparison),
        ("stats_paired_comparison_insufficient", test_stats_paired_comparison_insufficient),
        ("stats_read_metrics_csv", test_stats_read_metrics_csv),
        ("stats_read_nonexistent", test_stats_read_nonexistent),
        ("stats_format_markdown", test_stats_format_markdown),
        # Failure panel
        ("panel_rank_worst", test_panel_rank_cases_worst),
        ("panel_rank_best", test_panel_rank_cases_best),
        ("panel_rank_hd95_worst", test_panel_rank_hd95_worst),
        ("panel_rank_missing_metric", test_panel_rank_missing_metric),
        ("panel_build_manifest", test_panel_build_manifest),
        ("panel_markdown", test_panel_markdown),
        ("panel_from_real_data", test_panel_from_real_data),
        # Compute
        ("compute_count_params", test_compute_count_params),
        ("compute_parse_metrics", test_compute_parse_metrics),
        ("compute_parse_manifest", test_compute_parse_manifest),
        ("compute_markdown_report", test_compute_markdown_report),
        ("compute_csv_report", test_compute_csv_report),
        # Freeze spec
        ("freeze_validate_pass", test_freeze_validate_condition_pass),
        ("freeze_validate_fail", test_freeze_validate_condition_fail),
        ("freeze_spec_completeness", test_freeze_spec_completeness),
        ("freeze_spec_loads", test_freeze_spec_loads),
        ("freeze_format_report", test_freeze_format_report),
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
