"""Tests for scripts/validate_data.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_data import (
    generate_report,
    load_dataset_config,
    resolve_source_config,
    validate_paths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_totalseg_config(tmp_path: Path, root: str, manifest_path: str | None = None):
    """Create a minimal totalseg dataset config and return the YAML path."""
    cfg = {
        "data": {
            "source": "totalseg",
            "totalseg": {
                "root": root,
                "organ": "liver",
                "shape": [32, 32, 32],
            },
        }
    }
    if manifest_path:
        cfg["data"]["totalseg"]["split_manifest"] = manifest_path
    else:
        cfg["data"]["totalseg"]["train_ids"] = ["s0001", "s0002"]
        cfg["data"]["totalseg"]["val_ids"] = ["s0003"]

    cfg_path = tmp_path / "ds.yaml"
    _write_yaml(cfg_path, cfg)
    return cfg_path


# ---------------------------------------------------------------------------
# Test: imports work
# ---------------------------------------------------------------------------

def test_imports():
    """Verify all public functions are importable."""
    from scripts.validate_data import (
        generate_report,
        load_dataset_config,
        main,
        parse_args,
        resolve_source_config,
        validate_paths,
        load_samples,
    )
    assert callable(generate_report)
    assert callable(main)


# ---------------------------------------------------------------------------
# Test: config loading and source resolution
# ---------------------------------------------------------------------------

def test_load_dataset_config_merges(tmp_path):
    base_cfg = {"model": {"name": "unet"}, "data": {"batch_size": 2}}
    ds_cfg = {"data": {"source": "totalseg", "totalseg": {"root": "/fake"}}}
    base_path = tmp_path / "base.yaml"
    ds_path = tmp_path / "ds.yaml"
    _write_yaml(base_path, base_cfg)
    _write_yaml(ds_path, ds_cfg)

    merged = load_dataset_config(str(ds_path), str(base_path))
    assert merged["data"]["source"] == "totalseg"
    assert merged["data"]["batch_size"] == 2


def test_resolve_source_config_with_manifest(tmp_path):
    manifest = {"train": ["s01", "s02"], "val": ["s03"]}
    manifest_path = tmp_path / "split.json"
    _write_json(manifest_path, manifest)

    cfg = {
        "data": {
            "source": "totalseg",
            "totalseg": {
                "root": "/fake",
                "split_manifest": str(manifest_path),
            },
        }
    }
    source, _, root, mp, train_ids, val_ids = resolve_source_config(cfg)
    assert source == "totalseg"
    assert train_ids == ["s01", "s02"]
    assert val_ids == ["s03"]


def test_resolve_source_config_inline_ids(tmp_path):
    cfg = {
        "data": {
            "source": "brats21",
            "brats21": {
                "root": "/fake",
                "train_ids": ["BraTS2021_00000"],
                "val_ids": ["BraTS2021_00001"],
            },
        }
    }
    source, _, root, mp, train_ids, val_ids = resolve_source_config(cfg)
    assert source == "brats21"
    assert train_ids == ["BraTS2021_00000"]
    assert mp is None


# ---------------------------------------------------------------------------
# Test: path validation
# ---------------------------------------------------------------------------

def test_validate_paths_missing_root(tmp_path):
    result = validate_paths(
        source="totalseg",
        source_cfg={"organ": "liver"},
        root="/nonexistent/path",
        manifest_path=None,
        train_ids=["s01"],
        val_ids=["s02"],
    )
    assert result["root_exists"] is False
    assert result["total_ids"] == 2
    assert len(result["missing_ids"]) == 2


def test_validate_paths_no_root_set():
    result = validate_paths(
        source="totalseg",
        source_cfg={},
        root=None,
        manifest_path=None,
        train_ids=[],
        val_ids=[],
    )
    assert "error" in result


def test_validate_paths_totalseg_found(tmp_path):
    """Create fake totalseg files and confirm they are found."""
    root = tmp_path / "totalseg"
    sid = "s0001"
    (root / sid / "segmentations").mkdir(parents=True)
    (root / sid / "ct.nii.gz").write_bytes(b"fake")
    (root / sid / "segmentations" / "liver.nii.gz").write_bytes(b"fake")

    result = validate_paths(
        source="totalseg",
        source_cfg={"organ": "liver"},
        root=str(root),
        manifest_path=None,
        train_ids=[sid],
        val_ids=[],
    )
    assert result["root_exists"] is True
    assert result["found_on_disk"] == 1
    assert len(result["missing_ids"]) == 0


def test_validate_paths_brats21_partial(tmp_path):
    """BraTS subject missing seg file is reported missing."""
    root = tmp_path / "brats"
    sid = "BraTS2021_00000"
    case_dir = root / sid
    case_dir.mkdir(parents=True)
    for mod in ("t1", "t1ce", "t2", "flair"):
        (case_dir / f"{sid}_{mod}.nii.gz").write_bytes(b"fake")
    # deliberately skip seg

    result = validate_paths(
        source="brats21",
        source_cfg={},
        root=str(root),
        manifest_path=None,
        train_ids=[sid],
        val_ids=[],
    )
    assert result["found_on_disk"] == 0
    assert sid in result["missing_ids"]
    assert sid in result["missing_files"]


def test_validate_paths_acdc(tmp_path):
    """ACDC subject with correct files is found."""
    root = tmp_path / "acdc"
    patient_dir = root / "patient001"
    patient_dir.mkdir(parents=True)
    (patient_dir / "frame01.nii.gz").write_bytes(b"fake")
    (patient_dir / "frame01_gt.nii.gz").write_bytes(b"fake")

    result = validate_paths(
        source="acdc",
        source_cfg={},
        root=str(root),
        manifest_path=None,
        train_ids=["patient001_frame01"],
        val_ids=[],
    )
    assert result["found_on_disk"] == 1


# ---------------------------------------------------------------------------
# Test: report generation
# ---------------------------------------------------------------------------

def test_report_phase1_only():
    path_result = {
        "source": "totalseg",
        "root": "/data/totalseg",
        "root_exists": True,
        "manifest_path": "/data/split.json",
        "manifest_exists": True,
        "total_ids": 5,
        "found_on_disk": 3,
        "missing_ids": ["s04", "s05"],
        "missing_files": {
            "s04": ["/data/totalseg/s04/ct.nii.gz"],
            "s05": ["/data/totalseg/s05/ct.nii.gz"],
        },
    }
    report = generate_report(path_result, load_result=None)
    assert "# Data Validation Report" in report
    assert "Phase 1" in report
    assert "Phase 2" not in report
    assert "`totalseg`" in report
    assert "Missing**: 2" in report


def test_report_with_loading():
    path_result = {
        "source": "brats21",
        "root": "/data/brats",
        "root_exists": True,
        "manifest_path": None,
        "manifest_exists": None,
        "total_ids": 3,
        "found_on_disk": 3,
        "missing_ids": [],
        "missing_files": {},
    }
    load_result = {
        "attempted": 3,
        "loaded": 2,
        "failed": 1,
        "errors": ["BraTS2021_00002: file missing"],
        "shapes": [(4, 128, 128, 128), (4, 128, 128, 128)],
        "label_values": {0, 1, 2, 3},
    }
    report = generate_report(path_result, load_result)
    assert "Phase 2" in report
    assert "Loaded**: 2" in report
    assert "Failed**: 1" in report
    assert "[0, 1, 2, 3]" in report
    assert "Volume shapes" in report


def test_report_empty_manifest():
    """Edge case: root not found produces clear messaging."""
    path_result = {
        "source": "acdc",
        "root": "/nonexistent",
        "root_exists": False,
        "manifest_path": "/nonexistent/split.json",
        "manifest_exists": False,
        "total_ids": 2,
        "found_on_disk": 0,
        "missing_ids": ["p1_f1", "p2_f1"],
        "missing_files": {},
    }
    report = generate_report(path_result)
    assert "Root exists**: False" in report
    assert "Missing**: 2" in report


# ---------------------------------------------------------------------------
# Test: CLI / main (end-to-end with fake config, no real data)
# ---------------------------------------------------------------------------

def test_main_prints_report(tmp_path, capsys):
    """main() with a config pointing to a nonexistent root still prints phase-1 report."""
    from scripts.validate_data import main

    cfg_path = _make_totalseg_config(tmp_path, root="/nonexistent/totalseg_root")
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, {})

    report = main([
        "--dataset-config", str(cfg_path),
        "--base-config", str(base_path),
        "--max-subjects", "2",
    ])
    assert "# Data Validation Report" in report
    assert "Root exists**: False" in report


def test_main_saves_to_file(tmp_path):
    from scripts.validate_data import main

    cfg_path = _make_totalseg_config(tmp_path, root="/nonexistent")
    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, {})
    out_file = tmp_path / "out" / "report.md"

    main([
        "--dataset-config", str(cfg_path),
        "--base-config", str(base_path),
        "--output", str(out_file),
    ])
    assert out_file.exists()
    content = out_file.read_text()
    assert "Data Validation Report" in content
