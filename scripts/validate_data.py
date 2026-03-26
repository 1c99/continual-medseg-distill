#!/usr/bin/env python3
"""Data validation and smoke-report script.

Validates that dataset files exist on disk and optionally loads a sample
of subjects through the actual dataset adapter to check shapes and labels.

Usage examples:
    python scripts/validate_data.py --dataset-config configs/datasets/totalseg_example.yaml
    python scripts/validate_data.py --dataset-config configs/datasets/brats21_example.yaml --max-subjects 5
    python scripts/validate_data.py --dataset-config configs/datasets/acdc_example.yaml --output report.md
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path so `from src.*` imports work when invoked as a script.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

from src.utils.config import load_yaml, merge_dicts
from src.data.registry import _load_ids_from_split_manifest


# ---------------------------------------------------------------------------
# Per-source file-existence checkers
# ---------------------------------------------------------------------------

def _expected_files_totalseg(root: Path, sid: str, organ: str) -> List[Path]:
    """Return list of files a TotalSeg subject should have."""
    base = root / sid
    return [
        base / "ct.nii.gz",
        base / "segmentations" / f"{organ}.nii.gz",
    ]


def _expected_files_brats21(root: Path, sid: str, layout: str = "per_case") -> List[Path]:
    """Return list of files a BraTS21 subject should have."""
    if layout == "flat":
        return [
            root / "imagesTr" / f"{sid}_t1_0000.nii.gz",
            root / "imagesTr" / f"{sid}_t1ce_0000.nii.gz",
            root / "etc" / "images_flair" / f"{sid}_0000.nii.gz",
            root / "etc" / "images_t2" / f"{sid}_0000.nii.gz",
            root / "etc" / "labels_4cls" / f"{sid}_seg.nii.gz",
        ]
    base = root / sid
    files = [base / f"{sid}_{mod}.nii.gz" for mod in ("t1", "t1ce", "t2", "flair")]
    files.append(base / f"{sid}_seg.nii.gz")
    return files


def _expected_files_acdc(root: Path, sid: str) -> List[Path]:
    """Return list of files an ACDC subject should have.

    sid format: patientXXX_frameYY
    """
    parts = sid.rsplit("_", 1)
    if len(parts) != 2:
        return []
    patient, frame = parts
    base = root / patient
    return [
        base / f"{frame}.nii.gz",
        base / f"{frame}_gt.nii.gz",
    ]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_dataset_config(dataset_config_path: str, base_config_path: str) -> Dict[str, Any]:
    """Load and merge base + dataset configs, return the merged dict."""
    base_cfg = load_yaml(base_config_path) if Path(base_config_path).exists() else {}
    ds_cfg = load_yaml(dataset_config_path)
    return merge_dicts(base_cfg, ds_cfg)


def resolve_source_config(cfg: Dict[str, Any]):
    """Return (source, source_cfg, root, split_manifest_path, train_ids, val_ids)."""
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source")
    if not source:
        raise ValueError("Config missing data.source")

    source_cfg = data_cfg.get(source, {})
    root = source_cfg.get("root")
    split_manifest = source_cfg.get("split_manifest")

    train_ids: List[str] = []
    val_ids: List[str] = []

    if split_manifest:
        manifest_path = Path(split_manifest)
        # resolve relative paths from repo root
        if not manifest_path.is_absolute():
            manifest_path = Path(_REPO_ROOT) / manifest_path
        if manifest_path.exists():
            train_ids, val_ids = _load_ids_from_split_manifest(manifest_path)
    else:
        manifest_path = None
        train_ids = [str(x) for x in source_cfg.get("train_ids", [])]
        val_ids = [str(x) for x in source_cfg.get("val_ids", [])]

    return source, source_cfg, root, manifest_path, train_ids, val_ids


# ---------------------------------------------------------------------------
# Phase 1: path validation
# ---------------------------------------------------------------------------

def validate_paths(
    source: str,
    source_cfg: Dict[str, Any],
    root: Optional[str],
    manifest_path: Optional[Path],
    train_ids: List[str],
    val_ids: List[str],
) -> Dict[str, Any]:
    """Check root dir, manifest, and per-subject file existence.

    Returns a dict with validation results.
    """
    result: Dict[str, Any] = {
        "source": source,
        "root": root,
        "root_exists": False,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_exists": None,
        "total_ids": 0,
        "found_on_disk": 0,
        "missing_ids": [],
        "missing_files": {},
    }

    if manifest_path is not None:
        result["manifest_exists"] = manifest_path.exists()

    if not root:
        result["error"] = f"data.{source}.root not set in config"
        return result

    root_path = Path(root)
    result["root_exists"] = root_path.exists()

    all_ids = train_ids + val_ids
    result["total_ids"] = len(all_ids)

    if not root_path.exists():
        result["missing_ids"] = all_ids
        return result

    organ = source_cfg.get("organ", "liver") if source == "totalseg" else None
    layout = source_cfg.get("layout", "per_case") if source == "brats21" else None

    for sid in all_ids:
        if source == "totalseg":
            expected = _expected_files_totalseg(root_path, sid, organ)
        elif source == "brats21":
            expected = _expected_files_brats21(root_path, sid, layout=layout)
        elif source == "acdc":
            expected = _expected_files_acdc(root_path, sid)
        else:
            expected = []

        missing = [str(f) for f in expected if not f.exists()]
        if missing:
            result["missing_ids"].append(sid)
            result["missing_files"][sid] = missing
        else:
            result["found_on_disk"] += 1

    return result


# ---------------------------------------------------------------------------
# Phase 2: sample loading
# ---------------------------------------------------------------------------

def load_samples(
    source: str,
    source_cfg: Dict[str, Any],
    train_ids: List[str],
    val_ids: List[str],
    max_subjects: int,
) -> Dict[str, Any]:
    """Load up to max_subjects through the real dataset adapter.

    Returns loading stats.
    """
    all_ids = (train_ids + val_ids)[:max_subjects]
    result: Dict[str, Any] = {
        "attempted": len(all_ids),
        "loaded": 0,
        "failed": 0,
        "errors": [],
        "shapes": [],
        "label_values": set(),
    }

    if not all_ids:
        return result

    root = source_cfg.get("root")
    if not root or not Path(root).exists():
        result["errors"].append(f"Root dir does not exist: {root}")
        result["failed"] = len(all_ids)
        return result

    # Build a dataset instance for the requested source
    try:
        ds = _build_dataset(source, source_cfg, all_ids)
    except Exception as exc:
        result["errors"].append(f"Dataset init failed: {exc}")
        result["failed"] = len(all_ids)
        return result

    for i in range(len(ds)):
        try:
            sample = ds[i]
            image = sample["image"]
            label = sample["label"]
            result["shapes"].append(tuple(image.shape))
            unique_labels = set(label.unique().tolist())
            result["label_values"] |= unique_labels
            result["loaded"] += 1
        except Exception as exc:
            result["failed"] += 1
            result["errors"].append(f"{all_ids[i]}: {exc}")

    return result


def _build_dataset(source: str, source_cfg: Dict[str, Any], ids: List[str]):
    """Instantiate the right dataset adapter for the given source."""
    root = source_cfg["root"]
    if source == "totalseg":
        from src.data.totalseg import TotalSegmentatorDataset
        return TotalSegmentatorDataset(
            root=root,
            split_ids=ids,
            organ=source_cfg.get("organ", "liver"),
            target_shape=tuple(source_cfg.get("shape", [128, 128, 128])),
        )
    elif source == "brats21":
        from src.data.brats21 import Brats21Dataset
        return Brats21Dataset(
            root=root,
            split_ids=ids,
            target_shape=tuple(source_cfg.get("shape", [128, 128, 128])),
            normalize_per_channel=bool(source_cfg.get("normalize_per_channel", True)),
            layout=source_cfg.get("layout", "per_case"),
        )
    elif source == "acdc":
        from src.data.acdc import ACDCDataset
        return ACDCDataset(
            root=root,
            split_ids=ids,
            target_shape=tuple(source_cfg.get("shape", [16, 160, 160])),
            normalize=bool(source_cfg.get("normalize", True)),
        )
    else:
        raise ValueError(f"Unsupported source for sample loading: {source}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    path_result: Dict[str, Any],
    load_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce a markdown smoke report."""
    lines: List[str] = []
    lines.append("# Data Validation Report")
    lines.append("")

    # Phase 1
    lines.append("## Phase 1: Path Validation")
    lines.append("")
    lines.append(f"- **Source**: `{path_result['source']}`")
    lines.append(f"- **Root**: `{path_result['root']}`")
    lines.append(f"- **Root exists**: {path_result['root_exists']}")

    if path_result["manifest_path"]:
        lines.append(f"- **Split manifest**: `{path_result['manifest_path']}`")
        lines.append(f"- **Manifest exists**: {path_result['manifest_exists']}")

    if "error" in path_result:
        lines.append(f"- **Error**: {path_result['error']}")
    else:
        total = path_result["total_ids"]
        found = path_result["found_on_disk"]
        missing_count = len(path_result["missing_ids"])
        lines.append(f"- **Total subject IDs**: {total}")
        lines.append(f"- **Found on disk**: {found}")
        lines.append(f"- **Missing**: {missing_count}")

        if path_result["missing_ids"]:
            lines.append("")
            lines.append("### Missing subjects")
            lines.append("")
            for sid in path_result["missing_ids"][:20]:
                files = path_result["missing_files"].get(sid, [])
                files_str = ", ".join(f"`{f}`" for f in files[:3])
                if len(files) > 3:
                    files_str += f" (+{len(files)-3} more)"
                lines.append(f"- `{sid}`: {files_str}")
            if len(path_result["missing_ids"]) > 20:
                lines.append(f"- ... and {len(path_result['missing_ids'])-20} more")

    lines.append("")

    # Phase 2
    if load_result is not None:
        lines.append("## Phase 2: Sample Loading")
        lines.append("")
        lines.append(f"- **Attempted**: {load_result['attempted']}")
        lines.append(f"- **Loaded**: {load_result['loaded']}")
        lines.append(f"- **Failed**: {load_result['failed']}")

        if load_result["errors"]:
            lines.append("")
            lines.append("### Errors")
            lines.append("")
            for err in load_result["errors"][:10]:
                lines.append(f"- {err}")
            if len(load_result["errors"]) > 10:
                lines.append(f"- ... and {len(load_result['errors'])-10} more")

        if load_result["shapes"]:
            import numpy as np
            shapes_arr = [s for s in load_result["shapes"]]
            # Report per-dimension stats
            dims = list(zip(*shapes_arr))
            lines.append("")
            lines.append("### Volume shapes")
            lines.append("")
            lines.append(f"- **Count**: {len(shapes_arr)}")
            lines.append(f"- **Min shape**: {tuple(int(min(d)) for d in dims)}")
            lines.append(f"- **Max shape**: {tuple(int(max(d)) for d in dims)}")
            medians = tuple(int(sorted(d)[len(d)//2]) for d in dims)
            lines.append(f"- **Median shape**: {medians}")

        label_vals = load_result.get("label_values", set())
        if label_vals:
            sorted_labels = sorted(label_vals)
            lines.append("")
            lines.append("### Label values")
            lines.append("")
            lines.append(f"- **Unique values across loaded samples**: `{sorted_labels}`")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset files and produce a smoke report."
    )
    parser.add_argument(
        "--dataset-config", required=True,
        help="Path to dataset YAML config (e.g. configs/datasets/totalseg_example.yaml)",
    )
    parser.add_argument(
        "--base-config", default="configs/base.yaml",
        help="Path to base config YAML for merging (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=10,
        help="Max subjects to actually load in phase 2 (default: 10)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save report to file; if omitted print to stdout",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> str:
    """Run validation and return the markdown report string."""
    args = parse_args(argv)

    cfg = load_dataset_config(args.dataset_config, args.base_config)
    source, source_cfg, root, manifest_path, train_ids, val_ids = resolve_source_config(cfg)

    # Phase 1
    path_result = validate_paths(source, source_cfg, root, manifest_path, train_ids, val_ids)

    # Phase 2 (only if root exists)
    load_result = None
    if path_result["root_exists"] and args.max_subjects > 0:
        load_result = load_samples(source, source_cfg, train_ids, val_ids, args.max_subjects)

    report = generate_report(path_result, load_result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report saved to {out_path}")
    else:
        print(report)

    return report


if __name__ == "__main__":
    main()
