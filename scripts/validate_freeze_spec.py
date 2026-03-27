#!/usr/bin/env python3
"""Validate that experiment configs match the frozen ablation spec.

Reads the paper_ablation_freeze.yaml spec and checks that each condition's
resolved config matches the frozen training protocol. Reports drift.

Usage:
    python scripts/validate_freeze_spec.py
    python scripts/validate_freeze_spec.py --spec configs/experiments/paper_ablation_freeze.yaml
    python scripts/validate_freeze_spec.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.config import merge_dicts, load_yaml


def _get_nested(cfg: Dict, dotpath: str) -> Any:
    """Get a value from nested dict using dot-separated path."""
    keys = dotpath.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return _MISSING
        node = node[k]
    return node


_MISSING = object()


def load_freeze_spec(path: Path) -> Dict[str, Any]:
    """Load and parse the freeze spec YAML."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_condition_against_spec(
    condition_name: str,
    resolved_cfg: Dict[str, Any],
    spec: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check a single condition's config against the freeze spec.

    Returns list of check results.
    """
    results: List[Dict[str, Any]] = []
    training_spec = spec.get("training", {})
    model_spec = spec.get("model", {})

    # Map spec training keys to config paths
    spec_to_config = {
        "epochs": ("train", "epochs"),
        "lr": ("train", "lr"),
        "batch_size": ("data", "batch_size"),
        "loss_type": ("train", "loss_type"),
        "amp": ("train", "amp", "enabled"),
    }

    for spec_key, config_path in spec_to_config.items():
        expected = training_spec.get(spec_key)
        if expected is None:
            continue

        dotpath = ".".join(config_path)
        actual = _get_nested(resolved_cfg, dotpath)

        if actual is _MISSING:
            results.append({
                "condition": condition_name,
                "key": dotpath,
                "status": "WARN",
                "expected": expected,
                "actual": "<missing>",
                "reason": "Key not set in config (will use default)",
            })
        elif actual != expected:
            results.append({
                "condition": condition_name,
                "key": dotpath,
                "status": "FAIL",
                "expected": expected,
                "actual": actual,
                "reason": f"Drift detected: expected {expected}, got {actual}",
            })
        else:
            results.append({
                "condition": condition_name,
                "key": dotpath,
                "status": "PASS",
                "expected": expected,
                "actual": actual,
                "reason": "Matches spec",
            })

    # Model checks
    for model_key in ["name", "in_channels", "out_channels", "channels", "strides", "num_res_units"]:
        expected = model_spec.get(model_key)
        if expected is None:
            continue
        dotpath = f"model.{model_key}"
        actual = _get_nested(resolved_cfg, dotpath)

        if actual is _MISSING:
            results.append({
                "condition": condition_name,
                "key": dotpath,
                "status": "WARN",
                "expected": expected,
                "actual": "<missing>",
                "reason": "Key not set in config",
            })
        elif actual != expected:
            results.append({
                "condition": condition_name,
                "key": dotpath,
                "status": "FAIL",
                "expected": expected,
                "actual": actual,
                "reason": f"Drift: expected {expected}, got {actual}",
            })
        else:
            results.append({
                "condition": condition_name,
                "key": dotpath,
                "status": "PASS",
                "expected": expected,
                "actual": actual,
                "reason": "Matches spec",
            })

    return results


def validate_spec_completeness(spec: Dict[str, Any]) -> List[str]:
    """Check that the spec itself is well-formed."""
    warnings: List[str] = []

    if "conditions" not in spec:
        warnings.append("spec missing 'conditions' list")
    elif not spec["conditions"]:
        warnings.append("spec 'conditions' list is empty")

    if "seeds" not in spec:
        warnings.append("spec missing 'seeds' list")

    if "training" not in spec:
        warnings.append("spec missing 'training' section")

    if "model" not in spec:
        warnings.append("spec missing 'model' section")

    if "metrics" not in spec:
        warnings.append("spec missing 'metrics' section")

    # Check condition method configs exist
    repo_root = Path(__file__).resolve().parents[1]
    for cond in spec.get("conditions", []):
        method_cfg = cond.get("method_config")
        if method_cfg:
            p = repo_root / method_cfg
            if not p.exists():
                warnings.append(f"Condition '{cond['name']}': method config not found: {method_cfg}")

    return warnings


def format_report(
    results: List[Dict[str, Any]],
    spec_warnings: List[str],
    spec: Dict[str, Any],
) -> str:
    """Format validation results as markdown."""
    lines: List[str] = []
    lines.append("# Freeze Spec Validation Report\n")
    lines.append(f"**Spec version:** {spec.get('spec_version', '?')}")
    lines.append(f"**Frozen at:** {spec.get('frozen_at', '?')}\n")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    warned = sum(1 for r in results if r["status"] == "WARN")
    overall = "PASS" if failed == 0 else "FAIL"

    lines.append(f"**Overall: {overall}** ({passed} passed, {failed} failed, {warned} warnings)\n")

    if spec_warnings:
        lines.append("## Spec Completeness Warnings\n")
        for w in spec_warnings:
            lines.append(f"- {w}")
        lines.append("")

    if failed > 0:
        lines.append("## Drift Detected\n")
        lines.append("| Condition | Key | Expected | Actual |")
        lines.append("|-----------|-----|----------|--------|")
        for r in results:
            if r["status"] == "FAIL":
                lines.append(
                    f"| {r['condition']} | `{r['key']}` | "
                    f"`{r['expected']}` | `{r['actual']}` |"
                )
        lines.append("")

    lines.append("## Full Results\n")
    lines.append("| Condition | Key | Status | Expected | Actual |")
    lines.append("|-----------|-----|--------|----------|--------|")
    for r in results:
        lines.append(
            f"| {r['condition']} | `{r['key']}` | {r['status']} | "
            f"`{r['expected']}` | `{r['actual']}` |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate experiment configs against frozen ablation spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_freeze_spec.py
  python scripts/validate_freeze_spec.py --spec configs/experiments/paper_ablation_freeze.yaml
  python scripts/validate_freeze_spec.py --output-dir outputs/freeze_validation
        """,
    )
    p.add_argument(
        "--spec",
        default="configs/experiments/paper_ablation_freeze.yaml",
        help="Path to freeze spec YAML",
    )
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--json-only", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / args.spec
    base_path = repo_root / args.base_config

    if not spec_path.exists():
        print(f"ERROR: Freeze spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = load_freeze_spec(spec_path)
    spec_warnings = validate_spec_completeness(spec)

    all_results: List[Dict[str, Any]] = []

    for cond in spec.get("conditions", []):
        cond_name = cond["name"]
        method_config = cond.get("method_config")
        if not method_config:
            continue

        method_path = repo_root / method_config
        if not method_path.exists():
            all_results.append({
                "condition": cond_name,
                "key": "method_config",
                "status": "FAIL",
                "expected": method_config,
                "actual": "<not found>",
                "reason": "Method config file does not exist",
            })
            continue

        # Resolve config
        base = load_yaml(base_path) if base_path.exists() else {}
        method = load_yaml(method_path)
        resolved = merge_dicts(base, method)

        results = validate_condition_against_spec(cond_name, resolved, spec)
        all_results.extend(results)

    report = format_report(all_results, spec_warnings, spec)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "freeze_validation.json").write_text(
            json.dumps({"results": all_results, "spec_warnings": spec_warnings},
                       indent=2, default=str),
            encoding="utf-8",
        )
        if not args.json_only:
            (out / "freeze_validation.md").write_text(report, encoding="utf-8")
            print(report)
        else:
            print(json.dumps({"results": all_results}, indent=2, default=str))
    else:
        if args.json_only:
            print(json.dumps({"results": all_results}, indent=2, default=str))
        else:
            print(report)

    if any(r["status"] == "FAIL" for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
