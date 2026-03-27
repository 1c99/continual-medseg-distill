#!/usr/bin/env python3
"""Experiment fairness guardrail.

Validates that compared experimental conditions share identical training
parameters (epochs, steps, dataset split, seed policy, augmentation flags,
batch size, etc.) so that differences in results are attributable to the
method, not accidental config drift.

Usage:
    python scripts/check_experiment_fairness.py \\
        --configs configs/methods/finetune.yaml \\
                  configs/methods/distill_replay_ewc.yaml \\
                  configs/methods/distill_replay_ewc_ortholora.yaml \\
        --base-config configs/base.yaml \\
        --output-dir outputs/fairness_report

    python scripts/check_experiment_fairness.py --help
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


# Keys that MUST be identical across all compared conditions for a fair comparison
PARITY_KEYS: List[Tuple[str, ...]] = [
    ("train", "epochs"),
    ("train", "lr"),
    ("train", "max_steps_per_epoch"),
    ("train", "loss_type"),
    ("train", "amp"),
    ("data", "source"),
    ("data", "batch_size"),
    ("data", "val_batch_size"),
    ("data", "synthetic", "train_samples"),
    ("data", "synthetic", "val_samples"),
    ("data", "synthetic", "shape"),
    ("data", "synthetic", "num_classes"),
    ("model", "name"),
    ("model", "in_channels"),
    ("model", "out_channels"),
    ("model", "channels"),
    ("model", "strides"),
    ("model", "num_res_units"),
    ("experiment", "seed"),
    ("output", "best_metric"),
    ("output", "best_mode"),
]

# Keys that may legitimately differ between methods
ALLOWED_DIFF_PREFIXES = [
    ("method",),
    ("model", "lora"),
]


def _get_nested(cfg: Dict, keys: Tuple[str, ...]) -> Any:
    """Retrieve a nested config value by key path. Returns _MISSING sentinel if absent."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return _MISSING
        node = node[k]
    return node


_MISSING = object()


def _dotpath(keys: Tuple[str, ...]) -> str:
    return ".".join(keys)


def resolve_method_config(
    base_config_path: Path,
    method_config_path: Path,
    dataset_config_path: Path | None = None,
) -> Dict[str, Any]:
    """Load and merge base + method (+ optional dataset) configs."""
    base = load_yaml(base_config_path)
    method = load_yaml(method_config_path)
    cfg = merge_dicts(base, method)
    if dataset_config_path:
        dataset = load_yaml(dataset_config_path)
        cfg = merge_dicts(cfg, dataset)
    return cfg


def check_parity(
    configs: Dict[str, Dict[str, Any]],
    parity_keys: List[Tuple[str, ...]] | None = None,
) -> List[Dict[str, Any]]:
    """Check that all configs agree on parity keys.

    Args:
        configs: {condition_name: resolved_config}
        parity_keys: list of key paths to check (defaults to PARITY_KEYS)

    Returns:
        List of check results, each with:
            key: dotpath string
            status: "PASS" | "FAIL" | "SKIP"
            values: {condition_name: value}
            reason: human-readable explanation
    """
    if parity_keys is None:
        parity_keys = PARITY_KEYS

    results: List[Dict[str, Any]] = []
    names = list(configs.keys())

    for key_path in parity_keys:
        dotpath = _dotpath(key_path)
        values = {}
        for name in names:
            v = _get_nested(configs[name], key_path)
            values[name] = v if v is not _MISSING else "<missing>"

        unique = set()
        for v in values.values():
            unique.add(json.dumps(v, sort_keys=True, default=str))

        if len(unique) == 1:
            # All the same (including all missing)
            first_val = next(iter(values.values()))
            if first_val == "<missing>":
                results.append({
                    "key": dotpath,
                    "status": "SKIP",
                    "values": values,
                    "reason": "Key absent in all configs (using defaults)",
                })
            else:
                results.append({
                    "key": dotpath,
                    "status": "PASS",
                    "values": values,
                    "reason": "Identical across all conditions",
                })
        else:
            results.append({
                "key": dotpath,
                "status": "FAIL",
                "values": values,
                "reason": f"Mismatch: {len(unique)} distinct values across {len(names)} conditions",
            })

    return results


def format_markdown_report(
    results: List[Dict[str, Any]],
    condition_names: List[str],
) -> str:
    """Generate a markdown fairness report."""
    lines: List[str] = []
    lines.append("# Experiment Fairness Report\n")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")

    overall = "PASS" if failed == 0 else "FAIL"
    lines.append(f"**Overall: {overall}** ({passed} passed, {failed} failed, {skipped} skipped)\n")
    lines.append(f"**Conditions compared:** {', '.join(condition_names)}\n")
    lines.append("---\n")

    if failed > 0:
        lines.append("## Failures\n")
        lines.append("| Key | Status | Values |")
        lines.append("|-----|--------|--------|")
        for r in results:
            if r["status"] == "FAIL":
                vals = " / ".join(
                    f"`{n}`: `{json.dumps(v, default=str)}`"
                    for n, v in r["values"].items()
                )
                lines.append(f"| `{r['key']}` | **FAIL** | {vals} |")
        lines.append("")

    lines.append("## Full Check Matrix\n")
    lines.append("| Key | Status | Reason |")
    lines.append("|-----|--------|--------|")
    for r in results:
        lines.append(f"| `{r['key']}` | {r['status']} | {r['reason']} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check experiment fairness: validate parity across conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare method configs against base
  python scripts/check_experiment_fairness.py \\
      --configs configs/methods/finetune.yaml \\
               configs/methods/distill_replay_ewc.yaml

  # With custom dataset config
  python scripts/check_experiment_fairness.py \\
      --configs configs/methods/finetune.yaml \\
               configs/methods/distill_replay_ewc_ortholora.yaml \\
      --dataset-config configs/datasets/totalseg_train_clean.yaml
        """,
    )
    p.add_argument(
        "--configs", nargs="+", required=True,
        help="Method config files to compare",
    )
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--output-dir", default=None, help="Directory for report output")
    p.add_argument("--json-only", action="store_true", help="Output JSON only (no markdown)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_path = repo_root / args.base_config
    dataset_path = (repo_root / args.dataset_config) if args.dataset_config else None

    configs: Dict[str, Dict[str, Any]] = {}
    for cfg_path_str in args.configs:
        cfg_path = repo_root / cfg_path_str
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
            sys.exit(1)
        name = cfg_path.stem
        configs[name] = resolve_method_config(base_path, cfg_path, dataset_path)

    results = check_parity(configs)

    # Output
    report_json = {
        "conditions": list(configs.keys()),
        "checks": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["status"] == "PASS"),
            "failed": sum(1 for r in results if r["status"] == "FAIL"),
            "skipped": sum(1 for r in results if r["status"] == "SKIP"),
            "overall": "PASS" if all(r["status"] != "FAIL" for r in results) else "FAIL",
        },
    }

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fairness_report.json").write_text(
            json.dumps(report_json, indent=2, default=str), encoding="utf-8"
        )
        if not args.json_only:
            md = format_markdown_report(results, list(configs.keys()))
            (out / "fairness_report.md").write_text(md, encoding="utf-8")
            print(md)
        else:
            print(json.dumps(report_json, indent=2, default=str))
    else:
        if args.json_only:
            print(json.dumps(report_json, indent=2, default=str))
        else:
            md = format_markdown_report(results, list(configs.keys()))
            print(md)

    if report_json["summary"]["overall"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
