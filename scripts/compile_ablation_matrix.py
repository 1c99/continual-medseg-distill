#!/usr/bin/env python3
"""Aggregate all ablation runs into a method × metric matrix with mean/std.

Usage:
    python scripts/compile_ablation_matrix.py outputs/ablations/
    python scripts/compile_ablation_matrix.py outputs/ablations/ --output ablation_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(v) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def compile_matrix(runs_dir: Path) -> dict[str, Any]:
    """Scan runs_dir for ablation run directories and aggregate results.

    Returns dict with 'matrix', 'methods', 'metrics', 'warnings'.
    """
    # method -> list of metric dicts
    method_runs: dict[str, list[dict]] = defaultdict(list)
    warnings: list[str] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        agg = run_dir / "aggregate_metrics.csv"
        rows = _read_csv(agg)
        if not rows:
            # Try per-method subdirs
            for method_dir in sorted(run_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                metrics_csv = method_dir / "metrics.csv"
                mrows = _read_csv(metrics_csv)
                if mrows:
                    last = mrows[-1]
                    last["method"] = method_dir.name
                    last["_run"] = run_dir.name
                    method_runs[method_dir.name].append(last)
            continue

        for row in rows:
            method = row.get("method", "unknown")
            row["_run"] = run_dir.name
            method_runs[method].append(row)

    if not method_runs:
        warnings.append(f"No runs found in {runs_dir}")
        return {"matrix": [], "methods": [], "metrics": [], "warnings": warnings}

    # Determine common metric keys
    all_keys: set[str] = set()
    for runs in method_runs.values():
        for r in runs:
            all_keys |= {k for k in r.keys() if k not in ("method", "_run", "status", "return_code", "command", "config_hash", "method_hash")}
    metric_keys = sorted(all_keys)

    # Build matrix: method -> metric -> [values]
    matrix_rows = []
    for method in sorted(method_runs.keys()):
        runs = method_runs[method]
        row: dict[str, Any] = {"method": method, "n_runs": len(runs)}
        for mk in metric_keys:
            vals = [_float(r.get(mk)) for r in runs]
            vals = [v for v in vals if v is not None]
            if vals:
                import statistics
                row[f"{mk}_mean"] = statistics.mean(vals)
                row[f"{mk}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            else:
                row[f"{mk}_mean"] = "NA"
                row[f"{mk}_std"] = "NA"
                warnings.append(f"{method}: metric '{mk}' has no valid values")
        matrix_rows.append(row)

    return {
        "matrix": matrix_rows,
        "methods": sorted(method_runs.keys()),
        "metrics": metric_keys,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description="Compile ablation matrix")
    parser.add_argument("runs_dir", help="Directory containing ablation run directories")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    result = compile_matrix(Path(args.runs_dir))

    if result["warnings"]:
        for w in result["warnings"]:
            print(f"[warn] {w}", file=sys.stderr)

    if not result["matrix"]:
        print("[error] No data to compile")
        return

    out_path = Path(args.output) if args.output else Path(args.runs_dir) / "ablation_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(result["matrix"][0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(result["matrix"])

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(f"[ok] Matrix: {out_path}")
    print(f"[ok] JSON: {json_path}")
    print(f"[ok] {len(result['methods'])} methods, {len(result['metrics'])} metrics")


if __name__ == "__main__":
    main()
