#!/usr/bin/env python3
"""Offline-ready statistical report generator.

Reads metrics CSV files from completed runs and produces:
- Mean/std per metric
- Bootstrap confidence intervals (when enough samples)
- Paired comparison summary template
- Explicit caveats when data is insufficient

Usage:
    python scripts/stats_report.py \\
        --metrics outputs/baselines/clean_v1_5ep/*/metrics.csv \\
        --output-dir outputs/stats_report

    python scripts/stats_report.py --help
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def read_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    """Read a metrics CSV and return list of row dicts with numeric conversion."""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if v is None or v == "":
                out[k] = None
                continue
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except (ValueError, TypeError):
                out[k] = v
        parsed.append(out)
    return parsed


def last_epoch_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract metrics from the last epoch row."""
    if not rows:
        return {}
    return dict(rows[-1])


def aggregate_metrics(
    per_run: List[Dict[str, Any]],
    keys: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, min, max, count for each metric across runs.

    Args:
        per_run: List of metric dicts (one per run, typically last-epoch).
        keys: Metric keys to aggregate. If None, auto-detect numeric keys.

    Returns:
        {metric_key: {mean, std, min, max, count, values}}
    """
    if not per_run:
        return {}

    if keys is None:
        keys = []
        for row in per_run:
            for k, v in row.items():
                if k not in keys and isinstance(v, (int, float)) and not math.isnan(v):
                    keys.append(k)

    result: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        values = []
        for row in per_run:
            v = row.get(key)
            if isinstance(v, (int, float)) and not math.isnan(v):
                values.append(float(v))

        if not values:
            result[key] = {"mean": math.nan, "std": math.nan, "min": math.nan,
                           "max": math.nan, "count": 0, "values": []}
            continue

        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = variance ** 0.5
        else:
            std = 0.0

        result[key] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "count": n,
            "values": values,
        }

    return result


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval.

    Returns:
        {lower, upper, ci_level} or {caveat} if insufficient data.
    """
    import random

    if len(values) < 3:
        return {
            "caveat": f"Insufficient data for bootstrap CI ({len(values)} samples, need >= 3)",
            "ci_level": ci,
        }

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - ci
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    return {
        "lower": means[lo_idx],
        "upper": means[hi_idx],
        "ci_level": ci,
    }


def paired_comparison(
    group_a: List[float],
    group_b: List[float],
    name_a: str = "A",
    name_b: str = "B",
) -> Dict[str, Any]:
    """Generate a paired comparison summary between two groups.

    Returns a template with descriptive stats and caveat if insufficient.
    """
    result: Dict[str, Any] = {
        "group_a": name_a,
        "group_b": name_b,
        "n_a": len(group_a),
        "n_b": len(group_b),
    }

    if group_a:
        result["mean_a"] = sum(group_a) / len(group_a)
    else:
        result["mean_a"] = math.nan

    if group_b:
        result["mean_b"] = sum(group_b) / len(group_b)
    else:
        result["mean_b"] = math.nan

    if group_a and group_b:
        result["delta"] = result["mean_b"] - result["mean_a"]
    else:
        result["delta"] = math.nan

    min_n = min(len(group_a), len(group_b))
    if min_n < 3:
        result["caveat"] = (
            f"Insufficient paired samples (min n={min_n}). "
            "Statistical significance testing requires >= 3 paired observations per group. "
            "Report descriptive stats only."
        )
    elif min_n < 30:
        result["caveat"] = (
            f"Small sample size (n={min_n}). "
            "Consider Wilcoxon signed-rank test or permutation test rather than t-test. "
            "Results should be interpreted with caution."
        )
    else:
        result["note"] = "Sufficient samples for standard statistical tests."

    return result


def format_markdown(
    aggregated: Dict[str, Dict[str, Dict[str, Any]]],
    comparisons: List[Dict[str, Any]] | None = None,
    caveats: List[str] | None = None,
) -> str:
    """Generate markdown stats report.

    Args:
        aggregated: {condition_name: {metric: {mean, std, ...}}}
        comparisons: optional paired comparison results
        caveats: global caveats
    """
    lines: List[str] = []
    lines.append("# Statistical Summary Report\n")

    if caveats:
        lines.append("## Caveats\n")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    # Per-condition summary
    for cond_name, metrics in aggregated.items():
        lines.append(f"## {cond_name}\n")
        lines.append("| Metric | Mean | Std | Min | Max | N |")
        lines.append("|--------|------|-----|-----|-----|---|")
        for key, stats in metrics.items():
            if key == "epoch":
                continue
            lines.append(
                f"| {key} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |"
            )
        lines.append("")

    # Paired comparisons
    if comparisons:
        lines.append("## Paired Comparisons\n")
        for comp in comparisons:
            a, b = comp["group_a"], comp["group_b"]
            lines.append(f"### {a} vs {b}\n")
            lines.append(f"- Mean {a}: {comp.get('mean_a', 'N/A'):.4f}" if isinstance(comp.get("mean_a"), float) else f"- Mean {a}: N/A")
            lines.append(f"- Mean {b}: {comp.get('mean_b', 'N/A'):.4f}" if isinstance(comp.get("mean_b"), float) else f"- Mean {b}: N/A")
            if isinstance(comp.get("delta"), float) and not math.isnan(comp["delta"]):
                lines.append(f"- Delta ({b} - {a}): {comp['delta']:.4f}")
            if "caveat" in comp:
                lines.append(f"- **Caveat:** {comp['caveat']}")
            if "note" in comp:
                lines.append(f"- Note: {comp['note']}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate statistical summary from metrics CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single condition (glob)
  python scripts/stats_report.py --metrics outputs/seed_*/metrics.csv

  # Multiple named conditions
  python scripts/stats_report.py \\
      --conditions finetune:outputs/baselines/*/finetune/metrics.csv \\
                   distill:outputs/baselines/*/distill/metrics.csv

  # With output directory
  python scripts/stats_report.py --metrics outputs/run1/metrics.csv --output-dir outputs/stats
        """,
    )
    p.add_argument(
        "--metrics", nargs="*", default=[],
        help="Metrics CSV files (unnamed condition)",
    )
    p.add_argument(
        "--conditions", nargs="*", default=[],
        help="Named conditions as name:glob_pattern (e.g. finetune:outputs/*/finetune/metrics.csv)",
    )
    p.add_argument(
        "--metric-keys", nargs="*", default=None,
        help="Specific metric columns to report (default: auto-detect)",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CIs")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    args = p.parse_args()

    import glob as globmod

    # Collect conditions
    conditions: Dict[str, List[Path]] = {}

    if args.metrics:
        paths = []
        for pattern in args.metrics:
            matched = globmod.glob(pattern, recursive=True)
            paths.extend(Path(m) for m in matched)
        if not paths:
            # Try as literal paths
            paths = [Path(m) for m in args.metrics if Path(m).exists()]
        if paths:
            conditions["default"] = paths

    for spec in args.conditions:
        if ":" not in spec:
            print(f"WARNING: ignoring malformed condition spec: {spec}", file=sys.stderr)
            continue
        name, pattern = spec.split(":", 1)
        matched = globmod.glob(pattern, recursive=True)
        conditions[name] = [Path(m) for m in matched]

    if not conditions:
        print("No metrics files found. Provide --metrics or --conditions.", file=sys.stderr)
        p.print_help()
        sys.exit(1)

    # Aggregate per condition
    caveats: List[str] = []
    aggregated: Dict[str, Dict[str, Dict[str, Any]]] = {}
    per_condition_values: Dict[str, Dict[str, List[float]]] = {}

    for cond_name, paths in conditions.items():
        existing = [p for p in paths if p.exists()]
        if not existing:
            caveats.append(f"Condition '{cond_name}': no metrics files found")
            continue
        if len(existing) == 1:
            caveats.append(
                f"Condition '{cond_name}': only 1 run found. "
                "Standard deviation and CI are not meaningful."
            )

        per_run = [last_epoch_metrics(read_metrics_csv(p)) for p in existing]
        per_run = [r for r in per_run if r]  # filter empty

        if not per_run:
            caveats.append(f"Condition '{cond_name}': all metrics files empty")
            continue

        agg = aggregate_metrics(per_run, keys=args.metric_keys)
        aggregated[cond_name] = agg

        # Store values for paired comparisons and bootstrap
        vals: Dict[str, List[float]] = {}
        for key, stats in agg.items():
            vals[key] = stats.get("values", [])
        per_condition_values[cond_name] = vals

        # Bootstrap CIs
        if args.bootstrap:
            for key in agg:
                values = agg[key].get("values", [])
                if values:
                    ci_result = bootstrap_ci(values, n_bootstrap=args.bootstrap_n, ci=args.ci)
                    agg[key]["bootstrap_ci"] = ci_result

    # Paired comparisons (all pairs)
    comparisons: List[Dict[str, Any]] = []
    cond_names = list(per_condition_values.keys())
    primary_metric = "val_dice_mean"
    if args.metric_keys and len(args.metric_keys) > 0:
        primary_metric = args.metric_keys[0]

    for i in range(len(cond_names)):
        for j in range(i + 1, len(cond_names)):
            a_name, b_name = cond_names[i], cond_names[j]
            a_vals = per_condition_values[a_name].get(primary_metric, [])
            b_vals = per_condition_values[b_name].get(primary_metric, [])
            comp = paired_comparison(a_vals, b_vals, a_name, b_name)
            comp["metric"] = primary_metric
            comparisons.append(comp)

    # Output
    report = {
        "conditions": {k: len(v) for k, v in conditions.items()},
        "caveats": caveats,
        "aggregated": {
            cond: {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                   for k, v in metrics.items()}
            for cond, metrics in aggregated.items()
        },
        "comparisons": comparisons,
    }

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stats_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        if not args.json_only:
            md = format_markdown(aggregated, comparisons, caveats)
            (out / "stats_report.md").write_text(md, encoding="utf-8")
            print(md)
        else:
            print(json.dumps(report, indent=2, default=str))

        # Also write aggregated CSV
        csv_path = out / "stats_summary.csv"
        all_rows: List[Dict[str, Any]] = []
        for cond, metrics in aggregated.items():
            for key, stats in metrics.items():
                all_rows.append({
                    "condition": cond,
                    "metric": key,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "count": stats["count"],
                })
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(all_rows)
    else:
        if args.json_only:
            print(json.dumps(report, indent=2, default=str))
        else:
            md = format_markdown(aggregated, comparisons, caveats)
            print(md)


if __name__ == "__main__":
    main()
