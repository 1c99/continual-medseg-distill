#!/usr/bin/env python3
"""Failure-case panel generator for qualitative analysis.

Identifies the worst-performing cases from evaluation metrics and produces:
- An indexed panel manifest (worst-k cases ranked by Dice or HD95)
- A markdown report template for qualitative figures

This is pipeline scaffolding: no heavy image processing is done.
It reads evaluation outputs and metadata, then generates manifests
and templates that can later be populated with actual visualizations.

Usage:
    python scripts/build_failure_panel.py \\
        --eval-dir outputs/smoke_AB_real \\
        --metric dice_mean --direction worst \\
        --top-k 10 --output-dir outputs/failure_panel

    python scripts/build_failure_panel.py --help
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


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read CSV with numeric type coercion."""
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


def find_eval_csvs(eval_dir: Path) -> List[Path]:
    """Discover task_eval_matrix.csv and metrics.csv files."""
    results: List[Path] = []
    for p in sorted(eval_dir.rglob("task_eval_matrix.csv")):
        results.append(p)
    for p in sorted(eval_dir.rglob("metrics.csv")):
        results.append(p)
    return results


def rank_cases(
    rows: List[Dict[str, Any]],
    metric_key: str,
    top_k: int = 10,
    direction: str = "worst",
) -> List[Dict[str, Any]]:
    """Rank rows by a metric, selecting worst or best cases.

    Args:
        rows: List of metric dicts.
        metric_key: Column to rank by.
        top_k: Number of cases to select.
        direction: "worst" (lowest Dice / highest HD95) or "best".

    Returns:
        List of ranked entries with rank, metric value, and source row.
    """
    # Determine sort direction based on metric type
    is_distance = "hd95" in metric_key.lower() or "hausdorff" in metric_key.lower()

    scored: List[tuple] = []
    for i, row in enumerate(rows):
        val = row.get(metric_key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        scored.append((float(val), i, row))

    if not scored:
        return []

    # For "worst": low Dice is bad, high HD95 is bad
    if direction == "worst":
        reverse = is_distance  # high HD95 = worst, so sort descending
    else:
        reverse = not is_distance  # high Dice = best, sort descending

    scored.sort(key=lambda x: x[0], reverse=reverse)

    ranked: List[Dict[str, Any]] = []
    for rank, (val, idx, row) in enumerate(scored[:top_k], 1):
        entry = {
            "rank": rank,
            "metric_key": metric_key,
            "metric_value": val,
            "direction": direction,
            "source_index": idx,
            "row": row,
        }
        ranked.append(entry)

    return ranked


def build_manifest(
    ranked: List[Dict[str, Any]],
    eval_dir: Path,
    metric_key: str,
    direction: str,
) -> Dict[str, Any]:
    """Build a structured panel manifest."""
    return {
        "eval_dir": str(eval_dir),
        "metric_key": metric_key,
        "direction": direction,
        "num_cases": len(ranked),
        "cases": ranked,
        "placeholder_note": (
            "This manifest indexes failure cases by metric rank. "
            "Actual image panels should be generated separately using the "
            "source_index and row metadata to locate prediction/GT volumes."
        ),
    }


def format_markdown_panel(
    manifest: Dict[str, Any],
    output_dir: Path | None = None,
) -> str:
    """Generate markdown report template for failure panel."""
    lines: List[str] = []
    lines.append("# Failure Case Panel\n")
    lines.append(f"**Eval directory:** `{manifest['eval_dir']}`")
    lines.append(f"**Metric:** `{manifest['metric_key']}` ({manifest['direction']})")
    lines.append(f"**Cases:** {manifest['num_cases']}\n")

    lines.append("---\n")
    lines.append("## Ranked Cases\n")
    lines.append("| Rank | Metric Value | Source Details |")
    lines.append("|------|-------------|----------------|")

    for case in manifest["cases"]:
        row = case["row"]
        # Build a compact source description
        details_parts = []
        for k in ["trained_on", "evaluated_on", "epoch", "subject_id"]:
            if k in row and row[k] is not None:
                details_parts.append(f"{k}={row[k]}")
        details = ", ".join(details_parts) if details_parts else f"index={case['source_index']}"
        lines.append(
            f"| {case['rank']} | {case['metric_value']:.4f} | {details} |"
        )

    lines.append("")
    lines.append("---\n")
    lines.append("## Figure Template\n")
    lines.append("For each ranked case, generate a panel showing:\n")
    lines.append("1. **Input** — CT slice (axial, sagittal, or coronal)")
    lines.append("2. **Ground Truth** — Overlaid segmentation mask")
    lines.append("3. **Prediction** — Overlaid predicted mask")
    lines.append("4. **Error Map** — FP/FN voxel overlay\n")
    lines.append("```")
    lines.append("# Placeholder: generate panels with")
    lines.append("# python scripts/render_panel.py --manifest failure_panel_manifest.json")
    lines.append("```\n")

    if manifest.get("placeholder_note"):
        lines.append(f"> **Note:** {manifest['placeholder_note']}\n")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build failure-case panel manifest for qualitative analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From task eval matrix
  python scripts/build_failure_panel.py \\
      --eval-dir outputs/smoke_AB_real --metric val_dice_mean --top-k 5

  # Worst HD95 cases
  python scripts/build_failure_panel.py \\
      --eval-dir outputs/baselines/clean_v1_5ep/finetune \\
      --metric val_hd95_mean --top-k 10

  # Custom CSV input
  python scripts/build_failure_panel.py \\
      --csv outputs/custom_eval.csv --metric dice_mean
        """,
    )
    p.add_argument("--eval-dir", default=None, help="Evaluation output directory to scan")
    p.add_argument("--csv", default=None, help="Direct path to a metrics CSV")
    p.add_argument("--metric", default="val_dice_mean", help="Metric key to rank by")
    p.add_argument("--direction", choices=["worst", "best"], default="worst")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--json-only", action="store_true")
    args = p.parse_args()

    # Collect rows from all sources
    all_rows: List[Dict[str, Any]] = []
    eval_dir_str = args.eval_dir or "."

    if args.csv:
        csv_path = Path(args.csv)
        if csv_path.exists():
            all_rows.extend(read_csv(csv_path))
        else:
            print(f"WARNING: CSV not found: {csv_path}", file=sys.stderr)

    if args.eval_dir:
        eval_dir = Path(args.eval_dir)
        if eval_dir.exists():
            for csv_path in find_eval_csvs(eval_dir):
                all_rows.extend(read_csv(csv_path))
        else:
            print(f"WARNING: eval-dir not found: {eval_dir}", file=sys.stderr)

    if not all_rows:
        print(
            "No data found. Generating empty manifest template.\n"
            "Provide --eval-dir or --csv with actual evaluation outputs.",
            file=sys.stderr,
        )
        manifest = build_manifest([], Path(eval_dir_str), args.metric, args.direction)
        manifest["caveat"] = "No evaluation data available. This is a template only."
    else:
        ranked = rank_cases(all_rows, args.metric, args.top_k, args.direction)
        manifest = build_manifest(ranked, Path(eval_dir_str), args.metric, args.direction)

        if not ranked:
            manifest["caveat"] = (
                f"Metric '{args.metric}' not found in any rows. "
                f"Available columns: {sorted(set(k for r in all_rows for k in r.keys()))}"
            )

    # Output
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure_panel_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        if not args.json_only:
            md = format_markdown_panel(manifest, out)
            (out / "failure_panel.md").write_text(md, encoding="utf-8")
            print(md)
        else:
            print(json.dumps(manifest, indent=2, default=str))
    else:
        if args.json_only:
            print(json.dumps(manifest, indent=2, default=str))
        else:
            md = format_markdown_panel(manifest)
            print(md)


if __name__ == "__main__":
    main()
