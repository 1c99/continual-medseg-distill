#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DICE_CANDIDATES = [
    "dice_mean",
    "val_dice_mean",
    "mean_dice",
    "val_mean_dice",
]

FORGETTING_CANDIDATES = [
    "forgetting",
    "val_forgetting",
    "avg_forgetting",
    "mean_forgetting",
]


@dataclass
class MethodResult:
    method: str
    method_dir: Path
    metrics: dict[str, Any]
    checkpoint_best: Path | None
    checkpoint_last: Path | None


def _parse_scalar(v: str) -> Any:
    s = (v or "").strip()
    if s == "":
        return ""
    try:
        if any(ch in s.lower() for ch in [".", "e"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append({k: _parse_scalar(v if isinstance(v, str) else str(v)) for k, v in row.items()})
    return parsed


def _pick_metric_key(metrics: dict[str, Any], candidates: list[str], contains_token: str) -> str | None:
    keys = list(metrics.keys())
    for c in candidates:
        if c in metrics:
            return c
    for k in keys:
        lk = k.lower()
        if contains_token in lk and "mean" in lk:
            return k
    for k in keys:
        lk = k.lower()
        if contains_token in lk:
            return k
    return None


def _float_or_none(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v.strip() != "":
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _find_method_results(run_dir: Path) -> list[MethodResult]:
    results: list[MethodResult] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("result_bundle"):
            continue

        metrics_path = child / "metrics.csv"
        rows = _read_csv_rows(metrics_path)
        if not rows:
            continue

        checkpoints = child / "checkpoints"
        best = checkpoints / "best.pt"
        last = checkpoints / "last.pt"

        results.append(
            MethodResult(
                method=child.name,
                method_dir=child,
                metrics=rows[-1],
                checkpoint_best=best if best.exists() else None,
                checkpoint_last=last if last.exists() else None,
            )
        )
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _rank_desc(values: list[tuple[str, float | None]]) -> dict[str, int | None]:
    ordered = sorted(values, key=lambda x: float("-inf") if x[1] is None else x[1], reverse=True)
    out: dict[str, int | None] = {}
    rank = 1
    for method, val in ordered:
        if val is None:
            out[method] = None
            continue
        out[method] = rank
        rank += 1
    return out


def _rank_asc(values: list[tuple[str, float | None]]) -> dict[str, int | None]:
    present = [(m, v) for m, v in values if v is not None]
    ordered = sorted(present, key=lambda x: x[1])
    out: dict[str, int | None] = {m: None for m, _ in values}
    rank = 1
    for method, _ in ordered:
        out[method] = rank
        rank += 1
    return out


def _copy_checkpoint_refs(bundle_dir: Path, results: list[MethodResult]) -> list[dict[str, Any]]:
    ref_dir = bundle_dir / "checkpoint_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for r in results:
        for tag, p in [("best", r.checkpoint_best), ("last", r.checkpoint_last)]:
            ref_file = ref_dir / f"{r.method}_{tag}.txt"
            if p is None:
                ref_file.write_text("missing\n", encoding="utf-8")
                rows.append({"method": r.method, "checkpoint": tag, "exists": False, "path": ""})
                continue

            rel = p.relative_to(bundle_dir.parent) if bundle_dir.parent in p.parents else p
            ref_file.write_text(f"{rel}\n", encoding="utf-8")
            rows.append(
                {
                    "method": r.method,
                    "checkpoint": tag,
                    "exists": True,
                    "path": str(rel),
                    "ref_file": str(ref_file.relative_to(bundle_dir)),
                }
            )
    return rows


def package_results(run_dir: Path, output_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    bundle_dir = (output_dir.resolve() if output_dir else run_dir / "result_bundle")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    results = _find_method_results(run_dir)
    if not results:
        raise RuntimeError(f"No per-method metrics.csv found under: {run_dir}")

    dice_key = _pick_metric_key(results[0].metrics, DICE_CANDIDATES, "dice")
    forgetting_key = _pick_metric_key(results[0].metrics, FORGETTING_CANDIDATES, "forget")

    dice_values: list[tuple[str, float | None]] = []
    forgetting_values: list[tuple[str, float | None]] = []
    summary_rows: list[dict[str, Any]] = []

    for r in results:
        dice = _float_or_none(r.metrics.get(dice_key)) if dice_key else None
        forgetting = _float_or_none(r.metrics.get(forgetting_key)) if forgetting_key else None
        dice_values.append((r.method, dice))
        forgetting_values.append((r.method, forgetting))

    dice_rank = _rank_desc(dice_values)
    forgetting_rank = _rank_asc(forgetting_values)

    for r in results:
        row = {
            "method": r.method,
            "dice_key": dice_key or "",
            "dice_mean": _float_or_none(r.metrics.get(dice_key)) if dice_key else None,
            "rank_dice_mean": dice_rank.get(r.method),
            "forgetting_key": forgetting_key or "",
            "forgetting": _float_or_none(r.metrics.get(forgetting_key)) if forgetting_key else None,
            "rank_forgetting": forgetting_rank.get(r.method),
        }
        for k, v in r.metrics.items():
            if k not in row:
                row[k] = v
        summary_rows.append(row)

    summary_rows.sort(key=lambda x: (999999 if x["rank_dice_mean"] is None else x["rank_dice_mean"]))

    _write_csv(bundle_dir / "method_summary.csv", summary_rows)
    ckpt_rows = _copy_checkpoint_refs(bundle_dir, results)
    _write_csv(bundle_dir / "checkpoint_refs.csv", ckpt_rows)

    aggregate = run_dir / "aggregate_metrics.csv"
    if aggregate.exists():
        shutil.copy2(aggregate, bundle_dir / "aggregate_metrics.csv")

    caveats: list[str] = []
    if dice_key is None:
        caveats.append("No dice-like metric field found; dice ranking is unavailable.")
    if forgetting_key is None:
        caveats.append("No forgetting-like metric field found; forgetting ranking is unavailable.")
    else:
        missing_f = [m for m, v in forgetting_values if v is None]
        if missing_f:
            caveats.append(
                "Forgetting key was detected but missing for methods: " + ", ".join(missing_f)
            )

    md_lines = [
        "# Result Bundle Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Bundle directory: `{bundle_dir}`",
        f"- Methods found: {len(results)}",
        f"- Dice key: `{dice_key}`" if dice_key else "- Dice key: _not found_",
        f"- Forgetting key: `{forgetting_key}`" if forgetting_key else "- Forgetting key: _not found_",
        "",
        "## Ranking",
        "",
        "| method | dice_mean | rank_dice | forgetting | rank_forgetting |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        md_lines.append(
            "| {method} | {dice} | {dr} | {fg} | {fr} |".format(
                method=row["method"],
                dice="" if row["dice_mean"] is None else f"{row['dice_mean']:.6f}",
                dr="" if row["rank_dice_mean"] is None else row["rank_dice_mean"],
                fg="" if row["forgetting"] is None else f"{row['forgetting']:.6f}",
                fr="" if row["rank_forgetting"] is None else row["rank_forgetting"],
            )
        )

    if caveats:
        md_lines.extend(["", "## Caveats", ""])
        md_lines.extend([f"- {c}" for c in caveats])

    md_lines.extend(
        [
            "",
            "## Bundle files",
            "",
            "- `method_summary.csv`: unified metrics + ranking fields",
            "- `checkpoint_refs.csv`: references to per-method `best.pt` / `last.pt`",
            "- `checkpoint_refs/*.txt`: one reference text file per key checkpoint",
            "- `aggregate_metrics.csv`: copied from run root when present",
        ]
    )

    (bundle_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    package_meta = {
        "run_dir": str(run_dir),
        "bundle_dir": str(bundle_dir),
        "methods": [r.method for r in results],
        "dice_key": dice_key,
        "forgetting_key": forgetting_key,
        "caveats": caveats,
    }
    (bundle_dir / "bundle_meta.json").write_text(json.dumps(package_meta, indent=2), encoding="utf-8")

    return bundle_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Package ablation run outputs into a paper-friendly result bundle"
    )
    ap.add_argument(
        "run_dir",
        help="Ablation run directory (contains one subdirectory per method)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Optional bundle output dir (default: <run_dir>/result_bundle)",
    )
    args = ap.parse_args()

    bundle_dir = package_results(Path(args.run_dir), Path(args.output_dir) if args.output_dir else None)
    print(f"[ok] result bundle written: {bundle_dir}")


if __name__ == "__main__":
    main()
