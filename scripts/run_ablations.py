#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_METHODS = ["finetune", "replay", "distill", "distill_replay_ewc"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_last_metrics_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    last = dict(rows[-1])
    for k, v in list(last.items()):
        if v is None:
            continue
        try:
            if "." in v or "e" in v.lower():
                last[k] = float(v)
            else:
                last[k] = int(v)
        except Exception:
            pass
    return last


def run_method(
    repo_root: Path,
    base_config: Path,
    dataset_config: Path | None,
    method_name: str,
    run_dir: Path,
    dry_run: bool,
    synthetic: bool,
) -> tuple[int, dict[str, Any], list[str]]:
    method_config = repo_root / "configs" / "methods" / f"{method_name}.yaml"
    if not method_config.exists():
        raise FileNotFoundError(f"Unknown method config: {method_config}")

    base = load_yaml(base_config)
    method_cfg = load_yaml(method_config)
    cfg = merge_dicts(base, method_cfg)
    if dataset_config:
        cfg = merge_dicts(cfg, load_yaml(dataset_config))

    cfg.setdefault("experiment", {})
    cfg.setdefault("output", {})
    cfg["experiment"]["name"] = f"ablation_{method_name}"
    cfg["output"]["dir"] = str(run_dir)

    if synthetic:
        cfg.setdefault("data", {})
        cfg["data"]["source"] = "synthetic"

    resolved_cfg_path = run_dir / "resolved_config.yaml"
    write_yaml(resolved_cfg_path, cfg)

    cmd = [sys.executable, "scripts/train.py", "--config", str(resolved_cfg_path)]
    if dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    log_path = run_dir / "train.log"
    err_path = run_dir / "train.stderr.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    err_path.write_text(proc.stderr or "", encoding="utf-8")

    metrics = read_last_metrics_row(run_dir / "metrics.csv")
    return proc.returncode, metrics, cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Run method ablations and aggregate metrics")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--dataset-config", default=None)
    p.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Method names to run",
    )
    p.add_argument("--output-root", default="outputs/ablations")
    p.add_argument("--run-name", default=None, help="Optional custom run folder name")
    p.add_argument("--dry-run", action="store_true", help="Forward dry-run to train.py")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Force data.source=synthetic for smoke tests",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip method if metrics.csv already exists",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_config = (repo_root / args.base_config).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    dataset_config = None
    if args.dataset_config:
        dataset_config = (repo_root / args.dataset_config).resolve()
        if not dataset_config.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config}")

    output_root = (repo_root / args.output_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ablation_run_{timestamp}"
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []

    for method in args.methods:
        method_dir = run_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        existing_metrics = method_dir / "metrics.csv"
        if args.skip_existing and existing_metrics.exists():
            print(f"[skip] {method}: metrics.csv already exists")
            metrics = read_last_metrics_row(existing_metrics)
            row = {
                "method": method,
                "status": "skipped",
                "return_code": 0,
                **metrics,
            }
            aggregate_rows.append(row)
            continue

        print(f"[run] {method} -> {method_dir}")
        rc, metrics, cmd = run_method(
            repo_root=repo_root,
            base_config=base_config,
            dataset_config=dataset_config,
            method_name=method,
            run_dir=method_dir,
            dry_run=args.dry_run,
            synthetic=args.synthetic,
        )

        row = {
            "method": method,
            "status": "ok" if rc == 0 else "failed",
            "return_code": rc,
            "command": " ".join(cmd),
            **metrics,
        }
        aggregate_rows.append(row)
        print(f"[done] {method} status={row['status']} rc={rc}")

    all_keys: list[str] = []
    for r in aggregate_rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    aggregate_csv = run_root / "aggregate_metrics.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in aggregate_rows:
            w.writerow(r)

    summary = {
        "run_root": str(run_root),
        "base_config": str(base_config),
        "dataset_config": str(dataset_config) if dataset_config else None,
        "methods": args.methods,
        "dry_run": args.dry_run,
        "synthetic": args.synthetic,
        "aggregate_metrics_csv": str(aggregate_csv),
        "results": aggregate_rows,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [r for r in aggregate_rows if r.get("status") == "failed"]
    print(f"\nAggregate CSV: {aggregate_csv}")
    print(f"Summary JSON : {run_root / 'summary.json'}")
    if failed:
        print(f"Failed methods: {[r['method'] for r in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
