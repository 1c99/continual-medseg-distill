#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


METHODS = ["finetune", "replay", "distill", "distill_replay_ewc"]


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


def run_method(
    repo_root: Path,
    method: str,
    base_cfg: dict[str, Any],
    method_cfg_path: Path,
    dataset_cfg: dict[str, Any] | None,
    method_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    run_cfg = merge_dicts(base_cfg, load_yaml(method_cfg_path))
    if dataset_cfg:
        run_cfg = merge_dicts(run_cfg, dataset_cfg)

    run_cfg.setdefault("experiment", {})
    run_cfg.setdefault("output", {})
    run_cfg["experiment"]["name"] = f"baseline_{method}"
    run_cfg["output"]["dir"] = str(method_dir)

    resolved_config_path = method_dir / "resolved_config.yaml"
    write_yaml(resolved_config_path, run_cfg)

    cmd = [sys.executable, "scripts/train.py", "--config", str(resolved_config_path)]
    if dry_run:
        cmd.append("--dry-run")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    duration_sec = round(time.time() - t0, 2)

    (method_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (method_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    return {
        "method": method,
        "status": "ok" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "duration_sec": duration_sec,
        "command": " ".join(cmd),
        "run_dir": str(method_dir),
        "stdout_log": str(method_dir / "stdout.log"),
        "stderr_log": str(method_dir / "stderr.log"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run baseline methods sequentially and collect one suite summary")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--dataset-config", default=None, help="Optional dataset override YAML")
    p.add_argument("--output-root", default="outputs/baselines")
    p.add_argument("--timestamp", default=None, help="Optional run id (default: YYYYMMDD_HHMMSS)")
    p.add_argument("--dry-run", action="store_true", help="Forward dry-run to train.py")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    base_config_path = (repo_root / args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    dataset_cfg: dict[str, Any] | None = None
    dataset_config_path: Path | None = None
    if args.dataset_config:
        dataset_config_path = (repo_root / args.dataset_config).resolve()
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        dataset_cfg = load_yaml(dataset_config_path)

    run_id = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_root = (repo_root / args.output_root / run_id).resolve()
    suite_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(base_config_path)
    summary_rows: list[dict[str, Any]] = []

    print(f"[suite] run_id={run_id}")
    print(f"[suite] output={suite_root}")

    for method in METHODS:
        method_cfg_path = (repo_root / "configs" / "methods" / f"{method}.yaml").resolve()
        if not method_cfg_path.exists():
            raise FileNotFoundError(f"Method config not found: {method_cfg_path}")

        method_dir = suite_root / method
        method_dir.mkdir(parents=True, exist_ok=True)

        print(f"[run] {method}")
        row = run_method(
            repo_root=repo_root,
            method=method,
            base_cfg=base_cfg,
            method_cfg_path=method_cfg_path,
            dataset_cfg=dataset_cfg,
            method_dir=method_dir,
            dry_run=args.dry_run,
        )
        summary_rows.append(row)
        print(f"[done] {method} status={row['status']} rc={row['return_code']}")

    summary_csv = suite_root / "suite_summary.csv"
    fieldnames = [
        "method",
        "status",
        "return_code",
        "duration_sec",
        "command",
        "run_dir",
        "stdout_log",
        "stderr_log",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\n[suite] summary={summary_csv}")

    failed = [row["method"] for row in summary_rows if row["status"] != "ok"]
    if failed:
        print(f"[suite] failed_methods={failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
