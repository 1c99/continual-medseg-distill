#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Ensure repo root is on sys.path so `from src.*` imports work when invoked as a script.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils.reproducibility import collect_env_info


DEFAULT_METHODS = ["finetune", "replay", "distill", "distill_replay_ewc"]
DATA_SOURCES_WITH_SPLITS = {"totalseg", "brats21", "acdc"}


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return value[:length]


def resolve_path_like(raw: Any, base_dir: Path) -> Any:
    if not isinstance(raw, str) or raw.strip() == "":
        return raw
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def resolve_dataset_split_manifests(dataset_payload: dict[str, Any], dataset_config_path: Path) -> dict[str, Any]:
    """Resolve data.<source>.split_manifest relative to the dataset config file location."""
    out = merge_dicts({}, dataset_payload)
    data_cfg = out.get("data")
    if not isinstance(data_cfg, dict):
        return out

    base_dir = dataset_config_path.parent
    for source in DATA_SOURCES_WITH_SPLITS:
        source_cfg = data_cfg.get(source)
        if not isinstance(source_cfg, dict):
            continue
        if "split_manifest" in source_cfg:
            source_cfg["split_manifest"] = resolve_path_like(source_cfg["split_manifest"], base_dir)

    return out


def validate_run_config(cfg: dict[str, Any], cfg_name: str) -> list[str]:
    errors: list[str] = []

    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        return [f"{cfg_name}: missing 'data' mapping"]

    source = data_cfg.get("source")
    if not isinstance(source, str) or not source.strip():
        errors.append(f"{cfg_name}: data.source is required")
        return errors

    if source == "synthetic":
        synth = data_cfg.get("synthetic")
        if not isinstance(synth, dict):
            errors.append(f"{cfg_name}: data.synthetic mapping is required for source=synthetic")
        else:
            for field in ["train_samples", "val_samples", "channels", "num_classes", "shape"]:
                if field not in synth:
                    errors.append(f"{cfg_name}: data.synthetic.{field} is required for source=synthetic")
        return errors

    if source not in DATA_SOURCES_WITH_SPLITS:
        errors.append(
            f"{cfg_name}: unsupported data.source='{source}' for ablation runner preflight "
            "(supported: synthetic, totalseg, brats21, acdc)"
        )
        return errors

    source_cfg = data_cfg.get(source)
    if not isinstance(source_cfg, dict):
        errors.append(f"{cfg_name}: data.{source} mapping is required for source={source}")
        return errors

    root = source_cfg.get("root")
    if not isinstance(root, str) or not root.strip():
        errors.append(f"{cfg_name}: data.{source}.root is required for source={source}")

    split_manifest = source_cfg.get("split_manifest")
    train_ids = source_cfg.get("train_ids")
    val_ids = source_cfg.get("val_ids")

    has_train_ids = isinstance(train_ids, list) and len(train_ids) > 0
    has_val_ids = isinstance(val_ids, list) and len(val_ids) > 0

    if split_manifest:
        manifest_path = Path(str(split_manifest))
        if not manifest_path.exists():
            errors.append(
                f"{cfg_name}: data.{source}.split_manifest does not exist: {manifest_path}"
            )
    else:
        if not (has_train_ids and has_val_ids):
            errors.append(
                f"{cfg_name}: provide data.{source}.split_manifest or non-empty "
                f"data.{source}.train_ids and data.{source}.val_ids"
            )

    return errors


def run_method(
    repo_root: Path,
    cfg: dict[str, Any],
    method_name: str,
    run_dir: Path,
    dry_run: bool,
) -> tuple[int, dict[str, Any], list[str], str]:
    cfg.setdefault("experiment", {})
    cfg.setdefault("output", {})
    cfg["experiment"]["name"] = f"ablation_{method_name}"
    cfg["output"]["dir"] = str(run_dir)

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
    config_hash = sha256_json(cfg)
    return proc.returncode, metrics, cmd, config_hash


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
    dataset_payload: dict[str, Any] = {}
    if args.dataset_config:
        dataset_config = (repo_root / args.dataset_config).resolve()
        if not dataset_config.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config}")
        dataset_payload = resolve_dataset_split_manifests(load_yaml(dataset_config), dataset_config)

    output_root = (repo_root / args.output_root).resolve()

    base_payload = load_yaml(base_config)
    method_hashes: dict[str, str] = {}
    method_cfgs: dict[str, dict[str, Any]] = {}
    config_hashes: dict[str, str] = {}
    validation_errors: list[str] = []

    for method in args.methods:
        method_config = repo_root / "configs" / "methods" / f"{method}.yaml"
        if not method_config.exists():
            raise FileNotFoundError(f"Unknown method config: {method_config}")

        method_hashes[method] = file_hash(method_config)
        method_cfg = merge_dicts(base_payload, load_yaml(method_config))
        if dataset_payload:
            method_cfg = merge_dicts(method_cfg, dataset_payload)

        if args.synthetic:
            method_cfg.setdefault("data", {})
            method_cfg["data"]["source"] = "synthetic"

        method_cfgs[method] = method_cfg
        config_hashes[method] = sha256_json(method_cfg)
        validation_errors.extend(validate_run_config(method_cfg, f"method={method}"))

    if validation_errors:
        msg = "\n".join(["Config validation failed before run start:", *[f"- {e}" for e in validation_errors]])
        raise ValueError(msg)

    base_hash = file_hash(base_config)
    dataset_hash = file_hash(dataset_config) if dataset_config else "none"

    run_identity = {
        "base_hash": base_hash,
        "dataset_hash": dataset_hash,
        "method_hashes": method_hashes,
        "config_hashes": config_hashes,
        "methods": args.methods,
        "dry_run": args.dry_run,
        "synthetic": args.synthetic,
    }
    run_fingerprint = sha256_json(run_identity)

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"ablation_{short_hash(run_fingerprint)}"

    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "run_name": run_name,
        "run_root": str(run_root),
        "created_at": datetime.now().isoformat(),
        "base_config": str(base_config),
        "dataset_config": str(dataset_config) if dataset_config else None,
        "methods": args.methods,
        "dry_run": args.dry_run,
        "synthetic": args.synthetic,
        "base_hash": base_hash,
        "dataset_hash": dataset_hash,
        "method_hashes": method_hashes,
        "config_hashes": config_hashes,
        "run_fingerprint": run_fingerprint,
    }
    run_manifest["environment"] = collect_env_info(
        {"experiment": {"seed": base_payload.get("experiment", {}).get("seed")}}
    )
    (run_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

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
                "config_hash": config_hashes[method],
                **metrics,
            }
            aggregate_rows.append(row)
            continue

        print(f"[run] {method} -> {method_dir}")
        rc, metrics, cmd, resolved_hash = run_method(
            repo_root=repo_root,
            cfg=method_cfgs[method],
            method_name=method,
            run_dir=method_dir,
            dry_run=args.dry_run,
        )

        row = {
            "method": method,
            "status": "ok" if rc == 0 else "failed",
            "return_code": rc,
            "command": " ".join(cmd),
            "config_hash": resolved_hash,
            "method_hash": method_hashes[method],
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
        "run_manifest": str(run_root / "run_manifest.json"),
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
    print(f"\nRun manifest : {run_root / 'run_manifest.json'}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON : {run_root / 'summary.json'}")
    if failed:
        print(f"Failed methods: {[r['method'] for r in failed]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
