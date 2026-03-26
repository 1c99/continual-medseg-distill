#!/usr/bin/env python3
"""Experiment registry viewer and status updater.

Usage:
    python scripts/experiment_status.py                    # print status table
    python scripts/experiment_status.py --pending          # show only pending
    python scripts/experiment_status.py --update <id> done # update status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = _REPO_ROOT / "experiments" / "registry.yaml"
VALID_STATUSES = {"pending", "running", "done", "failed"}


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        print(f"[error] Registry not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_registry(data: dict, path: Path = REGISTRY_PATH) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def validate_config_refs(experiments: list[dict]) -> list[str]:
    """Check that config_ref files exist on disk."""
    warnings = []
    for exp in experiments:
        ref = exp.get("config_ref", "")
        if ref:
            full = _REPO_ROOT / ref
            if not full.exists():
                warnings.append(f"{exp['id']}: config_ref not found: {ref}")
        for mc in exp.get("method_configs", []):
            full = _REPO_ROOT / mc
            if not full.exists():
                warnings.append(f"{exp['id']}: method_config not found: {mc}")
    return warnings


def print_table(experiments: list[dict], filter_status: str | None = None) -> None:
    filtered = experiments
    if filter_status:
        filtered = [e for e in experiments if e.get("status") == filter_status]

    if not filtered:
        print(f"No experiments with status='{filter_status}'" if filter_status else "No experiments found")
        return

    # Simple table
    print(f"{'ID':<35} {'STATUS':<10} {'DESCRIPTION'}")
    print("-" * 80)
    for exp in filtered:
        eid = exp.get("id", "?")[:34]
        status = exp.get("status", "?")
        desc = exp.get("description", "")[:40]
        print(f"{eid:<35} {status:<10} {desc}")

    counts = {}
    for e in experiments:
        s = e.get("status", "unknown")
        counts[s] = counts.get(s, 0) + 1
    print(f"\nTotal: {len(experiments)} | " + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def update_status(experiments: list[dict], exp_id: str, new_status: str) -> bool:
    if new_status not in VALID_STATUSES:
        print(f"[error] Invalid status '{new_status}'. Valid: {VALID_STATUSES}", file=sys.stderr)
        return False
    for exp in experiments:
        if exp.get("id") == exp_id:
            old = exp.get("status")
            exp["status"] = new_status
            print(f"[ok] {exp_id}: {old} -> {new_status}")
            return True
    print(f"[error] Experiment '{exp_id}' not found", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description="Experiment registry status")
    parser.add_argument("--pending", action="store_true", help="Show only pending")
    parser.add_argument("--running", action="store_true", help="Show only running")
    parser.add_argument("--failed", action="store_true", help="Show only failed")
    parser.add_argument("--validate", action="store_true", help="Check config refs exist")
    parser.add_argument("--update", nargs=2, metavar=("ID", "STATUS"), help="Update experiment status")
    args = parser.parse_args()

    registry = load_registry()
    experiments = registry.get("experiments", [])

    if args.validate:
        warnings = validate_config_refs(experiments)
        if warnings:
            for w in warnings:
                print(f"[warn] {w}")
        else:
            print("[ok] All config refs valid")
        return

    if args.update:
        exp_id, new_status = args.update
        if update_status(experiments, exp_id, new_status):
            save_registry(registry)
        return

    filter_status = None
    if args.pending:
        filter_status = "pending"
    elif args.running:
        filter_status = "running"
    elif args.failed:
        filter_status = "failed"

    print_table(experiments, filter_status)


if __name__ == "__main__":
    main()
