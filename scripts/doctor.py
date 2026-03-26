#!/usr/bin/env python3
"""Lightweight environment and config preflight checks."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


REQUIRED_MODULES = ["yaml", "torch", "numpy", "nibabel", "monai"]


def check_modules() -> list[str]:
    issues = []
    for m in REQUIRED_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:
            issues.append(f"missing module '{m}': {e}")
    return issues


def check_paths(dataset_root: str | None) -> list[str]:
    issues = []
    if dataset_root:
        p = Path(dataset_root)
        if not p.exists():
            issues.append(f"dataset root does not exist: {p}")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args()

    issues = []
    issues.extend(check_modules())
    issues.extend(check_paths(args.dataset_root))

    if issues:
        print("[doctor] FAIL")
        for i in issues:
            print(" -", i)
        return 1

    print("[doctor] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
