#!/usr/bin/env python3
"""Generate a clean split by removing corrupt subjects.

Usage:
    python scripts/make_clean_split.py \
        --split data/splits/totalseg_train_v1.json \
        --valid outputs/reports/valid_subjects.json \
        --output data/splits/totalseg_train_clean_v1.json

Preserves original split structure, only removes IDs not in valid_subjects.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_clean_split(split_path: Path, valid_path: Path, output_path: Path) -> None:
    with open(split_path) as f:
        manifest = json.load(f)

    with open(valid_path) as f:
        valid_ids = set(json.load(f))

    train_key = "train" if "train" in manifest else "train_ids"
    val_key = "val" if "val" in manifest else "val_ids"

    orig_train = manifest[train_key]
    orig_val = manifest[val_key]

    clean_train = [sid for sid in orig_train if sid in valid_ids]
    clean_val = [sid for sid in orig_val if sid in valid_ids]

    removed_train = set(orig_train) - set(clean_train)
    removed_val = set(orig_val) - set(clean_val)

    print(f"Train: {len(orig_train)} → {len(clean_train)}", end="")
    if removed_train:
        print(f"  (removed: {', '.join(sorted(removed_train))})")
    else:
        print("  (no changes)")

    print(f"Val:   {len(orig_val)} → {len(clean_val)}", end="")
    if removed_val:
        print(f"  (removed: {', '.join(sorted(removed_val))})")
    else:
        print("  (no changes)")

    clean_manifest = {
        "split_version": manifest.get("split_version", "v1") + "_clean",
        "dataset": manifest.get("dataset", "unknown"),
        "description": (
            f"Clean split from {split_path.name}: "
            f"removed {len(removed_train) + len(removed_val)} corrupt subjects"
        ),
        "source_split": split_path.name,
        train_key: clean_train,
        val_key: clean_val,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean_manifest, f, indent=2)
    print(f"\nWritten: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate clean split excluding corrupt subjects")
    parser.add_argument("--split", required=True, help="Original split manifest JSON")
    parser.add_argument("--valid", required=True, help="valid_subjects.json from scan_corrupt_nifti.py")
    parser.add_argument("--output", required=True, help="Output clean split JSON path")
    args = parser.parse_args()

    make_clean_split(Path(args.split), Path(args.valid), Path(args.output))


if __name__ == "__main__":
    main()
