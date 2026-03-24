#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json


def parse_args():
    p = argparse.ArgumentParser(description="Prepare local dataset metadata (no downloads).")
    p.add_argument("--dataset", type=str, default="msd_task_placeholder")
    p.add_argument("--input-dir", type=str, required=True, help="Local extracted dataset path")
    p.add_argument("--output-json", type=str, default="data/splits/train_val.json")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {in_dir}")

    # TODO: implement dataset-specific parsing logic by dataset key.
    payload = {
        "dataset": args.dataset,
        "root": str(in_dir.resolve()),
        "train": [],
        "val": [],
        "note": "Scaffold only. Fill train/val entries with image/label paths.",
    }

    if args.dry_run:
        print("[DRY-RUN] Would write:")
        print(json.dumps(payload, indent=2))
        return

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
