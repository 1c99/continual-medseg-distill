#!/usr/bin/env python3
"""Scan a split manifest for corrupt NIfTI files.

Usage:
    python scripts/scan_corrupt_nifti.py \
        --split data/splits/totalseg_train_v1.json \
        --root /media/user/data2/data2/data/Totalsegmentator_dataset \
        --dataset totalseg

Outputs (in outputs/reports/):
    corrupt_scan_report.md   — human-readable summary
    corrupt_subjects.json    — list of corrupt subject IDs + error details
    valid_subjects.json      — list of clean subject IDs
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _check_nifti_readable(path: Path) -> str | None:
    """Try to gzip-decompress and nibabel-load a NIfTI file.
    Returns None if OK, or an error string if corrupt."""
    if not path.exists():
        return f"MISSING: {path}"
    try:
        # First check: gzip CRC (fast, catches the known failure mode)
        with gzip.open(str(path), "rb") as f:
            while f.read(1 << 20):  # read in 1MB chunks
                pass
        # Second check: nibabel header parse
        import nibabel as nib
        img = nib.load(str(path))
        _ = img.shape  # force header read
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _get_files_for_subject(dataset: str, root: Path, sid: str) -> list[Path]:
    """Return list of files to check for a given subject."""
    if dataset == "totalseg":
        files = [root / sid / "ct.nii.gz"]
        seg_dir = root / sid / "segmentations"
        if seg_dir.exists():
            # Check liver as the default organ used in training
            liver = seg_dir / "liver.nii.gz"
            if liver.exists():
                files.append(liver)
        return files
    elif dataset == "brats21":
        # Check all 4 modalities + seg
        suffixes = ["_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz", "_flair.nii.gz", "_seg.nii.gz"]
        return [root / sid / f"{sid}{s}" for s in suffixes]
    else:
        return [root / sid / "ct.nii.gz"]


def scan(split_path: Path, root: Path, dataset: str, report_dir: Path) -> dict:
    """Scan all subjects in a split manifest. Returns summary dict."""
    with open(split_path) as f:
        manifest = json.load(f)

    train_ids = manifest.get("train", manifest.get("train_ids", []))
    val_ids = manifest.get("val", manifest.get("val_ids", []))
    all_ids = [(sid, "train") for sid in train_ids] + [(sid, "val") for sid in val_ids]

    corrupt = []
    valid = []
    t0 = time.time()

    for i, (sid, split) in enumerate(all_ids):
        files = _get_files_for_subject(dataset, root, sid)
        errors = []
        for fp in files:
            err = _check_nifti_readable(fp)
            if err:
                errors.append({"file": str(fp.relative_to(root)), "error": err})

        if errors:
            corrupt.append({"id": sid, "split": split, "errors": errors})
            status = "CORRUPT"
        else:
            valid.append(sid)
            status = "ok"

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(all_ids)}] {sid} ({split}): {status}  [{elapsed:.1f}s]")

    elapsed_total = time.time() - t0

    # Write outputs
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "corrupt_subjects.json", "w") as f:
        json.dump(corrupt, f, indent=2)

    with open(report_dir / "valid_subjects.json", "w") as f:
        json.dump(valid, f, indent=2)

    # Markdown report
    lines = [
        "# NIfTI Corruption Scan Report",
        "",
        f"- **Split:** `{split_path.name}`",
        f"- **Dataset:** {dataset}",
        f"- **Root:** `{root}`",
        f"- **Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"- **Scan time:** {elapsed_total:.1f}s",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total subjects | {len(all_ids)} |",
        f"| Valid | {len(valid)} |",
        f"| Corrupt | {len(corrupt)} |",
        f"| Train corrupt | {sum(1 for c in corrupt if c['split'] == 'train')} |",
        f"| Val corrupt | {sum(1 for c in corrupt if c['split'] == 'val')} |",
        "",
    ]

    if corrupt:
        lines.append("## Corrupt Subjects")
        lines.append("")
        lines.append("| Subject | Split | File | Error |")
        lines.append("|---------|-------|------|-------|")
        for c in corrupt:
            for err in c["errors"]:
                lines.append(f"| {c['id']} | {c['split']} | `{err['file']}` | {err['error']} |")
        lines.append("")

    lines.append("## Clean Split Counts (after removal)")
    lines.append("")
    corrupt_ids = {c["id"] for c in corrupt}
    clean_train = [s for s in train_ids if s not in corrupt_ids]
    clean_val = [s for s in val_ids if s not in corrupt_ids]
    lines.append(f"- Train: {len(train_ids)} → {len(clean_train)}")
    lines.append(f"- Val: {len(val_ids)} → {len(clean_val)}")
    lines.append("")

    with open(report_dir / "corrupt_scan_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nDone: {len(valid)} valid, {len(corrupt)} corrupt ({elapsed_total:.1f}s)")
    print(f"Reports: {report_dir}/")

    return {
        "total": len(all_ids),
        "valid": len(valid),
        "corrupt": len(corrupt),
        "corrupt_subjects": corrupt,
    }


def main():
    parser = argparse.ArgumentParser(description="Scan split manifest for corrupt NIfTI files")
    parser.add_argument("--split", required=True, help="Path to split manifest JSON")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--dataset", default="totalseg", choices=["totalseg", "brats21", "acdc"])
    parser.add_argument("--report-dir", default="outputs/reports", help="Output directory for reports")
    args = parser.parse_args()

    scan(
        split_path=Path(args.split),
        root=Path(args.root),
        dataset=args.dataset,
        report_dir=Path(args.report_dir),
    )


if __name__ == "__main__":
    main()
