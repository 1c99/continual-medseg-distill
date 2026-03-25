# Data Split Protocol (Frozen, Patient-Level)

This protocol defines split rules for continual segmentation experiments to prevent leakage and ensure reproducibility.

## Core Rules
1. Patient-level split only (no scan/frame leakage across train/val/test).
2. Each task has fixed split manifests committed to repo.
3. Cross-task overlap must be explicitly checked and logged.
4. Split versioning via `split_version` field in each manifest.

## Manifest Format (JSON/YAML)
```json
{
  "split_version": "v1",
  "dataset": "totalseg",
  "train": ["s0001", "s0002"],
  "val": ["s0100"],
  "test": ["s0200"]
}
```

## Required Validation Checks
- non-empty train/val
- unique IDs per split
- no overlap among train/val/test
- IDs exist on disk

## Task Sequence Recommendation
- Task A: TotalSegmentator (CT)
- Task B: TotalSegmentator hard subset (CT)
- Task C: BraTS21 (MRI)
- Task D: ACDC (MRI)

## Reproducibility Checklist
- commit hash recorded per experiment
- split manifest path logged in run outputs
- random seed logged for each run
