# Experiment Protocol: Continual Distillation for 3D Medical Segmentation

## Overview

This document defines the experimental design for evaluating knowledge distillation
in a continual learning setting on 3D CT segmentation using TotalSegmentator.

**Core hypothesis:** A student model trained with knowledge distillation from a
pretrained teacher (MedSAM3) suffers less catastrophic forgetting when learning
new tasks compared to a student without distillation.

---

## Task Definitions

### Task A — Abdominal Organ Segmentation

| Class | Label | TotalSeg Mask File |
|-------|-------|--------------------|
| 0 | Background | — |
| 1 | Liver | `liver.nii.gz` |
| 2 | Spleen | `spleen.nii.gz` |
| 3 | Left Kidney | `kidney_left.nii.gz` |
| 4 | Right Kidney | `kidney_right.nii.gz` |
| 5 | Pancreas | `pancreas.nii.gz` |

Config: `configs/tasks/taskA_organs.yaml`

### Task B — Pelvic Muscle Segmentation

| Class | Label | TotalSeg Mask File |
|-------|-------|--------------------|
| 0 | Background | — |
| 1 | Left Gluteus Maximus | `gluteus_maximus_left.nii.gz` |
| 2 | Right Gluteus Maximus | `gluteus_maximus_right.nii.gz` |
| 3 | Left Gluteus Medius | `gluteus_medius_left.nii.gz` |
| 4 | Right Gluteus Medius | `gluteus_medius_right.nii.gz` |
| 5 | Left Iliopsoas | `iliopsoas_left.nii.gz` |

Config: `configs/tasks/taskB_muscles.yaml`

**No class overlap** between A and B. Both tasks use the same subject pool
(same CT volumes) with different segmentation targets.

---

## Baseline Matrix

Four experimental conditions, each evaluated with the A→B continual sequence:

| # | Condition | Method Config | Description |
|---|-----------|---------------|-------------|
| 1 | Student finetune (no KD) | `finetune.yaml` | Lower bound: naive sequential training |
| 2 | Student + MedSAM3 KD (Task A only) | `distill_medsam3_baseline.yaml` | Distillation on first task, then finetune on B |
| 3 | Student continual A→B (no KD) | `replay.yaml` | Replay buffer only, no teacher |
| 4 | Student continual A→B with KD+replay+EWC | `distill_replay_ewc.yaml` | Full method: distillation + replay + EWC |

### Model Architecture

- **Student:** MONAI 3D U-Net (`monai_unet`)
  - Channels: [16, 32, 64, 128], Strides: [2, 2, 2], 2 residual units
  - Input: 1 channel (CT), Output: 6 channels (BG + 5 classes)
- **Teacher (MedSAM3):** Loaded from checkpoint, same architecture
  - Config: `configs/methods/distill_medsam3_baseline.yaml`
  - Teacher is frozen during training

### Training Protocol

| Parameter | Value |
|-----------|-------|
| Epochs per task | 50 |
| Learning rate | 0.001 |
| Batch size | 2 |
| Loss | Dice + CE |
| Input shape | 128 × 128 × 128 |
| Seed | 42 |
| Split | `totalseg_train_clean_v1.json` (100 train / 19 val) |

---

## Evaluation Protocol

### Per-Task Evaluation

After training on each task, evaluate on **all** tasks seen so far:
- After Task A: eval on A
- After Task B: eval on A and B

### Metrics

| Metric | Description |
|--------|-------------|
| Dice (per-class) | Overlap coefficient per organ/muscle |
| Dice (mean) | Mean across all foreground classes |
| HD95 | 95th-percentile Hausdorff distance |
| Voxel Accuracy | Fraction of correctly classified voxels |

### Forgetting Metrics

| Metric | Formula |
|--------|---------|
| Forgetting | perf(A, after A) − perf(A, after B) |
| BWT | perf(A, after B) − perf(A, after A) |
| FWT | perf(B, before training B) |

---

## Expected Artifacts

Each experimental run produces:

| Artifact | Path |
|----------|------|
| Task eval matrix | `{output}/task_eval_matrix.csv` |
| Forgetting metrics | `{output}/forgetting.json` |
| Summary | `{output}/multi_task_summary.json` |
| Per-task checkpoints | `{output}/{task_id}/checkpoints/after_{task_id}.pt` |
| Training logs | `{output}/{task_id}/metrics.csv` |

---

## Evidence Quality Controls

1. **No overclaiming:** Synthetic runs are sanity checks only. Real-data results
   from `totalseg_train_clean_v1.json` are the primary evidence.
2. **Reproducibility:** All artifacts include config hash + commit hash + seed.
3. **Clean data:** Only validated subjects (corrupt s0864 excluded).
4. **Statistical validity:** Report mean ± std across 3 seeds for final results.

---

## Execution Order

1. Validate multi-class data loading (synthetic A→B dry run)
2. Run Condition 1: finetune A→B (baseline lower bound)
3. Run Condition 3: replay A→B (replay-only baseline)
4. Acquire/prepare MedSAM3 teacher checkpoint
5. Run Condition 2: MedSAM3 KD on A, then finetune B
6. Run Condition 4: KD + replay + EWC on A→B
7. Compile forgetting comparison table
8. Repeat with seeds {42, 123, 456} for statistical significance

---

## Runner Commands

### Synthetic sanity check (A→B dry run)

```bash
python scripts/run_continual.py \
  --base-config configs/base.yaml \
  --task-config configs/tasks/totalseg_AB_sequence.yaml \
  --method-config configs/methods/finetune.yaml \
  --dry-run
```

### Real-data continual run

```bash
python scripts/run_continual.py \
  --base-config configs/base.yaml \
  --task-config configs/tasks/totalseg_AB_sequence.yaml \
  --dataset-config configs/datasets/totalseg_train_clean.yaml \
  --method-config configs/methods/finetune.yaml
```

### Baseline suite runner

```bash
python scripts/run_baseline_suite.py \
  --base-config configs/base.yaml \
  --dataset-config configs/datasets/totalseg_train_clean.yaml
```

---

## Status

- [x] Task A/B definitions with explicit class groups
- [x] Multi-class TotalSeg adapter
- [x] Experiment protocol documented
- [x] MedSAM3 baseline config
- [x] A→B continual pipeline (multi_task_trainer.py)
- [x] Forgetting/BWT/FWT computation
- [x] 126+ tests passing
- [ ] MedSAM3 checkpoint acquisition
- [ ] Full 50-epoch A→B runs on real data
- [ ] Multi-seed statistical analysis
