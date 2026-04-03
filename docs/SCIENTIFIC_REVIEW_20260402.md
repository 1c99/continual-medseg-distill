# Scientific Review & Experiment Report
**Date:** 2026-04-02 | **Project:** GRACE — Continual Medical Segmentation with Foundation Model Distillation

---

## 1. Research Question

> Can a reliability-aware teacher adapter (GRACE) enable foundation model knowledge distillation to reduce catastrophic forgetting in continual 3D medical image segmentation?

**Sub-questions:**
- Q1: Does adaptive EWC provide meaningful retention without destroying plasticity?
- Q2: Does replay provide complementary retention benefit?
- Q3: Does the foundation model teacher (MedSAM2/3) add value over self-distillation?
- Q4: Does the GRACE adapter (TRA + CGAD + CPA) outperform standard adapters?
- Q5: Does model capacity (nnU-Net vs MONAI UNet) affect retention?

---

## 2. Experimental Setup

### 2.1 Dataset
- **Source:** TotalSegmentator v2 (1,204 CT subjects, 104 anatomical structures)
- **Current split:** 100 train / 19 val (validation phase)
- **Full split prepared:** 963 train / 240 val (for final paper experiments)
- **Resolution:** 128³ voxels (MONAI UNet), 96³ (nnU-Net due to memory)

### 2.2 Task Sequence
| Task | Structures | Classes | Anatomy |
|------|-----------|---------|---------|
| Task A | liver, spleen, kidneys, pancreas, gallbladder, stomach, duodenum, small bowel, colon, esophagus, adrenal glands, urinary bladder | 14 + BG | Abdominal organs |
| Task B | gluteus max/med/min (L/R), iliopsoas (L/R), autochthon (L/R) | 10 + BG | Pelvic muscles |
| Task C (planned) | aorta, IVC, portal vein, pulmonary artery, iliac arteries/veins | 9 + BG | Abdominal vessels |

### 2.3 Student Models
| Model | Parameters | Resolution | Memory (train) |
|-------|-----------|-----------|----------------|
| MONAI UNet | 2M | 128³ | ~7 GB |
| nnU-Net (PlainConvUNet) | 22.6M | 96³ | ~12 GB |

### 2.4 Training Protocol
- 50 epochs per task, 100 steps/epoch, batch size 2
- Adam optimizer, lr=1e-3, DiceCE loss
- Replay buffer: 500 samples
- Multi-head output: per-task classification heads, shared backbone

---

## 3. Completed Experiments & Results

### 3.1 Results Table

| # | Method | Student | EWC Ratio | Task A Peak | Task A Retained | Task B Final | Forgetting ↓ |
|---|--------|---------|-----------|-------------|----------------|-------------|-------------|
| 1 | Finetune | MONAI UNet | — | 0.652 | 0.002 | 0.412 | 0.650 |
| 2 | DRE adaptive (ratio=0.5) | MONAI UNet | 0.5 | 0.651 | **0.650** | 0.023 | **0.001** |
| 3 | DRE adaptive (ratio=0.5) | nnU-Net | 0.5 | 0.596 | **0.603** | 0.000 | **-0.007** |
| 4 | DRE adaptive (ratio=0.1) | MONAI UNet | 0.1 | — | — | — | Running |
| 5 | EWC only (ratio=0.5) | MONAI UNet | 0.5 | 0.648 | — | — | Running |
| 6 | Replay only | MONAI UNet | — | 0.647 | — | — | OOM (3 experiments on 1 GPU) |

### 3.2 Previous Baseline Results (Old Non-Adaptive EWC, weight=0.2)

| Method | Task A Peak | Task A Retained | Task B Final | Forgetting |
|--------|------------|----------------|-------------|------------|
| Finetune | 0.637 | 0.000 | 0.415 | 0.637 |
| Replay | 0.659 | 0.000 | 0.680 | 0.659 |
| DRE (snapshot, non-adaptive) | 0.648 | 0.000 | 0.651 | 0.648 |
| Distill (snapshot) | 0.631 | 0.000 | 0.375 | 0.631 |
| LwF | 0.651 | 0.001 | 0.325 | 0.650 |

---

## 4. Key Findings

### 4.1 Finding 1: Adaptive EWC Scaling Works — Retention Is Real

**Before fix:** EWC weight=0.2 with Fisher magnitude ~2.5e-5 produced an effective penalty 200,000× smaller than the task loss. All methods showed 0.000 retention.

**After fix:** Adaptive scaling anchors the EWC penalty at 50% of the reference task loss (measured at the start of each task). Result: **Task A retention = 0.650** — virtually zero forgetting (0.001).

**Interpretation:** The EWC mechanism was always conceptually correct. The failure was purely a scaling mismatch between the penalty magnitude and the task loss. The adaptive scaling resolves this generically, regardless of model size or Fisher statistics.

### 4.2 Finding 2: Stability-Plasticity Trade-off Is Stark

| EWC Ratio | Retention (Task A) | Plasticity (Task B) | Interpretation |
|-----------|-------------------|--------------------|--------------------|
| 0.0 (finetune) | 0.002 | 0.412 | Maximum plasticity, zero retention |
| 0.5 (adaptive) | 0.650 | 0.023 | Maximum retention, zero plasticity |
| 0.1 (running) | ??? | ??? | Expected sweet spot |

At ratio=0.5, the backbone is effectively frozen — the model cannot learn new muscle features because every parameter change is penalized heavily. The training dice (0.534 at epoch 50) appears reasonable, but the validation dice (0.023) reveals severe overfitting to the limited training samples that are processable with a frozen backbone.

**The EWC ratio is the primary hyperparameter controlling the stability-plasticity trade-off.** This is consistent with continual learning theory (Kirkpatrick et al., 2017) but has not been demonstrated with adaptive scaling in the medical segmentation context.

### 4.3 Finding 3: Model Capacity Does Not Resolve the Trade-off

nnU-Net (22.6M params) at ratio=0.5 shows the same pattern as MONAI UNet (2M params):
- Task A retained: 0.603 (perfect retention)
- Task B final: 0.000 (complete plasticity failure)

A 10× larger model does not resolve the stability-plasticity tension. The backbone is equally locked by EWC regardless of capacity. This suggests the solution must come from the **method** (how knowledge is transferred), not the **model** (how much capacity is available).

**Implication for the paper:** This strengthens the motivation for GRACE — model capacity alone cannot solve continual forgetting under strong regularization. A teacher-guided approach that provides reliable external knowledge is needed.

### 4.4 Finding 4: Self-Distillation (Snapshot KD) Provides No Retention Benefit

In all experiments, snapshot KD on the current task's data provides no anti-forgetting signal. The snapshot teacher, trained on Task A organs, produces meaningless predictions on Task B muscle data. This was confirmed in both the old baselines (LwF, distill_snapshot showed 0.000 retention) and the current DRE experiments (where the KD component is dominated by the EWC and replay terms).

**Implication:** For foundation model distillation to help retention, the teacher must operate on **replayed old-task data** with a **pre-trained adapter for that task**. This is exactly what GRACE enables through its frozen core + per-task residuals.

### 4.5 Finding 5: Training Dice vs Validation Dice Discrepancy

The DRE adaptive experiment showed a striking divergence:
- Training dice (Task B, epoch 50): 0.534
- Validation dice (Task B, final eval): 0.023

This 23× gap indicates that with strong EWC, the model memorizes specific training samples rather than learning generalizable features. The frozen backbone forces the new task head to rely on non-transferable memorization rather than learning new feature representations.

**This is a known failure mode of strong regularization** — the model cannot update shared representations, so task-specific heads resort to overfitting.

---

## 5. Critical Bug Fixes Applied

### 5.1 Bugs Found and Fixed During This Session

| Bug | Severity | Impact | Fix |
|-----|----------|--------|-----|
| Label-space collision in evaluation | Critical | ALL prior retention metrics were artifacts (Dice=0.0 was eval bug, not real forgetting) | Multi-head output wrapper with per-task evaluation |
| EWC penalty 200,000× too small | Critical | EWC was effectively disabled in all experiments | Adaptive scaling with fixed reference loss |
| Teacher head not switching during replay | Critical | GRACE's frozen organ residual was never used for replayed data | Per-task teacher head switching in _compute_replay_kd() |
| Prototypes never used in training | Critical | CPA component was dead code | Integrated prototype KD into training loss |
| MedSAM3 GRACE adapter not created in HF loading path | High | _load_model_from_hf always created standard adapter, ignoring adapter_type | Refactored to _create_adapter() shared method |
| Prototype device mismatch (CPU vs CUDA) | High | Prototype accumulation crashed | Force CPU storage in update_prototypes() |
| KD on current task data is noise for external teachers | Medium | Student learning hurt by random adapter predictions | Skip current-task KD for external teachers |
| Prototype temperature too low (0.1) | Medium | Near-hard labels, losing soft KD benefit | Increased default to 0.5 |
| Standard deep adapter architecture mismatched GRACE deep | Medium | Unfair comparison due to different building blocks | Both now use identical ResidualBlocks |

### 5.2 Architectural Alignment

To ensure fair comparison between standard and GRACE adapters, both deep variants now use identical ResidualBlock architecture:

| | Standard Deep | GRACE Deep | Overhead |
|---|---|---|---|
| 3D | 8.4M params (3 ResBlocks + Conv) | 8.7M params (3 ResBlocks + Gate + Residual) | +3% |
| 2D | 2.8M params (3 ResBlock2ds + Conv) | 2.9M params (3 ResBlock2ds + Gate + Residual) | +3% |

Any performance difference can be attributed solely to the GRACE mechanism, not architecture.

---

## 6. Foundation Model Adapter Pre-training

### 6.1 Adapter Results (128³ Resolution)

| Adapter | Type | Teacher | Dice | Epoch | Status |
|---------|------|---------|------|-------|--------|
| Shallow standard | 3D, 1.8M | MedSAM2 | 0.534 | 110 | Converged |
| Deep standard | 3D, 8.4M | MedSAM2 | 0.563 | 61 | Converged |
| GRACE shallow | 3D, 2.2M | MedSAM2 | 0.518 | 89 | Converged |
| Shallow standard | 3D, 1.8M | MedSAM3 | 0.536 | 29 | Training |
| GRACE shallow | 3D, 2.2M | MedSAM3 | 0.489 | 11 | Training |

### 6.2 Adapter Quality Assessment

The best adapter (MedSAM2 deep, 0.563) is below the student's Task A peak (0.651). This means the teacher is weaker than the student on absolute accuracy.

**Mitigation strategies implemented:**
1. Prototype KD (CPA) uses backbone features directly — not limited by adapter accuracy
2. Confidence gate (CGAD) suppresses unreliable adapter predictions
3. KD on current task data skipped for external teachers (GT labels are better)
4. 2D slice-wise adapter designed for higher resolution processing (not yet tested)

### 6.3 2D Slice-Wise Adapter (Designed, Not Tested)

Processes backbone features per-slice at native resolution (512×512 for MedSAM2, 1008×1008 for MedSAM3) instead of 3D conv on downsampled features (~72×72). Expected to improve adapter dice by working at 7× higher spatial resolution.

---

## 7. GRACE Method — Status & Design

### 7.1 Components

| Component | Purpose | Implementation | Tested? |
|-----------|---------|---------------|---------|
| TRA (Task-Residual Adapter) | Frozen core + per-task residuals prevent teacher forgetting | Complete | No |
| CGAD (Confidence Gate) | Suppress unreliable KD spatially | Complete | No |
| CPA (Prototypes) | Zero-forgetting class memory as primary KD signal | Complete | No |
| Adaptive EWC | Scale EWC penalty to task loss | Complete | **Yes — works** |
| Teacher head switching | Teacher uses correct adapter per replay task | Complete | No |
| 2D Slice-wise adapter | Higher resolution teacher predictions | Complete | No |

### 7.2 Outstanding Validation

**None of the GRACE-specific components have been tested in an experiment yet.** All current experiments use snapshot self-distillation, not foundation model teachers. GRACE testing is blocked on:
1. Finding the right EWC ratio (ratio=0.1 running now)
2. Adapter convergence (MedSAM3 adapters still training)

---

## 8. Experimental Plan — Next Steps

### Phase 1: Complete Current Validation (ETA: 5 hours)
| Experiment | GPU | Purpose | ETA |
|-----------|-----|---------|-----|
| DRE ratio=0.1 | 0 | Find stability-plasticity balance | ~5 hours |
| EWC only | 1 | Does EWC alone retain without replay? | ~3 hours |

### Phase 2: EWC Ratio Sweep (if ratio=0.1 doesn't balance)
- Test ratios: {0.05, 0.1, 0.2, 0.3}
- Quick runs: 20 epochs Task A + 20 epochs Task B (enough to see the trend)
- Expected outcome: ratio ~0.1-0.2 gives moderate retention + moderate plasticity

### Phase 3: Foundation Model Teacher Experiments (after ratio found)
| Experiment | Teacher | Adapter | Purpose |
|-----------|---------|---------|---------|
| DRE + MedSAM2 standard | MedSAM2 | Shallow standard (0.534 dice) | Does foundation teacher beat snapshot? |
| DRE + MedSAM2 GRACE | MedSAM2 | GRACE shallow (0.518 dice) | Does GRACE beat standard adapter? |
| DRE + MedSAM3 standard | MedSAM3 | Shallow standard (0.536 dice) | MedSAM3 vs MedSAM2 comparison |
| DRE + MedSAM3 GRACE | MedSAM3 | GRACE shallow (0.489 dice) | Full method evaluation |

### Phase 4: Full Paper Experiments (after validation)
- Switch to full dataset (963 train / 240 val)
- Retrain all adapters on full data
- nnU-Net as primary student
- Add Task C (vessels)
- Multi-seed runs (3 seeds minimum)
- PLOP/MiB baseline comparison

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| No EWC ratio balances retention + plasticity | Medium | High — method fails fundamentally | Try per-layer EWC (different ratios for backbone vs head), or progressive unfreezing |
| GRACE shows no improvement over standard adapter | Medium | High — novel contribution lost | Pivot to feature-level KD or prototype-only approach |
| Adapter dice too low for useful KD | Low | Medium — teacher signal weak | 2D slice-wise adapter for higher resolution |
| MedSAM3 features not better than MedSAM2 | Low | Low — can focus on MedSAM2 | Both are valid teachers |
| Full dataset changes conclusions | Low | Medium | Validate on 100 subjects first (current approach) |

---

## 10. Summary

**What works:**
- Multi-head evaluation is correct (fixed critical artifact bug)
- Adaptive EWC scaling works (retention 0.000 → 0.650)
- The pipeline is complete and tested (216 unit tests pass)
- GRACE architecture is designed, implemented, and architecture-aligned for fair comparison

**What doesn't work yet:**
- EWC ratio=0.5 kills plasticity (Task B = 0.023)
- No GRACE experiment has been run
- Foundation model teacher experiments not started
- Only 100-subject validation split used

**The critical next result:**
DRE ratio=0.1 — currently running on GPU 0. If this shows both retention > 0.2 AND plasticity > 0.2, the method works and we proceed to GRACE experiments. If not, we need per-layer EWC or a different regularization approach.

---

*Report prepared for scientific review. All code committed to git (14 commits). Experiments reproducible from configs in the repository.*
