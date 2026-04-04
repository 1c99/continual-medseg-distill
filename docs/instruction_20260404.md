# Experiment Instructions — 2026-04-04

**Status:** 2-task POC done. 5-task infra ready. Baselines implemented but have critical bugs. MedSAM3 adapter training (ep 63/100, 0.565 dice).
**Blocking:** Fix baseline implementations + KL bug before running any more experiments.

---

## Phase 0: Bug Fixes (DO THESE FIRST)

### 0.1 Fix KL Divergence Reduction

All KL divergence calls use `reduction='mean'` but should use `reduction='batchmean'` (PyTorch docs: `'mean'` divides by total elements including class dim, `'batchmean'` divides by batch size only — the mathematically correct KL divergence).

Files to fix:
- `src/methods/distill.py` — `_logit_kd_loss()`, `_compute_kd_loss()` (weighted and boundary modes)
- `src/methods/distill_replay_ewc.py` — `_compute_replay_kd()`, `_compute_prototype_kd()`
- `src/methods/mib.py` — KD loss computation

Search for `F.kl_div(` and change all `reduction="mean"` to `reduction="batchmean"`.

### 0.2 Fix PLOP Implementation (`src/methods/plop.py`)

The current implementation is NOT faithful to Douillard et al., CVPR 2021. Two critical issues:

1. **Pseudo-labeling is dead code** — the pseudo-label generation method exists but is never called. PLOP uses pseudo-labels from the old model for background pixels in the new task. Wire this into `training_loss()`.

2. **POD spatial pooling destroys locality** — the current pooling implementation loses the multi-scale local structure that is PLOP's key contribution. POD computes width-pooled intermediate features at multiple scales and uses L2 distance to anchor them. The pooling should:
   - Hook intermediate encoder features (not just the output)
   - For each feature map, compute width-wise pooling along each spatial dimension separately
   - Pool at multiple scales (1×1, 2×2, 4×4 spatial regions)
   - L2 normalize before computing distillation loss

Reference: Algorithm 1 in Douillard et al. The pod_scales should define spatial subdivision, not global pooling.

### 0.3 Fix MiB Implementation (`src/methods/mib.py`)

The current implementation is NOT faithful to Cermelli et al., CVPR 2020. Issues:

1. **Unbiased CE is wrong** — MiB's key insight is that old-task classes get merged into the "background" class for the new task. The unbiased CE should:
   - Get old model's per-class probabilities for old classes
   - Aggregate old-class probabilities into the new model's background channel
   - Use this aggregated probability as the target for background pixels (not just reweighting)

2. **`unkd_weight` is declared but unused** — either remove it or wire it into the loss.

3. **`kd_weight=10.0` in config is pathologically high** — change default to `1.0` in `configs/methods/mib.yaml`.

Reference: Equations 3-5 in Cermelli et al.

### 0.4 Fix DER++ Implementation (`src/methods/der.py`)

Two bugs:

1. **Buffer is FIFO, not reservoir sampling** — DER++ paper uses reservoir sampling to maintain a uniform distribution over all seen examples. Current implementation just appends and truncates. Fix: implement reservoir sampling (for each new sample, with probability buffer_size/n_seen, replace a random buffer entry).

2. **Self-replay bias** — Current code inserts the current batch into the buffer BEFORE sampling replay. This means the model can immediately replay what it just saw, biasing toward recent data. Fix: sample replay BEFORE inserting current batch.

3. **Buffer size mismatch** — DER++ config uses `method.der.buffer_size: 200` but other methods use `method.replay.buffer_size: 1000`. Update `configs/methods/der.yaml` to use `buffer_size: 1000` for fair comparison.

### 0.5 Fix Ablation Script (`scripts/run_grace_ablation.sh`)

1. **GPU oversubscription** — `GPUS=(0 1 2 3 0)` double-books GPU 0 for 2 jobs. Fix: either run 4 at a time and queue the 5th, or use `GPUS=(0 1 2 3)` with the 5th variant running sequentially after one finishes.

2. **No output-dir collision protection** — if run twice, outputs overwrite silently. Add timestamp or run-id to output paths.

---

## Phase 1: The Critical Experiment

After Phase 0 fixes, this is the make-or-break experiment for the paper.

### 1.1 Wait for MedSAM3 Adapter

MedSAM3 gated adapter is training on GPU 3 (epoch 63/100, val_dice=0.565). Let it finish. This is the only teacher good enough to test GRACE properly.

### 1.2 GRACE vs DRE: Matched Comparison at 128³

Run these two experiments with IDENTICAL hyperparameters:

| Setting | Value |
|---------|-------|
| Resolution | 128³ |
| Epochs | 50 |
| Batch size | 2 |
| LR | 1e-3 |
| Replay buffer | 1000 |
| KD weight | 0.1 |
| EWC weight | 0.2 |
| Seed | 42 |
| Tasks | A → B |
| Student | MONAI UNet [16,32,64,128] |

**Experiment A: DRE (self-distillation baseline)**
```bash
python scripts/run_continual.py \
  --base-config configs/base.yaml \
  --task-config configs/tasks/totalseg_AB_sequence.yaml \
  --method-config configs/methods/distill_replay_ewc.yaml \
  --dataset-config configs/datasets/totalseg_clean.yaml \
  --override-config configs/overrides/full_50ep.yaml \
  --output-dir outputs/matched_dre_128_s42
```

**Experiment B: GRACE with MedSAM3**
```bash
python scripts/run_continual.py \
  --base-config configs/base.yaml \
  --task-config configs/tasks/totalseg_AB_sequence.yaml \
  --method-config configs/methods/distill_replay_ewc_medsam3_gated.yaml \
  --dataset-config configs/datasets/totalseg_clean.yaml \
  --override-config configs/overrides/full_50ep.yaml \
  --output-dir outputs/matched_grace_medsam3_128_s42
```

**IMPORTANT:** Before running Experiment B, verify that `distill_replay_ewc_medsam3_gated.yaml` uses:
- `kd.weight: 0.1` (NOT 0.7)
- `adapter_ckpt_path:` pointing to the finished MedSAM3 adapter checkpoint
- `adapter_type: gated_residual`

### 1.3 Interpret Results

**If GRACE (MedSAM3) beats DRE on Task A retention:**
→ Paper story works. Foundation teacher features > snapshot accuracy. Proceed to Phase 2.

**If GRACE (MedSAM3) loses to DRE:**
→ Pivot options:
  - (a) Try GRACE + self-distillation combined (external features as auxiliary signal)
  - (b) Focus on 5-task sequence where snapshot degrades over time but GRACE doesn't
  - (c) Reframe as analysis paper: "characterizing the teacher reliability gap"

**If GRACE ties with DRE:**
→ The 5-task experiment becomes critical — GRACE should have a growing advantage as snapshots age.

---

## Phase 2: Scale to 5-Task Benchmark (only after Phase 1 is positive)

### 2.1 Run ABCDE with Top Methods

Run on the 5-task sequence at 128³, 50 epochs:
- Finetune (lower bound)
- DRE (strongest baseline)
- GRACE with MedSAM3
- PLOP (after fixing)
- DER++ (after fixing)

This requires pretraining MedSAM3 adapters for Tasks A through D (Task E doesn't need one since there's no task after it for KD).

### 2.2 GRACE Ablation on ABCDE

Run the 5 ablation variants (standard adapter / TRA only / TRA+CGAD / TRA+CPA / full GRACE) on the 5-task sequence.

### 2.3 Multi-Seed

Run top 3 methods with seeds 42, 123, 456.

---

## Phase 3: Analysis and Visualization

Only after Phase 2 results are in:

1. Performance matrix heatmap R[i,j]
2. Forgetting trajectory across 5 tasks
3. Gate activation maps (CGAD visualization)
4. Teacher quality vs student benefit scatter
5. Class-wise retention heatmap
6. Prototype t-SNE

---

## Key Principle

**Do not run large-scale experiments until:**
1. All baseline implementations are faithful to their papers
2. KL divergence bug is fixed
3. The Phase 1 matched comparison validates GRACE's premise

Running 5-task benchmarks with broken baselines wastes GPU days and produces unpublishable results.

---

## Config Reminders

- Override `kd.weight` to 0.1 for ALL methods in fair comparisons
- Override `replay.buffer_size` to 1000 for ALL replay methods
- Use `configs/overrides/full_50ep.yaml` for 128³ / 50 epoch training
- MedSAM3 adapter checkpoint will be at: `checkpoints/medsam3_gated_adapter_taskA.pt` (or wherever the pretraining script saves it — check the output)
- `unwrap_model()` from `src/engine/distributed.py` for any new code touching model attributes through DDP

## GPU Allocation Plan

| GPU | Assignment |
|-----|-----------|
| 0 | Phase 1 Experiment A (DRE matched) |
| 1 | Phase 1 Experiment B (GRACE matched) — after MedSAM3 adapter done |
| 2 | Baseline fixes: smoke-test corrected PLOP, MiB, DER++ |
| 3 | MedSAM3 adapter pretraining (finishing) |
