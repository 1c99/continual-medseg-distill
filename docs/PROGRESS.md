# Project Progress Log

This file is updated alongside repository progress so experiment intent and implementation details stay traceable.

---

## 2026-03-27 — Infrastructure Sprint (Implementation-Only, No Training Runs)

### Summary

Code-only sprint adding experiment infrastructure scripts: fairness guardrails,
statistical reporting, failure-case panel generation, compute-efficiency reporting,
and a frozen ablation spec with drift validation. **No training commands were executed.**

### What was completed

**Scripts:**
- `scripts/check_experiment_fairness.py` — validates parity (epochs, LR, batch size, split, seed) across compared conditions. Outputs markdown + JSON report with PASS/FAIL per key.
- `scripts/stats_report.py` — offline stats from metrics CSVs: mean/std, bootstrap CI, paired comparison templates. Explicit caveats when data is insufficient.
- `scripts/build_failure_panel.py` — indexes worst-k cases by metric, outputs panel manifest + markdown report template. Pipeline scaffolding for qualitative figures.
- `scripts/compute_report.py` — parses logs/manifests, reports params, throughput, epoch time, VRAM. Outputs CSV + markdown.
- `scripts/validate_freeze_spec.py` — validates run configs against the frozen ablation spec, detects drift.

**Configs:**
- `configs/experiments/paper_ablation_freeze.yaml` — locks the final experiment matrix: 5 conditions, 3 seeds, dataset/split, training protocol, metrics to report.

**Tests:**
- `tests/test_infra_scripts.py` — 31 unit tests covering parsing/validation logic for all new scripts.

### Implementation-only note

This sprint was explicitly constrained to infrastructure code only. No training
jobs were executed. All scripts work with existing metrics files or produce
explicit caveats when data is unavailable.

### What each script does (no-run mode)

| Script | Input | Output | Run mode |
|--------|-------|--------|----------|
| `check_experiment_fairness.py` | Method configs | Parity report (md + json) | Config-only |
| `stats_report.py` | Metrics CSVs | Stats summary (md + json + csv) | Reads existing files |
| `build_failure_panel.py` | Eval CSVs | Panel manifest (json + md) | Reads existing files |
| `compute_report.py` | Run dirs | Compute report (md + json + csv) | Reads existing files |
| `validate_freeze_spec.py` | Freeze spec + configs | Drift report (md + json) | Config-only |

---

## 2026-03-27 — Phase-Novelty: Orthogonal LoRA + Continual Distillation

### Summary

Added Orthogonal LoRA as a forgetting-mitigation mechanism for the student model.
Student-side LoRA adapters (low-rank Conv3d wrappers) are injected into the UNet,
with an orthogonality regularizer that penalizes subspace overlap between the
current task's adapter and adapters from previous tasks.

### What was completed

**Core Implementation:**
- `src/models/lora.py` — `LoRAConv3d` module + inject/extract/merge/load utilities
- `src/models/ortho_reg.py` — `orthogonality_loss()` computing ||A_curr^T @ A_prev||²_F + ||B_curr @ B_prev^T||²_F
- Student LoRA injection in `src/models/factory.py` (config-driven via `model.lora.*`)
- Optimizer param filtering in `src/engine/trainer.py` (only LoRA params when enabled)
- Ortho loss wired into `src/methods/distill_replay_ewc.py` training loop
- Per-task LoRA state persistence in save/load and `multi_task_trainer.py`

**Config:**
- `configs/methods/distill_replay_ewc_ortholora.yaml` — preset config (Condition 5)
- Config validation for `model.lora.*` keys in `src/utils/config_validation.py`

**Tests:**
- `tests/test_lora.py` — 10 unit tests (shape, freezing, roundtrip, merge, gradients)
- `tests/test_ortho_reg.py` — 6 unit tests (zero/nonzero/orthogonal/gradient/multi-prev)
- `tests/test_phase_next.py` — `test_smoke_ortho_lora_2task` integration smoke test

**Config keys** (`model.lora.*`):
- `enabled` (bool) — master toggle
- `mode` — `standard` | `orthogonal`
- `rank` (int) — LoRA rank (default: 8)
- `alpha` (float) — scaling factor (default: 16)
- `target_modules` (list[str]) — substring patterns for module targeting
- `ortho_lambda` (float) — orthogonality regularization weight

### Validated vs Pending

| Component | Status |
|-----------|--------|
| LoRA module + utilities | **Validated** (unit tests) |
| Orthogonality regularizer | **Validated** (unit tests) |
| Pipeline integration | **Validated** (smoke test) |
| Per-task adapter checkpoints | **Validated** (smoke test) |
| Config validation | **Validated** |
| Real-data Ortho LoRA vs standard comparison | **Pending** (GPU) |

---

## 2026-03-27 — Phase-Teacher-RealCkpt: Real Checkpoint Validation Sprint

### Summary

End-to-end integration of a real MedSAM3 teacher checkpoint (SAM3 base + LoRA weights from `lal-Joey/MedSAM3_v1`). Checkpoint acquisition, registry, contract validation script, and minimal KD benchmark configs are complete. GPU validation runs are pending.

### What was completed

**Checkpoint Acquisition (Validated):**
- MedSAM3 LoRA weights downloaded from HuggingFace (`lal-Joey/MedSAM3_v1`, file `best_lora_weights.pt`)
- Symlinked at `checkpoints/medsam3_lora.pt` (SHA-256: `499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91`)
- SAM3 base weights: auto-downloaded by `build_sam3_image_model(load_from_HF=True)`, cached by HuggingFace hub
- SAM3 standalone (`facebook/sam3`) is gated — using MedSAM3's bundled SAM3 copy instead

**Checkpoint Registry (Validated):**
- `configs/teachers/checkpoints.yaml` — records backend type, paths, source repo, and checksum
- `scripts/download_checkpoints.sh` — automated download script wrapping HF CLI

**Benchmark Configs (Validated — configs created, runs pending GPU):**
- `configs/runs/realckpt_finetune_taskA.yaml` — student finetune baseline (no teacher)
- `configs/runs/realckpt_kd_taskA.yaml` — student + real MedSAM3 teacher KD
- Both share: seed 42, 5 epochs, 20 steps/epoch, Task A (5 organs), shape `[64,64,64]`, loss dicece

### Validated vs Pending

| Component | Status | Artifact |
|-----------|--------|----------|
| MedSAM3 LoRA weights downloaded | **Validated** | `checkpoints/medsam3_lora.pt` (symlink to HF cache) |
| Checkpoint registry | **Validated** | `configs/teachers/checkpoints.yaml` |
| Download automation script | **Validated** | `scripts/download_checkpoints.sh` |
| Finetune baseline config | **Validated** | `configs/runs/realckpt_finetune_taskA.yaml` |
| KD with real teacher config | **Validated** | `configs/runs/realckpt_kd_taskA.yaml` |
| MedSAM3 forward-contract validation | **Blocked** (PyTorch 1.12 incompatible) | Script ready: `scripts/validate_teacher_contract.py` |
| Minimal KD benchmark (finetune run) | **Pending** (GPU not released) | Config ready: `configs/runs/realckpt_finetune_taskA.yaml` |
| Minimal KD benchmark (KD run) | **Blocked** (PyTorch 1.12 incompatible) | Config ready: `configs/runs/realckpt_kd_taskA.yaml` |
| Benchmark comparison summary | **Blocked** | Depends on above runs |
| Full 50-epoch A→B runs on real data | **Pending** | — |
| Continual A→B with real MedSAM3 teacher | **Pending** | — |
| Full ablation matrix (4 conditions x 3 seeds) | **Pending** | — |

### Blocker: PyTorch version incompatibility

**SAM3/MedSAM3 requires PyTorch 2.x** (uses `torch.nn.attention.sdpa_kernel` / `SDPBackend`).
This workstation has **PyTorch 1.12.1+cu113** — the SAM3 import chain fails at
`sam3/model/vl_combiner.py → from torch.nn.attention import sdpa_kernel, SDPBackend`.

This blocks:
- Contract validation with real MedSAM3 teacher
- KD runs with real MedSAM3 teacher
- Any code path that imports from `third_party/sam3/` or `third_party/medsam3/`

**Does NOT block:**
- Finetune baseline (no teacher import)
- UNet proxy KD runs (already validated in Phase-Teacher Integration)
- All existing tests (use mock/UNet backends only)

**Resolution options:**
1. Upgrade PyTorch to 2.x (check CUDA 11.3 compat — may need CUDA upgrade too)
2. Create a separate conda env with PyTorch 2.x for teacher validation
3. Defer real-teacher validation to a machine with PyTorch 2.x

### Commands to run when blocker is resolved

```bash
# 1. Contract validation
python scripts/validate_teacher_contract.py \
  --backend medsam3 \
  --lora-path checkpoints/medsam3_lora.pt \
  --output-channels 6 \
  --output-dir outputs/teacher_contract

# 2. Finetune baseline (can run now — no SAM3 dependency)
python scripts/train.py --config configs/runs/realckpt_finetune_taskA.yaml

# 3. KD with real teacher (requires PyTorch 2.x)
python scripts/train.py --config configs/runs/realckpt_kd_taskA.yaml
```

### Note on evidence quality

The 5-epoch / 20-step benchmark is a sanity check only — it verifies that the real teacher checkpoint loads, produces valid gradients, and the KD pipeline runs end-to-end. It is **not** publication-quality evidence. Full 50-epoch runs with statistical replication (seeds {42, 123, 456}) remain pending.

---

## 2026-03-27 — Phase-Teacher Integration: Backend Abstraction + External Teacher Support

### Summary

Introduced a pluggable teacher backend abstraction so the `Teacher` class can delegate to different model architectures (UNet, SAM3, MedSAM3) without changing the public API. This unblocks external teacher experiments.

### What was implemented

**Teacher Backend Abstraction (validated):**
- `src/methods/teacher_backends/base.py` — `TeacherBackend` ABC defining the interface: `load()`, `forward_logits()`, `forward_features()`, `metadata`, `to()`, `state_dict()`, `eval()`, `snapshot()`, `has_model`, `is_external`
- `src/methods/teacher_backends/unet.py` — `UNetBackend` extracted from original `Teacher` class. Supports snapshot and checkpoint modes. Full backward compatibility preserved.
- `src/methods/teacher_backends/__init__.py` — `create_backend(cfg)` factory dispatching on `teacher.type`
- `src/methods/teacher.py` — Refactored to delegate all operations to backend. Added `is_external` property. Public API unchanged.
- `src/utils/config_validation.py` — Extended valid teacher types: `{snapshot, checkpoint, sam3, medsam3}`. Added `output_channels` requirement for sam3/medsam3.

**External Teacher Types (pending implementation):**
- SAM3 backend (`type: sam3`) — registered in factory and config validation, implementation in progress
- MedSAM3 backend (`type: medsam3`) — registered in factory and config validation, pending SAM3 completion
- LoRA/PEFT wrapper — planned, off by default

**Snapshot Guard for External Teachers (pending):**
- `distill.py` and `distill_replay_ewc.py` `post_task_update()` currently call `self.teacher.snapshot(model)` unconditionally
- Guard pattern: `if not self.teacher.is_external: self.teacher.snapshot(model)` — to be added when external backends are wired

### Validation evidence

- All 10 new backend tests pass (`tests/test_teacher_backends.py`)
- All pre-existing teacher/KD tests pass (Teacher API backward compatible)
- Config validation accepts sam3/medsam3 types with required fields, rejects invalid types
- External backends correctly raise `NotImplementedError` on `snapshot()`

### Validated vs Pending

| Component | Status |
|-----------|--------|
| TeacherBackend ABC | Validated (tests pass) |
| UNetBackend (snapshot + checkpoint) | Validated (tests pass) |
| Teacher delegation to backend | Validated (backward compat confirmed) |
| Config validation for sam3/medsam3 | Validated (tests pass) |
| SAM3 backend implementation | Pending (in progress) |
| MedSAM3 backend implementation | Pending |
| `is_external` snapshot guard in distill | Pending |
| LoRA/PEFT integration | Pending |
| KD baseline with external teacher | Pending (blocked by checkpoint) |

---

## 2026-03-27 — Phase-Next Alignment: MedSAM3 Baseline + Task A/B Definition

### What was aligned

**PI direction adopted:**
- Task A = TotalSegmentator abdominal organs (liver, spleen, kidney_L, kidney_R, pancreas)
- Task B = TotalSegmentator pelvic muscles (gluteus_maximus_L/R, gluteus_medius_L/R, iliopsoas_L)
- Both tasks: 6 classes (BG + 5 structures), no overlap
- MedSAM3 = teacher baseline (checkpoint-based distillation)
- Student = MONAI 3D U-Net
- Order: Distillation first (A), then continual (A→B) with replay/EWC

**Baseline matrix (4 conditions):**
1. Student finetune A→B (no KD) — lower bound
2. Student + MedSAM3 KD on Task A — distillation baseline
3. Student A→B with replay only — no teacher
4. Student A→B with KD + replay + EWC — full method

### Implementation changes

| Change | Files |
|--------|-------|
| Multi-class TotalSeg adapter | `src/data/totalseg.py` (extended: `organs` list param), `src/data/registry.py` |
| Task A config | `configs/tasks/taskA_organs.yaml` |
| Task B config | `configs/tasks/taskB_muscles.yaml` |
| A→B sequence config | `configs/tasks/totalseg_AB_sequence.yaml` |
| MedSAM3 baseline config | `configs/methods/distill_medsam3_baseline.yaml` |
| Experiment protocol | `docs/experiment_protocol.md` (full rewrite) |
| Continual runner script | `scripts/run_continual.py` (NEW) |
| Synthetic A/B test config | `configs/tasks/synthetic_AB_test.yaml` |

### Validation evidence

- Synthetic A→B dry run: finetune — PASS (forgetting=-0.0077, all artifacts emitted)
- Synthetic A→B dry run: replay — PASS (forgetting=-0.0067, all artifacts emitted)
- Artifacts confirmed: `task_eval_matrix.csv`, `forgetting.json`, `multi_task_summary.json`
- All 126 tests pass

### Remaining blockers

1. **MedSAM3 checkpoint:** No pretrained checkpoint on this workstation. Needed for Conditions 2 and 4.
2. **Full real-data A→B run:** Not yet executed (requires multi-hour GPU time per condition).
3. **Multi-seed runs:** Statistical validation requires seeds {42, 123, 456}.

---

## 2026-03-26 — Phase-5.1: Data Corruption Unblock + Clean Baseline Rerun

### Summary
Identified and removed corrupt NIfTI files from TotalSegmentator split, ran all 4 baseline methods on clean data, validated AMP mixed precision path on GPU.

### Corruption Root Cause
- Subject **s0864** in `totalseg_train_v1.json` (val split) has a corrupt `ct.nii.gz` — gzip CRC check failure (`0x5ab25329 != 0x84f031a9`)
- This was the sole blocker for the Phase-5 baseline suite. All 100 train subjects are clean; only 1 of 20 val subjects is corrupt.
- Root cause: likely incomplete download or disk corruption of the TotalSegmentator dataset file.

### Clean Split
- Original: 100 train / 20 val → Clean: **100 train / 19 val**
- File: `data/splits/totalseg_train_clean_v1.json`

### Baseline Results (5 epochs, liver segmentation, 128^3 volumes)

| Method | Voxel Acc | Dice (liver) | HD95 | Time (s) | Status |
|--------|-----------|-------------|------|----------|--------|
| finetune | 0.9299 | 0.5926 | 34.54 | 322 | PASS |
| replay | 0.9121 | 0.5415 | 31.02 | 329 | PASS |
| distill | 0.9295 | 0.5883 | 34.97 | 324 | PASS |
| distill_replay_ewc | 0.9103 | 0.5363 | 31.26 | 366 | PASS |

Notes:
- All methods learning (loss decreasing, dice > 0.5 after 5 epochs)
- HD95 still high — expected at 5 epochs, would improve with full 50-epoch run
- finetune and distill are leading in dice; replay/ewc sacrifice dice for stability (lower HD95)

### AMP Validation
- AMP ON confirmed working on GPU with synthetic data (2 epochs, 10 steps)
- Peak GPU memory reduced by **29.9%** (138.4MB → 97.1MB) with AMP ON
- Convergence behavior identical (val_acc matches AMP OFF)

### New Scripts
- `scripts/scan_corrupt_nifti.py` — scans split manifests for corrupt NIfTI files (gzip CRC + nibabel header)
- `scripts/make_clean_split.py` — generates clean splits by removing corrupt subject IDs
- `scripts/amp_smoke.py` — AMP OFF vs ON comparison test with timing and memory stats

### Remaining Blockers
- Full 50-epoch baseline run not yet completed (5-epoch evidence run done)
- No real MedSAM3 checkpoint available for end-to-end teacher coupling test
- BraTS21 and ACDC datasets not yet validated/scanned for corruption

---

## 2026-03-26 — Code-Only Sprint: 7 Engineering Upgrades

### Summary
Production infrastructure sprint: DDP support, memory-efficient 3D pipeline, teacher caching, ablation compiler, figure generation, experiment registry, and CI/CD. 23 new tests (126 total). All CPU-runnable. CUDA 12.2 torch compatibility fixed (torch 2.4.1+cu121).

### Commit hashes by task

| Task | Hash | Description |
|------|------|-------------|
| 1 | `583b882` | DDP support (DistributedContext, rank-safe, grad accum) |
| 2 | `f95762c` | Patch sampler + OOM guard |
| 3 | `5cdef1a` | Teacher cache (disk-backed, sample+config hash keys) |
| 4 | `b828394` | Ablation matrix compiler (mean/std, NA handling) |
| 5 | `3847a02` | BWT/FWT bars + stability-plasticity scatter |
| 6 | `ebf7f09` | Experiment registry + status CLI |
| 7 | `b37b799` | GitHub Actions CI + pre-commit hooks |
| Tests | `0924037` | 23 new tests (126 total) |

### Files changed by task

| Task | Files |
|------|-------|
| 1 | `src/engine/distributed.py` (NEW) |
| 2 | `src/data/patch_sampler.py` (NEW), `src/utils/memory_guard.py` (NEW) |
| 3 | `src/methods/teacher_cache.py` (NEW) |
| 4 | `scripts/compile_ablation_matrix.py` (NEW) |
| 5 | `scripts/plot_results.py` (extended: BWT/FWT bars, stability-plasticity) |
| 6 | `experiments/registry.yaml` (NEW), `scripts/experiment_status.py` (NEW) |
| 7 | `.github/workflows/ci.yml` (NEW), `.pre-commit-config.yaml` (NEW) |
| Tests | `tests/test_sprint_code.py` (NEW, 23 tests) |

### Test summary (126/126 pass)

| Suite | Tests |
|-------|-------|
| test_sprint_code.py | 23 |
| test_phase_next.py | 34 |
| test_teacher_and_kd.py | 26 |
| test_multi_task.py | 13 |
| test_metrics_edge_cases.py | 12 |
| test_fisher_ewc.py | 8 |
| test_dicece_loss.py | 5 |
| test_reproducibility.py | 5 |

### Known limitations
1. DDP tested in disabled mode only (no multi-process launch in test env)
2. AMP/grad checkpointing config parsed but not wired into training loop (toggle-ready)
3. Teacher cache not auto-integrated into distill training_loss (opt-in API)
4. Plot scripts require matplotlib (graceful skip if missing)

### CUDA fix
torch 2.11+cu130 was incompatible with system CUDA 12.2 (driver 535). Fixed: `pip install torch==2.4.1+cu121`. 4x RTX 3090 now available.

---

## 2026-03-26 — Phase-Next: Architecture + Reproducibility Sprint

### Summary
Code-quality sprint while GPU compute is blocked. Standardized teacher interface, data pipeline hardening, early stopping, BWT/FWT metrics, config hashing, and 34 new tests. 103 total tests pass.

### What was implemented

**P0-1: Teacher Adapter**
- `forward_logits(x)` and `forward_features(x)` standardized interface
- `metadata` property: model_id, ckpt_hash (SHA256 of first 4KB), source_mode, frozen status
- Checkpoint hash for lineage tracking across runs

**P0-2: Data Pipeline Hardening**
- `src/data/label_remap.py`: configurable `LabelRemapper` with strict mode and domain verification
- `validate_subject()` classmethods on TotalSeg, BraTS21, ACDC adapters for fast schema checks
- `remap_from_config()` for config-driven label remapping

**P0-3: Experiment Reliability**
- `EarlyStopper` in trainer: configurable patience/metric/mode, logged when triggered
- `worker_init_fn()` for deterministic multi-worker DataLoader seeding
- `set_deterministic_mode()` for full reproducibility
- Resume lineage: `resume_count` tracked in `task_progress.json`
- `worker_init_fn` wired into all DataLoader construction via registry.py

**P1-4: Metrics + Analysis**
- BWT (Backward Transfer) and FWT (Forward Transfer) computed alongside forgetting
- `scripts/plot_results.py`: method comparison bars, forgetting heatmap, per-task dice curves
- Graceful fallback if matplotlib not installed

**P1-5: Config Quality**
- `compute_config_hash()`: deterministic SHA256 of config for reproducibility
- `save_resolved_config()`: writes resolved YAML + hash to run artifacts
- `validate_paths()`: checks data root and teacher checkpoint existence on disk

**P1-6: Test Expansion (34 new tests)**
- Teacher: forward_logits, forward_features, metadata, checkpoint hash (6)
- Data: LabelRemapper basic/torch/strict/domain, validate_subject per dataset (9)
- Reliability: EarlyStopper (3), worker seeding (1), deterministic mode (1), resume lineage (1)
- Metrics: BWT/FWT correctness, sign conventions (3)
- Config: hash determinism, path checks, resolved config round-trip (5)
- Integration: per-method 1-epoch tests (4), 4-method × 2-task smoke matrix (1)

### Test summary (103/103 pass)

| Test suite | Tests | Status |
|-----------|-------|--------|
| test_phase_next.py | 34 | PASS |
| test_teacher_and_kd.py | 26 | PASS |
| test_multi_task.py | 13 | PASS |
| test_metrics_edge_cases.py | 12 | PASS |
| test_fisher_ewc.py | 8 | PASS |
| test_dicece_loss.py | 5 | PASS |
| test_reproducibility.py | 5 | PASS |

### Files changed (12 files, 1156 insertions)

| File | Change |
|------|--------|
| `src/methods/teacher.py` | forward_logits/features, metadata, ckpt hash |
| `src/data/label_remap.py` | NEW — LabelRemapper utility |
| `src/data/totalseg.py` | validate_subject() |
| `src/data/brats21.py` | validate_subject() |
| `src/data/acdc.py` | validate_subject() |
| `src/data/registry.py` | worker_init_fn wiring |
| `src/engine/trainer.py` | EarlyStopper |
| `src/engine/multi_task_trainer.py` | BWT/FWT, resume_count |
| `src/utils/reproducibility.py` | worker_init_fn, set_deterministic_mode |
| `src/utils/config_validation.py` | config hash, path checks, save_resolved_config |
| `scripts/plot_results.py` | NEW — figure generation |
| `tests/test_phase_next.py` | NEW — 34 tests |

---

## 2026-03-26 — Phase-4: Teacher Abstraction, Multi-Mode KD, Resumable Engine, Config Validation

### Summary
Code-only sprint: built research-grade teacher integration, 4 KD modes (logit/feature/weighted/boundary), resumable multi-task engine, and strict config validation. All 69 tests pass. Synthetic ablation suite verified. No fake runtime claims — workstation compute blocked, code is execution-ready.

### What was implemented

**1. Teacher Integration Layer (`src/methods/teacher.py`)**
- Clean `Teacher` class with two modes: `snapshot` (deepcopy current model) and `checkpoint` (load from file)
- Automatic freeze/eval mode enforcement
- Hook-based intermediate feature extraction for feature KD
- State persistence (save_dict / load_state_dict)
- Config: `method.kd.teacher.type`, `method.kd.teacher.ckpt_path`, `method.kd.teacher.use_features`, `method.kd.teacher.feature_layers`

**2. Multi-Mode Distillation (`src/methods/distill.py`)**
- `logit`: KL-divergence on softened logits (baseline, unchanged behaviour)
- `feature`: MSE on intermediate representations + logit KD (requires `feature_layers`)
- `weighted`: uncertainty-weighted logit KD — down-weights uncertain teacher voxels using max-probability confidence
- `boundary`: boundary-aware logit KD — up-weights voxels near class boundaries via discrete Laplacian edge detection + Gaussian smoothing
- All modes fully config-toggleable via `method.kd.mode`
- No hardcoding by dataset

**3. Resumable Multi-Task Engine (`src/engine/multi_task_trainer.py`)**
- `task_progress.json` written after each task completion
- `resume=True` parameter: loads last checkpoint, restores model + method state, continues from next task
- Handles edge cases: all tasks already complete, missing checkpoint fallback
- Backward-compatible: `resume=False` (default) preserves existing behaviour

**4. Config Validation (`src/utils/config_validation.py`)**
- `validate_config(cfg, strict=True)` — fail-fast with actionable error messages
- Checks: model (channels), data (source, root, splits), method (name, KD mode, teacher settings, buffer sizes), train (epochs, lr, loss_type)
- Strict mode raises `ConfigError`; non-strict returns error list

**5. DistillReplayEWC updated (`src/methods/distill_replay_ewc.py`)**
- Refactored to use `Teacher` class instead of raw deepcopy
- Backward-compatible `teacher_model` property preserved

### Test summary (69/69 pass)

| Test suite | Tests | Status |
|-----------|-------|--------|
| test_teacher_and_kd.py | 26 | PASS |
| test_multi_task.py | 13 | PASS |
| test_metrics_edge_cases.py | 12 | PASS |
| test_fisher_ewc.py | 8 | PASS |
| test_dicece_loss.py | 5 | PASS |
| test_reproducibility.py | 5 | PASS |

**New test breakdown (26 tests):**
- Teacher: snapshot/freeze (2), independence (1), forward (1), checkpoint errors (2), round-trip (1), feature hooks (1)
- KD modes: logit (1), weighted (1), boundary (1), feature (1), mode-affects-loss (1)
- State round-trip: distill (1), distill_replay_ewc (1)
- Resume: interrupted run (1), already complete (1)
- Config validation: valid passes (1), 9 error detection tests

### Synthetic ablation re-verified (4/4 pass)

All methods still produce correct output after refactoring.

### Files changed

| File | Change |
|------|--------|
| `src/methods/teacher.py` | NEW — Teacher abstraction |
| `src/methods/distill.py` | REWRITE — multi-mode KD |
| `src/methods/distill_replay_ewc.py` | REWRITE — uses Teacher class |
| `src/engine/multi_task_trainer.py` | REWRITE — resume support |
| `src/utils/config_validation.py` | NEW — strict config validation |
| `configs/methods/distill.yaml` | Updated with KD mode + teacher config |
| `tests/test_teacher_and_kd.py` | NEW — 26 tests |
| `README.md` | Added KD modes, config validation, resume docs |
| `docs/experiment_protocol.md` | Marked code-complete / run-pending |

### Known limitations
1. Feature KD requires manual specification of `feature_layers` — no auto-detection
2. Boundary KD uses approximate Gaussian via avg_pool3d, not true Gaussian blur
3. Resume rebuilds val_loaders from scratch (adds ~seconds of overhead)
4. Config validation does not check file existence for non-absolute split manifest paths

### Next top 3 actions
1. Update CUDA driver → run `python scripts/train.py --config configs/base.yaml --dataset-config configs/datasets/totalseg_train.yaml`
2. Run multi-task ablation: TotalSeg → BraTS21 sequence, all 4 methods
3. Add multi-task ablation runner script wrapping `run_task_sequence()` across methods

---

## 2026-03-26 — M2 Achieved: Real-Data Validation on Workstation

### Summary
First real-data validation pass completed on workstation. TotalSegmentator and BraTS21 datasets validated end-to-end through the data pipeline (path checks + sample loading + shape/label verification). BraTS21 adapter extended to support flat nnUNet-style layout found on disk. ACDC dataset not present on this workstation.

### Data Discovery (workstation: /media/user/data2/data2/data)

| Dataset | Path | Subjects | Layout | Validation Status |
|---------|------|----------|--------|-------------------|
| TotalSegmentator | `Totalsegmentator_dataset/` | 1204 | `sXXXX/ct.nii.gz` + `sXXXX/segmentations/liver.nii.gz` | PASS |
| BraTS21 | `Brats21/` | 1151 (all 4 modalities) | flat: `imagesTr/`, `etc/images_flair/`, `etc/images_t2/`, `etc/labels_4cls/` | PASS (after adapter fix) |
| ACDC | Not present | 0 | N/A | BLOCKED — not on disk |

### TotalSegmentator Smoke Report
- 25/25 smoke-split subjects found on disk
- 5 subjects loaded through adapter successfully
- Volume shapes after crop/pad: (1, 128, 128, 128)
- Label values: {0, 1} (binary liver segmentation)
- No missing files or blockers

### BraTS21 Smoke Report
- 25/25 smoke-split subjects found on disk
- 5 subjects loaded through adapter successfully
- Volume shapes after crop/pad: (4, 128, 128, 128) — 4 MRI modalities
- Label values: {0, 1, 2, 3} (after standard 4→3 remap)
- No missing files or blockers

### What was implemented

**BraTS21 flat-layout adapter**
- `src/data/brats21.py`: Added `layout` parameter (`per_case` | `flat`). New `_resolve_paths()` method handles scattered file locations:
  - t1: `imagesTr/{sid}_t1_0000.nii.gz`
  - t1ce: `imagesTr/{sid}_t1ce_0000.nii.gz`
  - flair: `etc/images_flair/{sid}_0000.nii.gz`
  - t2: `etc/images_t2/{sid}_0000.nii.gz`
  - seg: `etc/labels_4cls/{sid}_seg.nii.gz`
- `src/data/registry.py`: Passes `layout` config to BraTS adapter
- `scripts/validate_data.py`: Updated path checker for flat layout

**Dataset configs and split manifests (workstation-specific)**
- `configs/datasets/totalseg_workstation.yaml` — smoke config (25 subjects)
- `configs/datasets/brats21_workstation.yaml` — smoke config (25 subjects)
- `configs/datasets/totalseg_train.yaml` — training config (100 train / 20 val)
- `configs/datasets/brats21_train.yaml` — training config (100 train / 20 val)
- `data/splits/totalseg_smoke.json` — 20 train / 5 val smoke split
- `data/splits/brats21_smoke.json` — 20 train / 5 val smoke split
- `data/splits/totalseg_train_v1.json` — 100 train / 20 val, seed=42
- `data/splits/brats21_train_v1.json` — 100 train / 20 val, seed=42

### Synthetic ablation re-verified (4/4 pass)

| Method | Dice Mean | HD95 Mean | Train Loss | Status |
|--------|-----------|-----------|------------|--------|
| finetune | 0.370 | 1.000 | 1.803 | ok |
| replay | 0.371 | 1.000 | 3.588 | ok |
| distill | 0.370 | 1.000 | 1.803 | ok |
| distill_replay_ewc | 0.371 | 1.000 | 3.588 | ok |

### Test summary (43/43 pass)

| Test suite | Tests | Status |
|-----------|-------|--------|
| test_dicece_loss.py | 5 | PASS |
| test_fisher_ewc.py | 8 | PASS |
| test_metrics_edge_cases.py | 12 | PASS |
| test_multi_task.py | 13 | PASS |
| test_reproducibility.py | 5 | PASS |

### Commands used
```bash
# Environment
conda create -y -p .venv python=3.10
pip install -e .
python scripts/doctor.py --dataset-root /media/user/data2/data2/data

# Real-data validation
python scripts/validate_data.py --dataset-config configs/datasets/totalseg_workstation.yaml --max-subjects 5 --output outputs/reports/totalseg_smoke_report.md
python scripts/validate_data.py --dataset-config configs/datasets/brats21_workstation.yaml --max-subjects 5 --output outputs/reports/brats21_smoke_report.md

# Synthetic ablation
python scripts/run_ablations.py --base-config configs/base.yaml --synthetic
```

### Blockers
1. **ACDC dataset not on disk** — cannot validate or run experiments for this task
2. **CUDA driver mismatch** — workstation CUDA driver (12020) too old for torch 2.11; training runs on CPU. GPU experiments need driver update.
3. **No real-data training run yet** — configs and splits ready, but no training executed (CPU-only would be too slow for 128^3 volumes)

### Next top 3 actions
1. Update CUDA driver on workstation to enable GPU training, then run first real-data TotalSeg baseline
2. Acquire ACDC dataset or adjust task sequence to CT-only (TotalSeg → TotalSeg-hard → BraTS21)
3. Run multi-task ablation with `run_task_sequence()` on TotalSeg → BraTS21 sequence

---

## 2026-03-26 — Baseline Suite Starter + Runbook

### Summary
Added a lightweight baseline-suite runner to execute the four core methods sequentially with consistent logging and a single CSV summary.

### What was implemented
- Added `scripts/run_baseline_suite.py`.
  - Runs `finetune`, `replay`, `distill`, `distill_replay_ewc` in order.
  - Supports `--dataset-config` to override dataset settings.
  - Supports `--dry-run` to forward smoke-mode execution to `scripts/train.py`.
  - Writes per-method run folders to `outputs/baselines/<timestamp>/<method>/`.
  - Captures per-method `stdout.log` and `stderr.log`.
  - Writes run-level `suite_summary.csv` with status, return code, duration, command, and log paths.
- Updated `docs/experiment_protocol.md` with one canonical baseline-suite command.

### Why this matters
- Provides a single, reproducible baseline entrypoint for quick experimentation and reporting.
- Makes failures/debugging easier with isolated method logs and explicit per-method status tracking.

## 2026-03-25 — Baseline Scaffold + First Runnable Wiring

### Summary
Set up an initial research scaffold for continual distillation in 3D CT/MRI segmentation and pushed first runnable wiring updates.

### What was implemented
- Repository scaffold with package structure:
  - `src/data`, `src/models`, `src/methods`, `src/engine`, `src/utils`
- Config-driven workflow:
  - base config, task config, method configs, dataset notes
- Scripts:
  - `scripts/train.py`, `scripts/eval.py`, `scripts/prepare_data.py`
- Synthetic-data mode for immediate dry-run path
- Initial methods exposed:
  - finetune, replay, distill, distill+replay+ewc (scaffold level)

### Commits pushed
- `446fa57` — Initial scaffold
- `a195ced` — Runnable train pipeline wiring + replay baseline scaffold

### Why this matters
- Gives us a reproducible starting point before real dataset integration.
- Keeps architecture modular so ablations are easy and publication-friendly.

---

## 2026-03-26 — M1 Achieved: Synthetic End-to-End Baseline Suite Passes

### Summary
Full 4-method ablation suite (finetune, replay, distill, distill+replay+ewc) runs end-to-end on synthetic 3D data and produces aggregate metrics. All supporting infrastructure (data adapters, metrics, config validation) hardened.

### What passed (M1)
All 4 methods complete training (2 epochs, 32^3 synthetic volumes, 3 classes) with status=ok:

| Method | Dice Mean | HD95 Mean | Train Loss |
|--------|-----------|-----------|------------|
| finetune | 0.3273 | 1.000 | 1.139 |
| replay | 0.3290 | 1.207 | 2.256 |
| distill | 0.3357 | 1.207 | 1.141 |
| distill_replay_ewc | 0.3131 | 1.414 | 2.267 |

Command used:
```bash
python scripts/run_ablations.py --base-config configs/base.yaml --synthetic
```

Output: `outputs/ablations/ablation_51b23c76b8c0/` with `aggregate_metrics.csv`, `summary.json`, per-method `metrics.csv` + logs.

### What was fixed (15 files, 332 insertions)

**Infra/Execution (Agent A):**
- `scripts/train.py`: added `sys.path` fix so subprocess invocations from `run_ablations.py` resolve `src.*` imports
- `src/utils/config.py`: renamed `_deep_update` → `merge_dicts` (public API used by `train.py`)

**Data Pipeline (Agent B):**
- `src/data/totalseg.py`: added root/subject directory existence checks, empty split guard, 3D shape validation, clear `FileNotFoundError` messages
- `src/data/brats21.py`: added 3D shape validation per modality, label value validation after remap (must be {0,1,2,3})
- `src/data/acdc.py`: added 3D shape/label validation, class value checks ({0=BG,1=RV,2=MYO,3=LV})

**Metrics/Evidence (Agent C):**
- `src/utils/metrics.py`: fixed `_surface()` to handle arbitrary ndim instead of hardcoded `(3,3,3)` structuring element
- `tests/test_metrics_edge_cases.py`: 12 new tests covering Dice and HD95 edge cases — all pass

**Method Core (Agent D):**
- `src/methods/base.py`: added `_validate_config()` virtual method
- `src/methods/distill.py`: added config validation with clear warnings for missing `kd.weight`, `kd.temperature`
- `src/methods/replay.py`: added config validation for missing `replay.buffer_size`
- `src/methods/distill_replay_ewc.py`: added full config validation (kd + replay + ewc); fixed double forward pass when no teacher exists
- `configs/methods/distill.yaml`: aligned keys (`kd.weight`, `kd.temperature`) with code expectations
- `configs/methods/replay.yaml`: aligned keys (`replay.buffer_size`, `replay.weight`)
- `configs/methods/distill_replay_ewc.yaml`: aligned keys for all three components

### Method readiness assessment

| Method | Status | Notes |
|--------|--------|-------|
| finetune | Production-grade | Simple CE baseline, fully functional |
| replay | Scaffold+ | In-memory buffer works, needs herding/prioritized sampling for paper |
| distill | Scaffold+ | Logit KD with temperature works, needs teacher checkpoint management |
| distill_replay_ewc | Scaffold | All 3 loss terms combine, but EWC uses L2 penalty (Fisher placeholder) |

### Metrics edge-case test results (12/12 pass)
- Dice: identical=1.0, no_overlap=0.0, empty_pred=0.0, empty_gt=0.0, both_empty=1.0, multiclass_mean correct
- HD95: identical=0.0, both_empty=0.0, one_empty=NaN, separated_blobs=positive, tiny_object=0.0, full integration works

---

## 2026-03-26 — Phase-2: Method Rigor + Real-Data Readiness

### Summary
Upgraded from scaffold to research-grade: Fisher-based EWC, DiceCE loss, reproducibility metadata, and data validation tooling. Synthetic ablation suite still passes with all changes.

### What was implemented

**1. Fisher-based EWC (replaces L2 placeholder)**
- `src/methods/distill_replay_ewc.py`: proper diagonal Fisher estimation via gradient squared averages
- `_estimate_fisher()`: samples N batches, computes per-param Fisher diagonals
- `_ewc_penalty()`: Fisher-weighted L2 penalty (was unweighted L2)
- `post_task_update()`: now accepts `train_loader` via `**kwargs` for Fisher estimation
- Config: `ewc.fisher_samples: 64` (configurable batch count for estimation)
- `src/engine/trainer.py`: passes `train_loader` to `method.post_task_update()`
- Tests: 8/8 pass (`tests/test_fisher_ewc.py`)

**2. DiceCE loss (replaces raw CE)**
- `src/methods/base.py`: added `_compute_loss()` dispatcher and `_dicece_loss()` implementation
  - Dice component: per-class (skip background), soft Dice with smoothing
  - CE component: standard cross-entropy
  - Combined: `dice_loss + ce_loss`
- Config switch: `train.loss_type: dicece` (default) or `ce` (fallback)
- All method subclasses updated to use `self._compute_loss()` instead of `F.cross_entropy()`
- Tests: 5/5 pass (`tests/test_dicece_loss.py`)

**3. Reproducibility hardening**
- New `src/utils/reproducibility.py`: `set_seed()`, `get_git_info()`, `collect_env_info()`
- `scripts/train.py`: sets random seeds (torch, numpy, random, CUDA) from `experiment.seed`
- `scripts/run_ablations.py`: embeds environment metadata in `run_manifest.json`:
  - `git_commit`, `git_dirty`, `python_version`, `torch_version`, `monai_version`, `platform`, `random_seed`
- Tests: 5/5 pass (`tests/test_reproducibility.py`)

**4. Data validation script (M2 prep)**
- New `scripts/validate_data.py`: validates real dataset paths before training
  - Phase 1: path existence checks (root, subjects, expected files)
  - Phase 2: sample loading with shape/label statistics
  - Supports totalseg, brats21, acdc sources
  - Produces markdown smoke report
- Tests: comprehensive path/report tests (`tests/test_validate_data.py`)

### Synthetic ablation results (Phase-2, with DiceCE + Fisher EWC)

| Method | Dice Mean | HD95 Mean | Train Loss |
|--------|-----------|-----------|------------|
| finetune | 0.370 | 1.000 | 1.803 |
| replay | 0.371 | 1.000 | 3.588 |
| distill | 0.370 | 1.000 | 1.803 |
| distill_replay_ewc | 0.371 | 1.000 | 3.588 |

Note: higher train_loss is expected with DiceCE (Dice+CE combined vs CE alone). Dice mean slightly improved from Phase-1 (~0.33 → ~0.37) on synthetic data.

### Test summary (all pass)

| Test suite | Tests | Status |
|-----------|-------|--------|
| test_metrics_edge_cases.py | 12 | PASS |
| test_dicece_loss.py | 5 | PASS |
| test_fisher_ewc.py | 8 | PASS |
| test_reproducibility.py | 5 | PASS |
| Synthetic ablation (4 methods) | 4 | PASS |

### Method readiness assessment (updated)

| Method | Status | Notes |
|--------|--------|-------|
| finetune | Production-grade | DiceCE loss, reproducible seeds |
| replay | Scaffold+ | DiceCE, in-memory FIFO buffer (needs reservoir sampling for paper) |
| distill | Scaffold+ | DiceCE + logit KD, needs teacher checkpoint persistence |
| distill_replay_ewc | **Research-grade** | DiceCE + KD + Fisher EWC, all three components production-quality |

### Run manifest now includes reproducibility metadata
```json
{
  "environment": {
    "git_commit": "2a41843",
    "git_dirty": true,
    "python_version": "3.11.11",
    "torch_version": "2.7.1",
    "monai_version": "1.5.2",
    "platform": "macOS-26.3.1-arm64-arm-64bit",
    "random_seed": 42
  }
}
```

---

## 2026-03-26 — Phase-3: Multi-Task Orchestrator + Teacher Persistence + Forgetting Evidence

### Summary
Implemented the continual learning execution backbone: multi-task sequential trainer with per-task evaluation, backward transfer (forgetting) measurement, and full method state persistence (teacher, Fisher, replay buffer). All 4 methods run through multi-task sequences on synthetic data. 43 tests pass across 5 suites.

### What was implemented

**1. Multi-task sequential trainer (`src/engine/multi_task_trainer.py`)**
- `run_task_sequence()`: trains a model on a sequence of tasks with per-task evaluation on all seen tasks
- Per-task config override merging — each task can specify its own data source/split
- Task checkpoint saving (model + method state) after each task
- `compute_forgetting()`: computes backward transfer matrix from eval history
  - Forgetting_i = perf(task_i, after_task_i) - perf(task_i, after_last_task)
  - Reports per-task forgetting + mean forgetting

**2. Evidence outputs (per-task tables + forgetting)**
- `task_eval_matrix.csv`: all (trained_on, evaluated_on) metric pairs
- `forgetting.json`: full forgetting matrix + per-task + mean forgetting
- `multi_task_summary.json`: task order, num tasks, forgetting stats

**3. Method state persistence (save_state / load_state)**
- `src/methods/base.py`: `save_state(path)`, `load_state(path)` interface
- `src/methods/replay.py`: saves/loads memory buffer
- `src/methods/distill.py`: saves/loads teacher model state_dict
- `src/methods/distill_replay_ewc.py`: saves/loads teacher + Fisher diagonals + prev_params + memory buffer
- Round-trip verified: save → fresh method → load → identical state

**4. Config: synthetic 2-task sequence (`configs/tasks/synthetic_2task.yaml`)**
- Defines a 2-task sequence for testing the orchestrator on synthetic data

### Test summary (43/43 pass)

| Test suite | Tests | Status |
|-----------|-------|--------|
| test_metrics_edge_cases.py | 12 | PASS |
| test_dicece_loss.py | 5 | PASS |
| test_fisher_ewc.py | 8 | PASS |
| test_reproducibility.py | 5 | PASS |
| test_multi_task.py | 13 | PASS |
| Synthetic ablation (4 methods) | 4 | PASS |

### test_multi_task.py breakdown (13/13 pass)

**Multi-task loop (6 tests):**
- finetune 2-task sequence: trains, evaluates both tasks, produces evidence files
- replay 2-task sequence: buffer has samples after training
- distill 2-task sequence: teacher exists after 2 tasks
- distill_replay_ewc 2-task: teacher + Fisher + prev_params + memory all populated
- checkpoint saved: per-task checkpoint files exist
- eval matrix CSV: correct rows (1 after task_0, 2 after task_1 = 3 total)

**Forgetting computation (4 tests):**
- No forgetting: identical perf → forgetting = 0
- Positive forgetting: decreased perf → forgetting > 0
- Negative forgetting: backward transfer → forgetting < 0
- Three tasks: correct matrix computation

**Teacher persistence (3 tests):**
- Distill save/load round-trip: teacher weights match exactly
- EWC save/load round-trip: Fisher + prev_params + teacher + memory all match
- Replay save/load round-trip: memory buffer matches

### Method readiness assessment (updated)

| Method | Status | Notes |
|--------|--------|-------|
| finetune | Production-grade | DiceCE loss, multi-task ready, reproducible seeds |
| replay | Research-grade | DiceCE, buffer save/load, multi-task ready |
| distill | Research-grade | DiceCE + KD, teacher persistence, multi-task ready |
| distill_replay_ewc | **Research-grade** | DiceCE + KD + Fisher EWC, full state persistence, multi-task ready |

---

## Current Technical Status

### Working now
- Full 4-method ablation suite with DiceCE loss (configurable)
- Multi-task sequential training with per-task evaluation and forgetting measurement
- Teacher/Fisher/memory checkpoint persistence (save/load between tasks and runs)
- Fisher-based EWC (diagonal Fisher estimation from training data)
- Reproducibility: deterministic seeds, git hash, env metadata in every run
- Data validation script ready for workstation deployment
- 43 tests across 5 suites, all passing

### Remaining blockers
1. **No real-data run yet** — `scripts/validate_data.py` written but workstation paths not accessible from dev machine. Run on workstation:
   ```bash
   python scripts/validate_data.py --dataset-config configs/datasets/totalseg_example.yaml
   ```
2. **BraTS21 path layout** — adapter assumes `root/{case_id}/{case_id}_*.nii.gz`; may need adjustment if data uses flat `imagesTr/labelsTr/` layout
3. **Multi-task ablation script** — `run_ablations.py` runs single-task per method; needs integration with `run_task_sequence()` for multi-task ablation runs

### Next top 3 actions
1. Run `scripts/validate_data.py` on workstation to produce M2 real-data smoke report
2. Add multi-task ablation runner (wraps `run_task_sequence()` across all 4 methods)
3. First real-data continual learning experiment with TotalSeg task sequence

---

## Research Rationale (Solid Paper Story)

### Why start with KD first
- Distillation is the most stable and interpretable first continual-learning building block.
- It establishes a clean baseline for all later contributions.
- Reviewer-friendly: easy to compare against fine-tune/replay-only baselines.

### Planned method progression
1. KD (logit) baseline
2. KD + feature distillation
3. KD + replay
4. KD + replay + uncertainty/boundary-aware weighting (novel contribution candidate)

---

## Split protocol layer
- Added `docs/DATA_SPLIT_PROTOCOL.md` with patient-level, leakage-safe split rules.
- Added example manifests:
  - `data/splits/example_totalseg_split.json`
  - `data/splits/example_brats21_split.json`
  - `data/splits/example_acdc_split.json`

## Target Deliverables
- [x] `v0.2`: KD baseline runnable on synthetic + config toggles
- [x] `v0.3`: Replay + KD combined training loop
- [x] `v0.4`: Dataset adapter stubs (TotalSeg/BraTS21/ACDC) with validation
- [x] `v0.5`: First ablation table auto-generated from logs
- [x] `v0.6`: Fisher-based EWC implementation
- [x] `v0.7`: DiceCE loss (medical segmentation standard)
- [x] `v0.8`: Reproducibility hardening (seed, git hash, env metadata)
- [x] `v0.9`: Data validation script for real-data smoke testing
- [x] `v1.0`: Multi-task sequence orchestrator with per-task evaluation
- [x] `v1.1`: Teacher/Fisher/memory checkpoint persistence (save/load)
- [x] `v1.2`: Forgetting measurement (backward transfer matrix + per-task + mean)
- [x] `v1.3`: Real-data validation (M2) — TotalSeg + BraTS21 validated on workstation
- [x] `v1.4`: Teacher abstraction with snapshot/checkpoint modes
- [x] `v1.5`: Multi-mode KD (logit/feature/weighted/boundary), all config-toggleable
- [x] `v1.6`: Resumable multi-task engine with task progress persistence
- [x] `v1.7`: Strict config validation with actionable errors
- [x] `v1.8`: Teacher adapter (forward_logits/features, metadata, ckpt hash)
- [x] `v1.9`: Data pipeline hardening (LabelRemapper, validate_subject)
- [x] `v2.0`: Early stopping, deterministic worker seeding, resume lineage
- [x] `v2.1`: BWT/FWT metrics, plot_results.py
- [x] `v2.2`: Config hash, resolved config persistence, path validation
- [x] `v2.3`: 103 tests total (34 new: integration + smoke matrix)
- [x] `v2.4`: DDP support, patch sampler, OOM guard
- [x] `v2.5`: Teacher cache, ablation matrix compiler
- [x] `v2.6`: Experiment registry, CI/CD, 126 total tests
- [ ] `v2.7`: Real-data TotalSeg training run (GPU ready, torch fixed)
- [ ] `v2.8`: Multi-task ablation runner (all 4 methods × task sequence)

---

## How to Run (current)

### Synthetic ablation suite (fastest verification):
```bash
python scripts/run_ablations.py --base-config configs/base.yaml --synthetic
```

### Data validation (on workstation with real data):
```bash
python scripts/validate_data.py --dataset-config configs/datasets/totalseg_example.yaml
```

### All tests:
```bash
python tests/test_metrics_edge_cases.py
python tests/test_dicece_loss.py
python tests/test_fisher_ewc.py
python tests/test_reproducibility.py
python tests/test_multi_task.py
```

If dependency errors occur, install project deps first:
```bash
pip install -e .
```

---

## Notes
- This log should be updated every meaningful repo push.
- Keep explanations concise but publication-oriented (method rationale + evidence impact).
