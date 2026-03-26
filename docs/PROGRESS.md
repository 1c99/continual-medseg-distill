# Project Progress Log

This file is updated alongside repository progress so experiment intent and implementation details stay traceable.

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

## Current Technical Status

### Working now
- Config loading, method dispatch, and model factory fully wired
- Synthetic end-to-end pipeline: train → eval → metrics.csv → aggregate_metrics.csv
- All 4 methods runnable via config switching
- Dice + HD95 metrics with robust edge-case handling
- Data adapters for TotalSeg, BraTS21, ACDC with input validation

### Remaining blockers
1. **No real-data run yet** — adapters validated at code level, but TotalSeg split-manifest flow not tested with actual disk data
2. **EWC Fisher estimation** — still uses L2 regularization placeholder, not diagonal Fisher
3. **Teacher checkpoint management** — distill methods create teacher via `copy.deepcopy` at task boundaries, no persistent checkpoint save/load
4. **Package_results.py** — not yet run on successful ablation output (blocked on having a run)

### Next top 3 actions
1. Run `package_results.py` on successful ablation output to validate M3
2. Run TotalSeg split-manifest mode on workstation with real data paths to validate M2
3. Implement Fisher-based EWC estimation for publication-grade distill+replay+ewc method

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

## Target Deliverables (Short-Term)
- [x] `v0.2`: KD baseline runnable on synthetic + config toggles
- [x] `v0.3`: Replay + KD combined training loop
- [x] `v0.4`: Dataset adapter stubs (TotalSeg/BraTS21/ACDC) with validation
- [x] `v0.5`: First ablation table auto-generated from logs
- [ ] `v0.6`: Real-data TotalSeg run with split-manifest
- [ ] `v0.7`: Paper-ready result bundle via package_results.py
- [ ] `v0.8`: Fisher-based EWC implementation

---

## How to Run (current)

### Synthetic ablation suite (fastest verification):
```bash
python scripts/run_ablations.py --base-config configs/base.yaml --synthetic
```

### Single method:
```bash
python scripts/train.py --config configs/base.yaml --dry-run
```

### Metrics edge-case tests:
```bash
python tests/test_metrics_edge_cases.py
```

If dependency errors occur, install project deps first:
```bash
pip install -e .
```

---

## Notes
- This log should be updated every meaningful repo push.
- Keep explanations concise but publication-oriented (method rationale + evidence impact).
