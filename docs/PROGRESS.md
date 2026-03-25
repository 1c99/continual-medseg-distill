# Project Progress Log

This file is updated alongside repository progress so experiment intent and implementation details stay traceable.

---

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

## Current Technical Status

### Working now
- Config loading and method dispatch are wired.
- Model/data factory compatibility fixes are in place.
- Replay method now has a basic in-memory buffer scaffold.
- Pipeline can be executed once environment dependencies are installed.

### Pending (next implementation)
1. Install/test environment (`torch`, `monai`) on workstation and run dry-run command.
2. Implement robust KD baseline (logit KD first).
3. Add feature distillation term.
4. Integrate replay + KD jointly in trainer (clean toggles via config).
5. Add checkpointing, metric logger, and reproducible seed control across scripts.

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

## Target Deliverables (Short-Term)
- [ ] `v0.2`: KD baseline runnable on synthetic + config toggles
- [ ] `v0.3`: Replay + KD combined training loop
- [ ] `v0.4`: Open-source dataset adapter stubs (BTCV/MSD/KiTS/AMOS/CHAOS/BraTS)
- [ ] `v0.5`: First ablation table auto-generated from logs

---

## How to Run (current)
```bash
python scripts/train.py --config configs/base.yaml --dry-run
```

If dependency errors occur, install project deps first:
```bash
pip install -e .
```

---

## Notes
- This log should be updated every meaningful repo push.
- Keep explanations concise but publication-oriented (method rationale + evidence impact).
