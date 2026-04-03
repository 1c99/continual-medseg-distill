# Continual MedSeg Distillation — GRACE

Continual learning for 3D medical image segmentation. GRACE (Gated Residual
Adapter for Continual Efficiency) provides teacher-side reliability control
for foundation model distillation (MedSAM2/MedSAM3 -> MONAI UNet / nnU-Net).

## Quick Reference

- Setup: `bash scripts/bootstrap_env.sh && bash scripts/setup_external.sh --all`
- Smoke test: `python scripts/train.py --config configs/base.yaml --dry-run`
- Continual run: `python scripts/run_continual.py --base-config configs/base.yaml --task-config configs/tasks/totalseg_AB_sequence.yaml --method-config configs/methods/distill_replay_ewc.yaml --dataset-config configs/datasets/totalseg_clean.yaml`
- Data: TotalSegmentator at path configured in dataset YAML
- Splits: `data/splits/totalseg_train_clean_v1.json`
- Configs merge: base -> method -> dataset -> task -> override

## Technical Notes

- Use `unwrap_model()` from `src/engine/distributed.py` for DDP-wrapped model attribute access
- Gated adapters use `state_dict_full()` / `load_state_dict_full()` (not plain state_dict)
- Outputs: `outputs/` | External models: `third_party/` | Checkpoints: `checkpoints/` (all gitignored)

## Current Instructions

See `docs/instruction_20260403.md` for the active research roadmap and experiment plan.
