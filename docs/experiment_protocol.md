# Experiment Protocol (Scaffold)

## Objective
Benchmark continual 3D medical segmentation across CT→MRI task sequence with:
1. Fine-tune baseline
2. Replay
3. Distillation
4. Distillation + Replay + EWC

## Suggested protocol

1. Define task order in `configs/tasks/ct_mri_sequence.yaml`.
2. Keep architecture fixed across methods.
3. For each method:
   - train sequentially task-by-task
   - evaluate on all seen tasks after each stage
4. Report:
   - average Dice over seen tasks
   - forgetting (drop from best historical performance)
   - forward/backward transfer (optional)

## Reproducibility checklist

- Set random seeds (`experiment.seed`)
- Log config snapshot per run
- Save model checkpoint at each task boundary
- Keep identical preprocessing across methods

## TODOs

- [ ] Implement per-task dataloaders and transitions
- [ ] Add checkpoint manager and task-wise evaluator
- [ ] Add full continual metrics table generation
