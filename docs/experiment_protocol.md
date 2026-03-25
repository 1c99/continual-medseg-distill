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

## Ablation runner (reproducible orchestration)

Use `scripts/run_ablations.py` to run multiple method configs in one shot and produce a comparable table.

### What it does

- Iterates methods: `finetune`, `replay`, `distill`, `distill_replay_ewc`
- Creates isolated run folders under one timestamped parent
- Writes per-method resolved config snapshot (`resolved_config.yaml`)
- Captures stdout/stderr logs (`train.log`, `train.stderr.log`)
- Collects each run's final row from `metrics.csv`
- Exports merged `aggregate_metrics.csv` (+ `summary.json`)

### Default full method sweep

```bash
python scripts/run_ablations.py \
  --base-config configs/base.yaml
```

### Fast smoke test (dry-run + synthetic)

```bash
python scripts/run_ablations.py \
  --base-config configs/base.yaml \
  --dry-run \
  --synthetic
```

### Run a subset of methods

```bash
python scripts/run_ablations.py \
  --base-config configs/base.yaml \
  --methods finetune replay
```

### Reuse existing finished runs

```bash
python scripts/run_ablations.py \
  --base-config configs/base.yaml \
  --skip-existing
```

### Output layout

```text
outputs/ablations/ablation_run_YYYYMMDD_HHMMSS/
  aggregate_metrics.csv
  summary.json
  finetune/
    resolved_config.yaml
    metrics.csv
    checkpoints/
    train.log
    train.stderr.log
  replay/
  distill/
  distill_replay_ewc/
```

## TODOs

- [ ] Implement per-task dataloaders and transitions
- [ ] Add checkpoint manager and task-wise evaluator
- [ ] Add full continual metrics table generation
