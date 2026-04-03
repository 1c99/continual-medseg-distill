# CLAUDE.md — Workstation Instructions for Continual MedSeg Distillation

## Project Overview

This is a **continual learning for 3D medical image segmentation** research project.
The core contribution is **GRACE** (Gated Residual Adapter for Continual Efficiency):
a teacher-side mechanism for reliable knowledge distillation from foundation models
(MedSAM2/MedSAM3) to a lightweight student (MONAI UNet / nnU-Net) across a growing
sequence of segmentation tasks.

**Current state:** POC with 2-task sequence (organs -> muscles). Bug fixes complete.
Ready to scale to full experiments.

## Quick Setup

```bash
# 1. Environment
bash scripts/bootstrap_env.sh          # creates .venv, installs deps
source .venv/bin/activate

# 2. External teacher models (needed for KD experiments)
bash scripts/setup_external.sh --all   # clones SAM3 + MedSAM3 into third_party/

# 3. Verify setup
python scripts/doctor.py --dataset-root /path/to/Totalsegmentator_dataset

# 4. Smoke test (synthetic data, no download needed)
python scripts/train.py --config configs/base.yaml --dry-run
```

## Data

- **TotalSegmentator** dataset required at: `/media/user/data2/data2/data/Totalsegmentator_dataset`
  (or override via dataset config YAML)
- Split manifests: `data/splits/totalseg_train_clean_v1.json` (100 train / 19 val)
- Full split: `data/splits/totalseg_full_v1.json` (~1200 subjects)

## Running Experiments

### Single-task training
```bash
python scripts/train.py \
  --config configs/base.yaml \
  --method-config configs/methods/distill_replay_ewc.yaml \
  --dataset-config configs/datasets/totalseg_clean.yaml \
  --task-config configs/tasks/taskA_organs.yaml
```

### Continual sequence (A -> B)
```bash
python scripts/run_continual.py \
  --base-config configs/base.yaml \
  --task-config configs/tasks/totalseg_AB_sequence.yaml \
  --method-config configs/methods/distill_replay_ewc.yaml \
  --dataset-config configs/datasets/totalseg_clean.yaml \
  --override-config configs/overrides/full_50ep.yaml \
  --output-dir outputs/experiment_name
```

### Teacher adapter pretraining (GRACE)
```bash
python scripts/pretrain_teacher_adapter.py \
  --config configs/runs/realckpt_kd_medsam2_taskA.yaml
```

### Key flags
- `--dry-run`: 1 epoch, 3 steps (for testing)
- `--resume`: Resume from last completed task checkpoint
- `--print-config`: Dump resolved config and exit

## Config System

Modular YAML configs merged in order: base -> method -> dataset -> task -> override

```
configs/
  base.yaml                    # Root defaults (model arch, train params)
  methods/                     # Training methods
    finetune.yaml              # Naive sequential (lower bound)
    replay.yaml                # Replay buffer only
    lwf.yaml                   # Learning without Forgetting
    distill.yaml               # KD only
    distill_replay_ewc.yaml    # Full method (KD + replay + EWC)
    distill_medsam2.yaml       # MedSAM2 teacher KD
    distill_replay_ewc_medsam2_gated.yaml  # GRACE with MedSAM2
    distill_replay_ewc_medsam3_gated.yaml  # GRACE with MedSAM3
  tasks/                       # Task definitions and sequences
    totalseg_AB_sequence.yaml  # A(organs) -> B(muscles) sequence
    taskA_organs.yaml          # 14 abdominal organs (15 ch)
    taskB_muscles.yaml         # 10 muscles (11 ch)
  datasets/                    # Data source configs
    totalseg_clean.yaml        # TotalSeg clean split
  overrides/                   # Training overrides
    full_50ep.yaml             # 50 epochs, full training
    reasonable_20ep.yaml       # 20 epochs
    smoke.yaml                 # Quick smoke test
  runs/                        # Complete experiment configs
```

## Architecture

```
src/
  models/
    factory.py              # build_model() dispatcher
    multi_head.py           # MultiHeadWrapper for task-incremental CL
  methods/
    __init__.py             # create_method() dispatcher
    base.py                 # ContinualMethod base class
    distill.py              # KD method (logit/feature/weighted/boundary)
    distill_replay_ewc.py   # Full method: KD + replay + EWC + GRACE
    replay.py               # Replay buffer
    teacher.py              # Teacher abstraction (delegates to backends)
    teacher_backends/
      base.py               # TeacherBackend interface
      medsam2.py            # MedSAM2 backend (SAM2 + adapter)
      medsam3.py            # MedSAM3 backend (SAM3 + LoRA + adapter)
      sam3.py               # SAM3 backend
      gated_adapter.py      # GRACE: GatedResidualAdapter (TRA+CGAD+CPA)
      slice_adapter.py      # 2D slice-wise adapter variant
  engine/
    trainer.py              # Single-task train loop
    multi_task_trainer.py   # Continual task sequence runner
    evaluator.py            # Validation (Dice, HD95, per-class)
    distributed.py          # DDP utilities (unwrap_model, barriers)
  data/
    registry.py             # Dataset/loader factory
  utils/
    losses.py, metrics.py, config.py, reproducibility.py
```

## Current Results (as of 2026-04-03)

Best result so far: **nnU-Net DRE gentle** (Experiment N4)
- Task A retained: 0.079 Dice (28% of peak)
- Task B final: 0.441 Dice
- First experiment showing BOTH retention AND plasticity

All GRACE-specific components (TRA, CGAD, CPA) remain **untested in continual runs**.
The gated_adapter.py is implemented but hasn't been validated end-to-end.

## Research Roadmap — What Needs to Be Done

### PRIORITY 1: Scale to 5-Task Benchmark (Critical for Publication)

The current 2-task (A->B) sequence is a POC. For MICCAI/TMI publication, need 5+ tasks.

**Create configs for 5-task TotalSegmentator sequence:**
1. Task A: Abdominal organs (liver, spleen, kidneys, pancreas, etc.) — 14 classes
2. Task B: Pelvic muscles (gluteus, iliopsoas, autochthon) — 10 classes
3. Task C: Abdominal vessels (aorta, IVC, portal vein, hepatic veins) — 7+ classes
4. Task D: Thoracic organs (lung lobes, trachea, heart chambers) — 8+ classes
5. Task E: Spine structures (vertebrae, spinal canal, ribs) — 10+ classes

All classes available in TotalSegmentator v2. Create:
- `configs/tasks/taskC_vessels.yaml`
- `configs/tasks/taskD_thorax.yaml`
- `configs/tasks/taskE_spine.yaml`
- `configs/tasks/totalseg_ABCDE_sequence.yaml`

### PRIORITY 2: Validate GRACE End-to-End

The GRACE components are implemented but never tested in actual continual training:
1. Run adapter pretraining for Task A with gated adapter (`scripts/pretrain_teacher_adapter.py`)
2. Run continual A->B with `distill_replay_ewc_medsam2_gated.yaml`
3. Verify: TRA residuals are added at task transition, CGAD gate modulates KD, CPA prototypes stored
4. Compare against standard (non-gated) adapter distillation

### PRIORITY 3: Missing Baselines

Reviewers will require comparison against:
- **PLOP** (pooled local distillation) — prominent CL segmentation baseline
- **MiB** (Modeling the Background) — background shift handling
- **DER++** (Dark Experience Replay++) — logit replay baseline
- **PackNet / HAT** — parameter isolation baselines

Implement at minimum PLOP and MiB in `src/methods/`.

### PRIORITY 4: Component Ablations

Need to prove each GRACE component contributes. Run these variants:
- Standard adapter only (no TRA/CGAD/CPA)
- TRA only (frozen core + residuals, no gate, no prototypes)
- TRA + CGAD (gate, no prototypes)
- TRA + CPA (prototypes, no gate)
- Full GRACE (TRA + CGAD + CPA)

Create override configs for each ablation.

### PRIORITY 5: Statistical Rigor

- Run all experiments with **3 seeds minimum** (42, 123, 456)
- Compute mean +/- std for all metrics
- Paired t-tests or Wilcoxon for key comparisons

### PRIORITY 6: Key Analyses and Visualizations

**Figures needed for the paper:**
1. **Teacher reliability visualization:** Teacher logits + gate map + student prediction + GT overlay on replay samples
2. **Forgetting trajectory plot:** Per-task Dice across the full 5-task sequence (not just final)
3. **Performance matrix heatmap:** R[i,j] = Dice on task j after training on task i
4. **Teacher quality vs student benefit scatter:** Adapter Dice (x) vs retention gain (y)
5. **Class-wise retention heatmap:** Small organs/vessels are where GRACE should shine
6. **Gate activation maps:** Show CGAD suppresses KD in unreliable regions
7. **Prototype t-SNE:** Visualize CPA prototype separation across tasks

**Clinical metrics to add:**
- Surface Dice (boundary accuracy)
- Organ volume error (clinical relevance)
- Per-class failure rate at clinically relevant thresholds

### PRIORITY 7: Research Innovations (for spotlight/oral)

High-impact improvements ordered by feasibility:

1. **Make teacher reliability measurable** (difficulty: 2/5)
   - Add reliability calibration curves
   - Error-vs-gate correlation analysis
   - Selective-KD risk curves on replay data

2. **Feature-level teacher-student consistency** (difficulty: 3/5)
   - Add MSE loss between MedSAM backbone features and student encoder features
   - Better than logit-only KD when adapter is weaker than student

3. **Uncertainty-aware Gaussian prototypes** (difficulty: 3/5)
   - Replace simple class means with per-class Gaussian distributions in feature space
   - Better support for small structures and multimodal anatomy

4. **Theoretical bound on selective KD** (difficulty: 3/5)
   - Even a simple proposition relating gate quality to upper bound on KD-induced risk
   - Differentiates from purely empirical work

5. **Gate trained against student improvement** (difficulty: 4/5, highest novelty)
   - Current gate predicts adapter correctness (weak proxy)
   - Gate that predicts student-improving regions would be genuinely novel (bilevel optimization)

## Paper Narrative

**Wrong framing:** "We combined replay, EWC, KD, adapters, and prototypes."

**Right framing:** "Foundation models are not static oracles in continual segmentation;
the teacher interface itself becomes unreliable under label-space shift."

**Three claims to demonstrate:**
1. External foundation teachers can be WORSE than self-distillation when adapter reliability is poor
2. Reliability is spatially structured and predictable (CGAD)
3. Selective teacher-side continualization fixes this better than more student-side regularization

## Important Notes

- All outputs go to `outputs/` (gitignored)
- External models go to `third_party/` (gitignored)
- Checkpoints in `checkpoints/` (gitignored)
- The `unwrap_model()` helper in `src/engine/distributed.py` must be used when accessing
  model attributes (register_head, current_task) through DDP wrapper
- Gated adapter save/load uses `state_dict_full()` / `load_state_dict_full()` for full
  state persistence (prototypes, task channels, core frozen flag)
