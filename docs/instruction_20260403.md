# Experiment Instructions — 2026-04-03

**Status:** POC complete (2-task A->B). Bug fixes merged. Ready to scale.
**Goal:** Prepare GRACE for MICCAI/TMI submission.

---

## Current Results

Best result: **nnU-Net DRE gentle** (N4) — Task A retained 0.079 Dice (28% of peak), Task B 0.441 Dice.
First experiment showing BOTH retention AND plasticity.

GRACE components (TRA, CGAD, CPA in `src/methods/teacher_backends/gated_adapter.py`) are
implemented but **untested in continual runs**.

---

## Priority 1: 5-Task TotalSegmentator Benchmark

2 tasks is a sanity check. Need 5+ for publication.

Create these task configs from TotalSegmentator v2 classes:

| Task | Anatomy | Classes | Config |
|------|---------|---------|--------|
| A | Abdominal organs (liver, spleen, kidneys, pancreas, etc.) | 14 | `taskA_organs.yaml` (exists) |
| B | Pelvic muscles (gluteus, iliopsoas, autochthon) | 10 | `taskB_muscles.yaml` (exists) |
| C | Abdominal vessels (aorta, IVC, portal vein, hepatic veins) | 7+ | `taskC_vessels.yaml` (create) |
| D | Thoracic organs (lung lobes, trachea, heart chambers) | 8+ | `taskD_thorax.yaml` (create) |
| E | Spine structures (vertebrae, spinal canal, ribs) | 10+ | `taskE_spine.yaml` (create) |

Create `configs/tasks/totalseg_ABCDE_sequence.yaml` with all 5 tasks.

## Priority 2: Validate GRACE End-to-End

1. Pretrain gated adapter on Task A: `python scripts/pretrain_teacher_adapter.py --config configs/runs/realckpt_kd_medsam2_taskA.yaml` (with `adapter_type: gated_residual`)
2. Run continual A->B with `configs/methods/distill_replay_ewc_medsam2_gated.yaml`
3. Verify: TRA residuals added at task transition, CGAD gate modulates KD, CPA prototypes stored
4. Compare vs standard (non-gated) adapter distillation

## Priority 3: Missing Baselines

Implement in `src/methods/`:
- **PLOP** — pooled local distillation (prominent CL segmentation baseline)
- **MiB** — Modeling the Background (background shift)
- **DER++** — Dark Experience Replay++ (logit replay)
- PackNet / HAT (parameter isolation) — lower priority

## Priority 4: Component Ablations

Run on the 5-task benchmark:
- Standard adapter only (no TRA/CGAD/CPA)
- TRA only (frozen core + residuals, no gate, no prototypes)
- TRA + CGAD (gate, no prototypes)
- TRA + CPA (prototypes, no gate)
- Full GRACE (TRA + CGAD + CPA)

Create override configs for each variant.

## Priority 5: Statistical Rigor

- All experiments with **3 seeds** (42, 123, 456)
- Report mean +/- std
- Paired t-tests or Wilcoxon for key comparisons

## Priority 6: Visualizations and Analysis

**Figures for the paper:**
1. Teacher logits + gate map + student prediction + GT overlay on replay samples
2. Forgetting trajectory: per-task Dice across full 5-task sequence
3. Performance matrix heatmap: R[i,j] = Dice on task j after training on task i
4. Teacher quality vs student benefit scatter: adapter Dice (x) vs retention gain (y)
5. Class-wise retention heatmap (small organs/vessels = where GRACE should shine)
6. Gate activation maps showing CGAD suppresses KD in unreliable regions
7. Prototype t-SNE across tasks

**Clinical metrics to add:** Surface Dice, organ volume error, per-class failure rate at thresholds.

## Priority 7: Research Innovations

Ordered by feasibility:

1. **Teacher reliability as measured phenomenon** (2/5) — calibration curves, error-vs-gate correlation, selective-KD risk curves
2. **Feature-level teacher-student consistency** (3/5) — MSE between MedSAM features and student encoder features
3. **Uncertainty-aware Gaussian prototypes** (3/5) — per-class Gaussian distributions, not just means
4. **Theoretical bound on selective KD** (3/5) — gate quality -> upper bound on KD risk
5. **Gate trained against student improvement** (4/5, highest novelty) — bilevel optimization, genuinely novel

---

## Paper Narrative

**Right framing:** "Foundation models are not static oracles in continual segmentation; the teacher interface itself becomes unreliable under label-space shift."

**Three claims to demonstrate:**
1. External foundation teachers can be WORSE than self-distillation when adapter reliability is poor
2. Reliability is spatially structured and predictable (CGAD)
3. Selective teacher-side continualization fixes this better than more student-side regularization

---

## Architecture Quick Reference

```
src/methods/distill_replay_ewc.py   — Main method (KD + replay + EWC + GRACE)
src/methods/teacher_backends/
  gated_adapter.py                   — GRACE: GatedResidualAdapter (TRA+CGAD+CPA)
  medsam2.py / medsam3.py            — Foundation model backends
src/engine/multi_task_trainer.py     — Continual task sequence runner
src/models/multi_head.py             — MultiHeadWrapper for task-incremental CL
configs/methods/distill_replay_ewc_medsam2_gated.yaml  — GRACE config
```

## Run Commands Cheat Sheet

```bash
# Single task training
python scripts/train.py --config configs/base.yaml --method-config configs/methods/distill_replay_ewc.yaml --dataset-config configs/datasets/totalseg_clean.yaml --task-config configs/tasks/taskA_organs.yaml

# Continual A -> B
python scripts/run_continual.py --base-config configs/base.yaml --task-config configs/tasks/totalseg_AB_sequence.yaml --method-config configs/methods/distill_replay_ewc.yaml --dataset-config configs/datasets/totalseg_clean.yaml --override-config configs/overrides/full_50ep.yaml --output-dir outputs/exp_name

# GRACE adapter pretraining
python scripts/pretrain_teacher_adapter.py --config configs/runs/realckpt_kd_medsam2_taskA.yaml

# Dry run (synthetic data, no GPU needed)
python scripts/train.py --config configs/base.yaml --dry-run

# Resume interrupted continual run
python scripts/run_continual.py ... --resume
```
