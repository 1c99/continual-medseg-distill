#!/bin/bash
set -euo pipefail
# Train all remaining adapters to convergence
# Each runs 100 epochs with 100 steps/epoch
# Monitors for convergence and launches next when GPU frees up

OUTPUT_ROOT=/media/user/data2/data2/data/medseg_outputs/multihead_128_50ep_s42
mkdir -p "$OUTPUT_ROOT"
BASE="configs/base.yaml"
TASK="configs/tasks/taskA_organs.yaml"
DATA="configs/datasets/totalseg_clean_128.yaml"

COMMON_ARGS="--base-config $BASE --task-config $TASK --dataset-config $DATA --epochs 100 --max-steps-per-epoch 100"

echo "=== Adapter Training Pipeline ==="
echo "Started at $(date)"

# --- Wait for current trainings to finish ---
echo "Waiting for current adapter trainings to finish..."
while ps aux | grep pretrain_teacher_adapter | grep -v grep | grep -v "run_all" > /dev/null 2>&1; do
    sleep 120
done
echo "All current trainings done at $(date)"

# --- Phase 1: MedSAM2 GRACE deep (GPU 0, ~8GB) ---
echo ""
echo "=== Phase 1: MedSAM2 GRACE deep (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 conda run -n medseg python scripts/pretrain_teacher_adapter.py \
    $COMMON_ARGS \
    --teacher-type medsam2 \
    --teacher-ckpt third_party/medsam2/checkpoints/MedSAM2_latest.pt \
    --output checkpoints/medsam2_deep_gated_adapter_taskA_128.pt \
    --adapter-type gated_residual \
    --deep-adapter \
    --task-id taskA_organs \
    --lr 1e-3 \
    > $OUTPUT_ROOT/adapter_medsam2_deep_gated.log 2>&1
echo "MedSAM2 GRACE deep done (exit=$?) at $(date)"

# --- Phase 2: MedSAM3 GRACE shallow (GPU 0, ~20GB) ---
echo ""
echo "=== Phase 2: MedSAM3 GRACE shallow (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 conda run -n medseg python scripts/pretrain_teacher_adapter.py \
    $COMMON_ARGS \
    --teacher-type medsam3 \
    --teacher-ckpt auto \
    --output checkpoints/medsam3_gated_adapter_taskA_128.pt \
    --adapter-type gated_residual \
    --task-id taskA_organs \
    --lr 1e-3 \
    > $OUTPUT_ROOT/adapter_medsam3_gated_shallow.log 2>&1
echo "MedSAM3 GRACE shallow done (exit=$?) at $(date)"

# --- Phase 3: MedSAM3 GRACE deep (GPU 0, ~20GB) ---
echo ""
echo "=== Phase 3: MedSAM3 GRACE deep (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 conda run -n medseg python scripts/pretrain_teacher_adapter.py \
    $COMMON_ARGS \
    --teacher-type medsam3 \
    --teacher-ckpt auto \
    --output checkpoints/medsam3_deep_gated_adapter_taskA_128.pt \
    --adapter-type gated_residual \
    --deep-adapter \
    --task-id taskA_organs \
    --lr 1e-3 \
    > $OUTPUT_ROOT/adapter_medsam3_deep_gated.log 2>&1
echo "MedSAM3 GRACE deep done (exit=$?) at $(date)"

echo ""
echo "=== ALL ADAPTERS COMPLETE at $(date) ==="

# Print final summary
echo ""
echo "=== Final Adapter Summary ==="
for f in checkpoints/*adapter*128*.pt; do
    conda run -n medseg python -c "
import torch
s = torch.load('$f', map_location='cpu', weights_only=False)
print('$(basename $f):', 'dice='+str(round(s['val_dice'],4)), 'ep='+str(s['epoch']))
" 2>/dev/null
done
