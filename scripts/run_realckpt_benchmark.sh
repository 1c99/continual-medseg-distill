#!/bin/bash
# Run the realckpt benchmark: finetune baseline + KD comparison
# Usage: bash scripts/run_realckpt_benchmark.sh [finetune|kd|both]
set -e

MODE=${1:-finetune}
LOGDIR=/home/user/1c/continual-medseg-distill/outputs/realckpt_benchmark

echo "=== RealCkpt Benchmark: mode=$MODE ==="

if [ "$MODE" = "finetune" ] || [ "$MODE" = "both" ]; then
    echo "--- Running finetune baseline ---"
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --config configs/runs/realckpt_finetune_taskA.yaml 2>&1
    echo "--- Finetune complete ---"
fi

if [ "$MODE" = "kd" ] || [ "$MODE" = "both" ]; then
    echo "--- Running KD with MedSAM3 teacher ---"
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --config configs/runs/realckpt_kd_taskA.yaml 2>&1
    echo "--- KD complete ---"
fi

echo "=== Benchmark done ==="
