#!/usr/bin/env bash
# Run GRACE component ablation on continual A→B→C→D→E (5-task benchmark).
#
# 5 variants on 4 GPUs:
#   1. Standard adapter only  (no GRACE)
#   2. TRA only               (frozen core + residuals)
#   3. TRA + CGAD             (+ confidence gate)
#   4. TRA + CPA              (+ prototypes)
#   5. Full GRACE             (TRA + CGAD + CPA)
#
# Usage:
#   bash scripts/run_grace_ablation.sh              # 5-task, 50 epochs
#   bash scripts/run_grace_ablation.sh --smoke      # 2-task AB, 5 epochs
#   bash scripts/run_grace_ablation.sh --seed=123   # different seed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse args
SMOKE=false
SEED=42
for arg in "$@"; do
    case $arg in
        --smoke) SMOKE=true ;;
        --seed=*) SEED="${arg#*=}" ;;
    esac
done

if [ "$SMOKE" = true ]; then
    TASK_CFG="configs/tasks/totalseg_AB_sequence.yaml"
    DATASET_CFG="configs/datasets/totalseg_train_clean_smoke.yaml"
    OVERRIDE_CFG="configs/overrides/smoke_5ep.yaml"
    TAG="grace_ablation_smoke_s${SEED}"
else
    TASK_CFG="configs/tasks/totalseg_ABCDE_sequence.yaml"
    DATASET_CFG="configs/datasets/totalseg_clean.yaml"
    OVERRIDE_CFG="configs/overrides/full_50ep.yaml"
    TAG="grace_ablation_s${SEED}"
fi

METHOD_CFG="configs/methods/distill_replay_ewc_medsam2_gated.yaml"
BASE_OUTPUT="outputs/${TAG}"
mkdir -p "$BASE_OUTPUT"

echo "============================================="
echo "GRACE Component Ablation"
echo "  Task config: $TASK_CFG"
echo "  Dataset: $DATASET_CFG"
echo "  Seed: $SEED"
echo "  Output: $BASE_OUTPUT"
echo "============================================="

# Ablation variants and GPU assignments (5 variants, 4 GPUs → GPU 0 gets 2)
VARIANTS=(standard_adapter tra_only tra_cgad tra_cpa full_grace)
OVERRIDE_NAMES=(ablation_standard_adapter ablation_tra_only ablation_tra_cgad ablation_tra_cpa ablation_full_grace)
GPUS=(0 1 2 3 0)

PIDS=()

for i in "${!VARIANTS[@]}"; do
    VARIANT="${VARIANTS[$i]}"
    OVERRIDE="${OVERRIDE_NAMES[$i]}"
    GPU="${GPUS[$i]}"
    OUT_DIR="${BASE_OUTPUT}/${VARIANT}"
    LOG_FILE="${BASE_OUTPUT}/${VARIANT}.log"

    echo "[GPU $GPU] Starting ${VARIANT} → $OUT_DIR"

    # Merge training override + ablation override into a temp file
    MERGED_OVERRIDE=$(mktemp "${BASE_OUTPUT}/${VARIANT}_override_XXXXXX.yaml")
    python -c "
import yaml, sys
a = yaml.safe_load(open('$OVERRIDE_CFG')) or {}
b = yaml.safe_load(open('configs/overrides/${OVERRIDE}.yaml')) or {}
def merge(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            merge(d1[k], v)
        else:
            d1[k] = v
    return d1
yaml.dump(merge(a, b), open('$MERGED_OVERRIDE', 'w'), default_flow_style=False)
"

    CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n medseg python scripts/run_continual.py \
        --base-config configs/base.yaml \
        --task-config "$TASK_CFG" \
        --dataset-config "$DATASET_CFG" \
        --method-config "$METHOD_CFG" \
        --override-config "$MERGED_OVERRIDE" \
        --output-dir "$OUT_DIR" \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
    sleep 2
done

echo ""
echo "Launched ${#PIDS[@]} jobs. PIDs: ${PIDS[*]}"
echo "Waiting for all to complete..."

# Wait and report
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    VARIANT="${VARIANTS[$i]}"
    if wait "$PID"; then
        echo "[DONE] $VARIANT (PID $PID) — SUCCESS"
    else
        echo "[DONE] $VARIANT (PID $PID) — FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================="
echo "GRACE ablation complete. Failed: $FAILED / ${#VARIANTS[@]}"
echo "Results: $BASE_OUTPUT/"
echo "============================================="

exit $FAILED
