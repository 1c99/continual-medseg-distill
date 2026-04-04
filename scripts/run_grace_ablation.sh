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
# Runs 4 variants in parallel (one per GPU), then queues the 5th after
# the first GPU frees up. No GPU oversubscription.
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
RUN_ID="$(date +%Y%m%d_%H%M%S)"
BASE_OUTPUT="outputs/${TAG}_${RUN_ID}"
mkdir -p "$BASE_OUTPUT"

echo "============================================="
echo "GRACE Component Ablation"
echo "  Task config: $TASK_CFG"
echo "  Dataset: $DATASET_CFG"
echo "  Seed: $SEED"
echo "  Run ID: $RUN_ID"
echo "  Output: $BASE_OUTPUT"
echo "============================================="

# Ablation variants
VARIANTS=(standard_adapter tra_only tra_cgad tra_cpa full_grace)
OVERRIDE_NAMES=(ablation_standard_adapter ablation_tra_only ablation_tra_cgad ablation_tra_cpa ablation_full_grace)

# Helper: launch one variant on a specific GPU
launch_variant() {
    local idx=$1
    local gpu=$2
    local VARIANT="${VARIANTS[$idx]}"
    local OVERRIDE="${OVERRIDE_NAMES[$idx]}"
    local OUT_DIR="${BASE_OUTPUT}/${VARIANT}"
    local LOG_FILE="${BASE_OUTPUT}/${VARIANT}.log"

    # Merge training override + ablation override into a temp file
    local MERGED_OVERRIDE="${BASE_OUTPUT}/${VARIANT}_override.yaml"
    python -c "
import yaml
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

    echo "[GPU $gpu] Starting ${VARIANT} → $OUT_DIR"

    CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n medseg python scripts/run_continual.py \
        --base-config configs/base.yaml \
        --task-config "$TASK_CFG" \
        --dataset-config "$DATASET_CFG" \
        --method-config "$METHOD_CFG" \
        --override-config "$MERGED_OVERRIDE" \
        --output-dir "$OUT_DIR" \
        > "$LOG_FILE" 2>&1 &

    echo $!
}

# Phase 1: Launch first 4 variants on GPUs 0-3
PIDS=()
for i in 0 1 2 3; do
    PID=$(launch_variant $i $i)
    PIDS+=($PID)
    sleep 2
done

echo ""
echo "Launched 4 jobs. PIDs: ${PIDS[*]}"
echo "Waiting for first to finish before launching variant 5..."

# Wait for ANY of the first 4 to finish, then launch variant 5 on that GPU
FAILED=0
FIFTH_LAUNCHED=false
for i in 0 1 2 3; do
    PID="${PIDS[$i]}"
    VARIANT="${VARIANTS[$i]}"
    if wait "$PID"; then
        echo "[DONE] $VARIANT (PID $PID) — SUCCESS"
    else
        echo "[DONE] $VARIANT (PID $PID) — FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi

    # Launch 5th variant on the first freed GPU
    if [ "$FIFTH_LAUNCHED" = false ]; then
        FIFTH_LAUNCHED=true
        FIFTH_PID=$(launch_variant 4 $i)
        PIDS+=($FIFTH_PID)
    fi
done

# Wait for 5th variant
FIFTH_PID="${PIDS[4]}"
VARIANT="${VARIANTS[4]}"
if wait "$FIFTH_PID"; then
    echo "[DONE] $VARIANT (PID $FIFTH_PID) — SUCCESS"
else
    echo "[DONE] $VARIANT (PID $FIFTH_PID) — FAILED (exit $?)"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "============================================="
echo "GRACE ablation complete. Failed: $FAILED / ${#VARIANTS[@]}"
echo "Results: $BASE_OUTPUT/"
echo "============================================="

exit $FAILED
