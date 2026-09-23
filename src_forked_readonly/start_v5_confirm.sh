#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?usage: start_v5_confirm.sh <seed> <gpu>}"
GPU="${2:?usage: start_v5_confirm.sh <seed> <gpu>}"
cd /CSTemp/dyt/GAPLsegnetV2
OUTPUT="outputs/v5_whu100_C_confidence_seed${SEED}"
LOG="logs/v5_whu100_C_confidence_seed${SEED}.log"
mkdir -p "$OUTPUT" logs
if [ -f "$OUTPUT/results.json" ]; then echo "already complete"; exit 0; fi
nohup ./run_v5_confirm.sh "$SEED" "$GPU" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$OUTPUT/train.pid"
echo "$PID"
