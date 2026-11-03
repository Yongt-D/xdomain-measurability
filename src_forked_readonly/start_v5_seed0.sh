#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: start_v5_seed0.sh <plain|confidence> <gpu>}"
GPU="${2:?usage: start_v5_seed0.sh <plain|confidence> <gpu>}"
cd /CSTemp/dyt/GAPLsegnetV2
OUTPUT="outputs/v5_whu100_C_${MODE}_seed0"
LOG="logs/v5_whu100_C_${MODE}_seed0.log"
mkdir -p "$OUTPUT" logs
if [ -f "$OUTPUT/results.json" ]; then echo "already complete"; exit 0; fi
nohup ./run_v5_seed0.sh "$MODE" "$GPU" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$OUTPUT/train.pid"
echo "$PID"
