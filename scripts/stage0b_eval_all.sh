#!/usr/bin/env bash
# Stage-0b cross-domain evaluation: 6 checkpoints x R=5 support draws (K=4, f_min=0.01).
# **Committed before it produced any number.** Stage-0b numbers must not enter any conclusion.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
K=${K:-4}; FMIN=${FMIN:-0.01}
for SEED in 0 1 2; do
  for TAG in Aplain Cconfidence; do
    CKPT="outputs/stage0b_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    for R in 0 1 2 3 4; do
      KEY="fmin${FMIN}_K${K}_r${R}"
      OUT="results/stage0b/${TAG}_seed${SEED}_${KEY}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      echo "=== $(date +%T) ${TAG} seed${SEED} ${KEY}"
      ${PY} scripts/eval_crossdomain.py --checkpoint "$CKPT" --manifest-key "$KEY" \
        --out "$OUT" --gpu "$GPU"
    done
  done
done
echo "STAGE0B EVAL DONE"
