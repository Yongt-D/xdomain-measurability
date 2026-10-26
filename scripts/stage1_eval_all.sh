#!/usr/bin/env bash
# Stage-1 cross-domain evaluation (pre-registration appendix F.3). **Committed before any stage-1 number existed.**
#
# Arm C+conf : K in {1,4,8} x R = 5   -> 12 x 3 x 5 = 180 runs
# Arms A / B : K in {1,4,8} x R = 1   -> 12 x 3 x 1 x 2 = 72 runs
#   Collapsing R for A/B is **structural**: src/gaplsegnet_v5_ch5.py:440,476 show that with prototype_mode == "none"
#   the support never enters the forward pass and logits = base_logits, so the A/B output is identically independent
#   of the support and of K (in stage 0b the R standard deviation of arm A was measured as 0.000000).
#   Keeping one run for each K in {1,4,8} treats these 3 numbers as an **empirical check of that identity**:
#   if A/B IoU is not exactly equal across K => the pipeline has a bug; stage1_check_invariance.py judges and stops.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
FMIN=${FMIN:-0.01}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
mkdir -p results/stage1

for SEED in ${SEEDS}; do
  for TAG in Aplain Bplain Cconfidence; do
    CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    if [ "$TAG" = "Cconfidence" ]; then RS="0 1 2 3 4"; else RS="0"; fi
    for K in 1 4 8; do
      for R in ${RS}; do
        KEY="fmin${FMIN}_K${K}_r${R}"
        OUT="results/stage1/${TAG}_seed${SEED}_${KEY}.json"
        [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
        echo "=== $(date +%T) ${TAG} seed${SEED} ${KEY}"
        ${PY} scripts/eval_crossdomain.py --checkpoint "$CKPT" --manifest-key "$KEY" \
          --out "$OUT" --gpu "$GPU"
      done
    done
  done
done
echo "STAGE1 EVAL DONE"
