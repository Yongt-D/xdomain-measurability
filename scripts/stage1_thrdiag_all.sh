#!/usr/bin/env bash
# Appendix Q.3 item 4: run the widened-threshold diagnostic of P.4 (0.05-0.95) once on the **stage-1 checkpoints**.
# **Committed before it produced any number.**
#
# Nature: **upper bound / diagnostic**, not in the results table; only used to quantify the size of the calibration mismatch (Q.3.4 / P.4).
# Setting: the L1 0.3 m target view (data_view/inria), K=4 r=0, arms A and C+conf (Q.3 is about C-A; B can be added via TAGS);
#          the frozen threshold is read from the L1 primary-protocol result JSON of that checkpoint
#          (results/stage1/{tag}_seed{s}_fmin0.01_K4_r0.json), with the same meaning as --frozen-threshold of threshold_range_diag.py.
# Machine: inference only, cross-machine judged E1; default dyt (the L1 result JSONs are local there).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Cconfidence"}
VIEW=${VIEW:-data_view/inria}
NW=${NW:-4}   # number of dataloader workers; affects speed only, not the numbers
KEY=fmin0.01_K4_r0
mkdir -p results/thrdiag_stage1

for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth"
    L1="results/stage1/${TAG}_seed${SEED}_${KEY}.json"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
    OUT="results/thrdiag_stage1/${TAG}_seed${SEED}_gsd1.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
    echo "=== $(date +%F_%T) ${TAG} seed${SEED} frozen=${FT}"
    ${PY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view "$VIEW" \
      --manifest-key "$KEY" --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --num-workers "$NW"
  done
done
echo "STAGE1 THRDIAG DONE"
