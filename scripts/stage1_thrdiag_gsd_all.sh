#!/usr/bin/env bash
# Appendix R.6 (M5): widened-threshold diagnostic (0.05-0.95) of the stage-1 checkpoints at the degraded points f in {2,4,8}.
# **Committed before it produced any number.** Nature: upper bound / diagnostic (O.5 / P.4), not in the results table.
# The frozen threshold is read from the L1 primary-protocol result JSON; views data_view_gsd/inria_gsd{f} (scripts/make_gsd_views.py); dyt, inference only.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Cconfidence"}
FACTORS=${FACTORS:-"2 4 8"}
VIEW_ROOT=${VIEW_ROOT:-data_view_gsd}
NW=${NW:-4}
KEY=fmin0.01_K4_r0
mkdir -p results/thrdiag_stage1

for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth"
    L1="results/stage1/${TAG}_seed${SEED}_${KEY}.json"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
    FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
    for F in ${FACTORS}; do
      OUT="results/thrdiag_stage1/${TAG}_seed${SEED}_gsd${F}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      VIEW="${VIEW_ROOT}/inria_gsd${F}"
      [ -d "$VIEW/test/image" ] || { echo "missing view: $VIEW"; exit 1; }
      echo "=== $(date +%F_%T) ${TAG} seed${SEED} f=${F} frozen=${FT}"
      ${PY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view "$VIEW" \
        --manifest-key "$KEY" --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --num-workers "$NW"
    done
  done
done
echo "STAGE1 THRDIAG GSD DONE"
