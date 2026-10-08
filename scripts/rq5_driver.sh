#!/usr/bin/env bash
# RQ5 (appendix T) driver: pixel-level stratified evaluation, arm x seed x GSD. **Committed before it produced any number.**
# The threshold is read from results/stage1/{tag}_seed{s}_fmin0.01_K4_r0.json (source_val_threshold, frozen in L1) and passed through unchanged.
# Layout: CKPT_LAYOUT=flat -> ${CKPT_DIR}/{tag}_seed{s}.pth; dyt -> outputs/stage1_{tag}_seed{s}/best_gaplsegnet_v5.pth
# Output results/rq5/{tag}_seed{s}_gsd{f}.json; every curve (the 4 factors of an arm x seed) is completed on one machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
FACTORS=${FACTORS:-"1 2 4 8"}
CKPT_LAYOUT=${CKPT_LAYOUT:-flat}
CKPT_DIR=${CKPT_DIR:-ckpt/stage1}
VIEW_ROOT=${VIEW_ROOT:-data_view_gsd}
NW=${NW:-8}
BS=${BS:-8}
KEY=fmin0.01_K4_r0
mkdir -p results/rq5
for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    case "${CKPT_LAYOUT}" in
      dyt)  CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth" ;;
      flat) CKPT="${CKPT_DIR}/${TAG}_seed${SEED}.pth" ;;
    esac
    L1="results/stage1/${TAG}_seed${SEED}_${KEY}.json"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
    FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
    for F in ${FACTORS}; do
      OUT="results/rq5/${TAG}_seed${SEED}_gsd${F}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      VIEW="${VIEW_ROOT}/inria_gsd${F}"
      [ -d "$VIEW/test/image" ] || { echo "missing view: $VIEW"; exit 1; }
      echo "=== $(date +%F_%T) $(hostname) ${TAG} seed${SEED} f=${F} thr=${FT}"
      ${PY} scripts/eval_stratified.py --checkpoint "$CKPT" --target-view "$VIEW" --manifest-key "$KEY" \
        --threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    done
  done
done
echo "RQ5 DONE $(hostname) seeds=[${SEEDS}] tags=[${TAGS}]"
