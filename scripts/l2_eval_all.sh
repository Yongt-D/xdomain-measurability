#!/usr/bin/env bash
# L2 12-seed evaluation (pre-registration section 5 + appendices L.3 / O.2). **This script was committed before any L2 12-seed number existed.**
#
# Setting (appendix O.2, fixed before running):
#   arms A / B / C+conf x seeds 100-111 x GSD factor f in {1,2,4,8} (0.3/0.6/1.2/2.4 m)
#   K = 4, R = 1 (manifest key fmin0.01_K4_r0)
#   inference only, re-using the stage-1 checkpoints, no re-training;
#   the threshold is selected and frozen by scripts/eval_crossdomain.py on the source-domain WHU validation set; the four GSD
#   points re-select it from the same checkpoint on the same machine -- the source view is unchanged, so the result is identical,
#   which l2_analyze.py asserts. The views are built by scripts/make_gsd_views.py (operator in appendix O.1).
#
# Machines: the cross-machine calibration was judged E1 (internal judgement document on cross-machine calibration), so seeds may be split between 30141/30007 and dyt;
#   every curve (the 4 GSD points of an arm x seed) is completed on one machine (O.2).
#   After every JSON, one line "tag seed f hostname" is appended to results/l2/machines.tsv so the analysis script can report cross-machine drift.
#
# Checkpoint layout (environment variable CKPT_LAYOUT):
#   dyt  : outputs/stage1_{tag}_seed{s}/best_gaplsegnet_v5.pth (original location on dyt)
#   flat : ${CKPT_DIR}/{tag}_seed{s}.pth (flat layout used when copying to 30141/30007)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
FACTORS=${FACTORS:-"1 2 4 8"}
CKPT_LAYOUT=${CKPT_LAYOUT:-dyt}
CKPT_DIR=${CKPT_DIR:-ckpt/stage1}
VIEW_ROOT=${VIEW_ROOT:-data_view_gsd}
NW=${NW:-4}
KEY=fmin0.01_K4_r0
HOST=$(hostname)
mkdir -p results/l2

for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    case "${CKPT_LAYOUT}" in
      dyt)  CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth" ;;
      flat) CKPT="${CKPT_DIR}/${TAG}_seed${SEED}.pth" ;;
      *) echo "unknown CKPT_LAYOUT=${CKPT_LAYOUT}"; exit 1 ;;
    esac
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    for F in ${FACTORS}; do
      OUT="results/l2/${TAG}_seed${SEED}_gsd${F}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      VIEW="${VIEW_ROOT}/inria_gsd${F}"
      [ -d "$VIEW/test/image" ] || { echo "missing view: $VIEW"; exit 1; }
      echo "=== $(date +%F_%T) ${HOST} ${TAG} seed${SEED} f=${F}"
      ${PY} scripts/eval_crossdomain.py --checkpoint "$CKPT" --target-view "$VIEW" \
        --manifest-key "$KEY" --out "$OUT" --gpu "$GPU" --num-workers "$NW"
      printf '%s\t%s\t%s\t%s\n' "$TAG" "$SEED" "$F" "$HOST" >> results/l2/machines.tsv
    done
  done
done
echo "L2 EVAL DONE ${HOST} seeds=[${SEEDS}]"
