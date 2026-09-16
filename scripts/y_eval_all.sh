#!/usr/bin/env bash
# Appendix Y: all evaluations of the three Inria-source arms x 12 seeds (30141). **Committed before any Y number existed.**
#   l1     : source-val (Inria) threshold selection + WHU target IoU / 21-point sensitivity -> results/y_whu/{tag}_seed{s}_{key}.json (C 15 keys; A/B K4_r0)
#   strata : stratified evaluation (frozen threshold) WHU (256,1024,4096; band 4)       -> results/y_whu_strata/{tag}_seed{s}_{key}.json
#            Massachusetts (23,92,369; band 1,4)                                        -> results/y_mass/{tag}_seed{s}_{key}.json
#   thr    : widened-threshold diagnostic K4_r0: WHU -> results/thrdiag_y_whu/{tag}_seed{s}.json; Mass -> results/thrdiag_y_mass/{tag}_seed{s}.json
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/root/anaconda3/envs/python38/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
STEPS=${STEPS:-"l1 strata thr"}
NW=${NW:-8}; BS=${BS:-8}
ENTRY=${ENTRY:-}          # 30141: ENTRY=scripts/run_fs_sharing.py (file_system sharing-strategy wrapper, same as training; the evaluation scripts are unchanged)
CKPT_DIR=${CKPT_DIR:-}    # non-empty => flat layout ${CKPT_DIR}/y_{tag}_seed{s}.pth (ckpt/y pulled back to the 5080); empty => outputs/y_{tag}_seed{s}/best_gaplsegnet_v5.pth (training machine)
KEYS_C="fmin0.01_K1_r0 fmin0.01_K1_r1 fmin0.01_K1_r2 fmin0.01_K1_r3 fmin0.01_K1_r4 fmin0.01_K4_r0 fmin0.01_K4_r1 fmin0.01_K4_r2 fmin0.01_K4_r3 fmin0.01_K4_r4 fmin0.01_K8_r0 fmin0.01_K8_r1 fmin0.01_K8_r2 fmin0.01_K8_r3 fmin0.01_K8_r4"
WHU_MAN=results/support_manifests/whu_support_manifests.json
MASS_MAN=results/support_manifests/mass_support_manifests.json
for SEED in ${SEEDS}; do for TAG in ${TAGS}; do
  if [ -n "$CKPT_DIR" ]; then CKPT="${CKPT_DIR}/y_${TAG}_seed${SEED}.pth"; else CKPT="outputs/y_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth"; fi
  [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
  if [ "$TAG" = "Cconfidence" ]; then KEYS="$KEYS_C"; else KEYS="fmin0.01_K4_r0"; fi
  if [[ " $STEPS " == *" l1 "* ]]; then
    for KEY in ${KEYS}; do
      OUT="results/y_whu/${TAG}_seed${SEED}_${KEY}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      echo "=== $(date +%F_%T) $(hostname) GPU${GPU} L1 ${TAG} seed${SEED} ${KEY}"
      ${PY} ${ENTRY} scripts/eval_crossdomain.py --checkpoint "$CKPT" --source-view data_view/inria --target-view data_view/whu_building \
        --manifest "$WHU_MAN" --manifest-key "$KEY" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    done
  fi
  L1="results/y_whu/${TAG}_seed${SEED}_fmin0.01_K4_r0.json"
  [ -f "$L1" ] || { echo "missing L1: $L1"; exit 1; }
  FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
  if [[ " $STEPS " == *" strata "* ]]; then
    for KEY in ${KEYS}; do
      OUT="results/y_whu_strata/${TAG}_seed${SEED}_${KEY}.json"
      if [ -f "$OUT" ]; then echo "skip $OUT"; else
        echo "=== $(date +%F_%T) $(hostname) GPU${GPU} strata-whu ${TAG} seed${SEED} ${KEY} thr=${FT}"
        ${PY} ${ENTRY} scripts/eval_stratified.py --checkpoint "$CKPT" --target-view data_view/whu_building --manifest "$WHU_MAN" --manifest-key "$KEY" \
          --threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW" --size-bins 256,1024,4096 --band 4
      fi
      OUT="results/y_mass/${TAG}_seed${SEED}_${KEY}.json"
      if [ -f "$OUT" ]; then echo "skip $OUT"; else
        echo "=== $(date +%F_%T) $(hostname) GPU${GPU} strata-mass ${TAG} seed${SEED} ${KEY} thr=${FT}"
        ${PY} ${ENTRY} scripts/eval_stratified.py --checkpoint "$CKPT" --target-view data_view_mass --manifest "$MASS_MAN" --manifest-key "$KEY" \
          --threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW" --size-bins 23,92,369 --band 1,4
      fi
    done
  fi
  if [[ " $STEPS " == *" thr "* ]]; then
    OUT="results/thrdiag_y_whu/${TAG}_seed${SEED}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; else
      echo "=== $(date +%F_%T) $(hostname) GPU${GPU} thrdiag-whu ${TAG} seed${SEED} frozen=${FT}"
      ${PY} ${ENTRY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view data_view/whu_building --manifest "$WHU_MAN" --manifest-key fmin0.01_K4_r0 \
        --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    fi
    OUT="results/thrdiag_y_mass/${TAG}_seed${SEED}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; else
      echo "=== $(date +%F_%T) $(hostname) GPU${GPU} thrdiag-mass ${TAG} seed${SEED} frozen=${FT}"
      ${PY} ${ENTRY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view data_view_mass --manifest "$MASS_MAN" --manifest-key fmin0.01_K4_r0 \
        --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    fi
  fi
done; done
echo "Y EVAL DONE $(hostname) gpu=${GPU} seeds=[${SEEDS}] tags=[${TAGS}] steps=[${STEPS}]"
