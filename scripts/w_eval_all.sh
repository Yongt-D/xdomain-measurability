#!/usr/bin/env bash
# Appendix W: all evaluations of the two architectures x 12 seeds (run on 4090d; flat checkpoints at CKPT_DIR/{arch}_seed{s}.pth).
# **Committed before any W number existed.**
#   1) L1 counterpart: source-val threshold selection (0.40-0.60) + Inria f=1 target IoU and 21-point sensitivity -> results/w/{arch}_seed{s}_fmin0.01_K4_r0.json
#   2) stratified evaluation (frozen threshold): Inria f in {1,2,4,8} -> results/w_l2/{arch}_seed{s}_gsd{f}.json; Massachusetts -> results/w_mass/{arch}_seed{s}_fmin0.01_K4_r0.json
#   3) widened-threshold diagnostic: Inria f=1 -> results/thrdiag_w/{arch}_seed{s}_gsd1.json; Massachusetts -> results/thrdiag_w_mass/{arch}_seed{s}.json
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
ARCHS=${ARCHS:-"deeplabv3_r50 segformer_b1"}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
CKPT_DIR=${CKPT_DIR:-ckpt/w}
NW=${NW:-8}; BS=${BS:-8}
INRIA_MAN=results/support_manifests/inria_support_manifests.json
MASS_MAN=results/support_manifests/mass_support_manifests.json
KEY=fmin0.01_K4_r0
STEPS=${STEPS:-"l1 strata thr"}
for ARCH in ${ARCHS}; do for SEED in ${SEEDS}; do
  CKPT="${CKPT_DIR}/${ARCH}_seed${SEED}.pth"
  [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
  L1="results/w/${ARCH}_seed${SEED}_${KEY}.json"
  if [[ " $STEPS " == *" l1 "* ]]; then
    if [ -f "$L1" ]; then echo "skip $L1"; else
      echo "=== $(date +%F_%T) $(hostname) L1 ${ARCH} seed${SEED}"
      ${PY} scripts/eval_crossdomain.py --checkpoint "$CKPT" --source-view data_view/whu_building --target-view data_view/inria \
        --manifest "$INRIA_MAN" --manifest-key "$KEY" --out "$L1" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    fi
  fi
  [ -f "$L1" ] || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
  FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
  if [[ " $STEPS " == *" strata "* ]]; then
    for F in 1 2 4 8; do
      OUT="results/w_l2/${ARCH}_seed${SEED}_gsd${F}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      echo "=== $(date +%F_%T) $(hostname) strata ${ARCH} seed${SEED} gsd${F} thr=${FT}"
      ${PY} scripts/eval_stratified.py --checkpoint "$CKPT" --target-view "data_view_gsd/inria_gsd${F}" --manifest "$INRIA_MAN" --manifest-key "$KEY" \
        --threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    done
    OUT="results/w_mass/${ARCH}_seed${SEED}_${KEY}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; else
      echo "=== $(date +%F_%T) $(hostname) strata ${ARCH} seed${SEED} mass thr=${FT}"
      ${PY} scripts/eval_stratified.py --checkpoint "$CKPT" --target-view data_view_mass --manifest "$MASS_MAN" --manifest-key "$KEY" \
        --threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW" --size-bins 23,92,369 --band 1,4
    fi
  fi
  if [[ " $STEPS " == *" thr "* ]]; then
    OUT="results/thrdiag_w/${ARCH}_seed${SEED}_gsd1.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; else
      echo "=== $(date +%F_%T) $(hostname) thrdiag ${ARCH} seed${SEED} inria frozen=${FT}"
      ${PY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view data_view_gsd/inria_gsd1 --manifest "$INRIA_MAN" --manifest-key "$KEY" \
        --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    fi
    OUT="results/thrdiag_w_mass/${ARCH}_seed${SEED}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; else
      echo "=== $(date +%F_%T) $(hostname) thrdiag ${ARCH} seed${SEED} mass frozen=${FT}"
      ${PY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view data_view_mass --manifest "$MASS_MAN" --manifest-key "$KEY" \
        --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
    fi
  fi
done; done
echo "W EVAL DONE $(hostname) archs=[${ARCHS}] seeds=[${SEEDS}] steps=[${STEPS}]"
