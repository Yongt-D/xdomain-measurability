#!/usr/bin/env bash
# Appendix U driver: Massachusetts target domain, arm x seed x support key. **Committed before it produced any number.**
# Arm C: fmin0.01 with K in {1,4,8} x r0-r4 (15 keys); arms A/B: only fmin0.01_K4_r0 (structural identity).
# Threshold = the L1 frozen value; strata U.6: --size-bins 23,92,369 --band 1,4. Output results/mass/{tag}_seed{s}_{key}.json
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
KEYS_C=${KEYS_C:-"fmin0.01_K1_r0 fmin0.01_K1_r1 fmin0.01_K1_r2 fmin0.01_K1_r3 fmin0.01_K1_r4 fmin0.01_K4_r0 fmin0.01_K4_r1 fmin0.01_K4_r2 fmin0.01_K4_r3 fmin0.01_K4_r4 fmin0.01_K8_r0 fmin0.01_K8_r1 fmin0.01_K8_r2 fmin0.01_K8_r3 fmin0.01_K8_r4"}
KEYS_AB=${KEYS_AB:-"fmin0.01_K4_r0"}
CKPT_LAYOUT=${CKPT_LAYOUT:-flat}; CKPT_DIR=${CKPT_DIR:-ckpt/stage1}
VIEW=${VIEW:-data_view_mass}; MANIFEST=${MANIFEST:-results/support_manifests/mass_support_manifests.json}
OUTDIR=${OUTDIR:-results/mass}
NW=${NW:-8}; BS=${BS:-4}
SIZE_BINS=${SIZE_BINS:-23,92,369}; BAND=${BAND:-1,4}     # appendices Y/Z may override; default = appendix U
L1_DIR=${L1_DIR:-results/stage1}                          # where the frozen threshold comes from (appendix Y uses results/y_whu)
mkdir -p "$OUTDIR"
[ -d "$VIEW/test/image" ] || { echo "missing view $VIEW"; exit 1; }
for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    case "${CKPT_LAYOUT}" in
      dyt)  CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth" ;;
      flat) CKPT="${CKPT_DIR}/${TAG}_seed${SEED}.pth" ;;
    esac
    L1="${L1_DIR}/${TAG}_seed${SEED}_fmin0.01_K4_r0.json"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
    FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
    if [ "$TAG" = "Cconfidence" ]; then KEYS="$KEYS_C"; else KEYS="$KEYS_AB"; fi
    for KEY in ${KEYS}; do
      OUT="${OUTDIR}/${TAG}_seed${SEED}_${KEY}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      echo "=== $(date +%F_%T) $(hostname) ${TAG} seed${SEED} ${KEY} thr=${FT}"
      ${PY} scripts/eval_stratified.py --checkpoint "$CKPT" --target-view "$VIEW" --manifest "$MANIFEST" --manifest-key "$KEY" \
        --threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW" --size-bins "$SIZE_BINS" --band "$BAND"
    done
  done
done
echo "MASS DONE $(hostname) seeds=[${SEEDS}] tags=[${TAGS}]"
