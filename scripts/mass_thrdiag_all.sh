#!/usr/bin/env bash
# Appendix U.9: widened-threshold diagnostic (0.05-0.95) on the Massachusetts target domain, A / B / C+conf x 12 seeds, K=4 r=0.
# **Committed before it produced any number.** Nature: upper bound / diagnostic (Q.4: the oracle uses target labels; not in the results table).
# The frozen threshold is read from the L1 result JSON (results/stage1/{tag}_seed{s}_fmin0.01_K4_r0.json, field source_val_threshold).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
VIEW=${VIEW:-data_view_mass}
MANIFEST=${MANIFEST:-results/support_manifests/mass_support_manifests.json}
CKPT_LAYOUT=${CKPT_LAYOUT:-flat}; CKPT_DIR=${CKPT_DIR:-ckpt/stage1}
OUTDIR=${OUTDIR:-results/thrdiag_mass}
NW=${NW:-4}; BS=${BS:-4}
KEY=fmin0.01_K4_r0
L1_DIR=${L1_DIR:-results/stage1}      # where the frozen threshold comes from (appendix Y uses results/y_whu)
mkdir -p "$OUTDIR"
for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    case "${CKPT_LAYOUT}" in
      dyt)  CKPT="outputs/stage1_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth" ;;
      flat) CKPT="${CKPT_DIR}/${TAG}_seed${SEED}.pth" ;;
    esac
    L1="${L1_DIR}/${TAG}_seed${SEED}_${KEY}.json"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
    OUT="${OUTDIR}/${TAG}_seed${SEED}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
    echo "=== $(date +%F_%T) $(hostname) ${TAG} seed${SEED} frozen=${FT}"
    ${PY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view "$VIEW" --manifest "$MANIFEST" \
      --manifest-key "$KEY" --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
  done
done
echo "MASS THRDIAG DONE $(hostname) tags=[${TAGS}]"
