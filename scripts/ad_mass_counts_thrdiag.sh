#!/usr/bin/env bash
# Appendix AD.2: widened-threshold sweep (0.05-0.95) with per-threshold pixel counts, Massachusetts, three arms x 12 seeds (K=4, draw r0) = 36 runs.
# **Committed before any of these numbers existed.** Same view, manifest key and frozen-threshold source as U.9 (scripts/mass_thrdiag_all.sh);
# the only difference is the script (threshold_range_counts.py stores tp/fp/fn per threshold).
# Nature: descriptive (prevalence-matched threshold and partial PR-AUC as a label-style control); no criterion, no verdict.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
CKPT_DIR=${CKPT_DIR:-ckpt/stage1}          # local 5080: the verified backup E:/Remote/CSTemp_backup/dyt/CH5Geo_cal/ckpt/stage1
VIEW=${VIEW:-data_view_mass}
MANIFEST=${MANIFEST:-results/support_manifests/mass_support_manifests.json}
OUTDIR=${OUTDIR:-results/thrdiag_mass_counts}
NW=${NW:-4}; BS=${BS:-4}
KEY=fmin0.01_K4_r0
L1_DIR=${L1_DIR:-results/stage1}           # frozen threshold source, as in U.9
mkdir -p "$OUTDIR"
for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    CKPT="${CKPT_DIR}/${TAG}_seed${SEED}.pth"
    L1="${L1_DIR}/${TAG}_seed${SEED}_${KEY}.json"
    [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
    [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
    OUT="${OUTDIR}/${TAG}_seed${SEED}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
    echo "=== $(date +%F_%T) $(hostname) ${TAG} seed${SEED} frozen=${FT}"
    ${PY} scripts/threshold_range_counts.py --checkpoint "$CKPT" --target-view "$VIEW" --manifest "$MANIFEST" \
      --manifest-key "$KEY" --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
  done
done
echo "AD COUNTS DONE $(hostname) seeds=[${SEEDS}] tags=[${TAGS}]"
