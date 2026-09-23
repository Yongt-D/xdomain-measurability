#!/usr/bin/env bash
# Appendix AB.2: widened-threshold sweep (0.05-0.95) of arm C on Massachusetts for support draws r1-r4 (K=4), 12 seeds x 4 draws = 48 runs.
# **Committed before any of these numbers existed.** Same script, view, manifest and frozen-threshold source as U.9 (scripts/mass_thrdiag_all.sh);
# only --manifest-key differs. Arms A/B do not depend on the support draw and re-use the U.9 r0 files.
# Nature: descriptive (checks whether the U.9 operating-point decomposition depends on the single draw r0); no criterion, no verdict.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
DRAWS=${DRAWS:-"1 2 3 4"}
CKPT_DIR=${CKPT_DIR:-ckpt/stage1}          # local 5080: the verified backup E:/Remote/CSTemp_backup/dyt/CH5Geo_cal/ckpt/stage1
VIEW=${VIEW:-data_view_mass}
MANIFEST=${MANIFEST:-results/support_manifests/mass_support_manifests.json}
OUTDIR=${OUTDIR:-results/thrdiag_mass_draws}
NW=${NW:-4}; BS=${BS:-4}
L1_DIR=${L1_DIR:-results/stage1}           # frozen threshold source, as in U.9
mkdir -p "$OUTDIR"
for SEED in ${SEEDS}; do
  CKPT="${CKPT_DIR}/Cconfidence_seed${SEED}.pth"
  L1="${L1_DIR}/Cconfidence_seed${SEED}_fmin0.01_K4_r0.json"
  [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
  [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
  FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
  for R in ${DRAWS}; do
    OUT="${OUTDIR}/Cconfidence_seed${SEED}_r${R}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== $(date +%F_%T) $(hostname) Cconfidence seed${SEED} r${R} frozen=${FT}"
    ${PY} scripts/threshold_range_diag.py --checkpoint "$CKPT" --target-view "$VIEW" --manifest "$MANIFEST" \
      --manifest-key "fmin0.01_K4_r${R}" --frozen-threshold "$FT" --out "$OUT" --gpu "$GPU" --batch-size "$BS" --num-workers "$NW"
  done
done
echo "AB DRAWS DONE $(hostname) seeds=[${SEEDS}] draws=[${DRAWS}]"
