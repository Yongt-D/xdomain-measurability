#!/usr/bin/env bash
# Appendix X: support-calibrated threshold diagnostic for 36 GAPL checkpoints x 2 target domains (local 5080). **Committed before any of its numbers existed.**
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; GPU=${GPU:-0}
CKPT_DIR=${CKPT_DIR:-ckpt/stage1}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}
DOMAINS=${DOMAINS:-"inria mass"}
for DOM in ${DOMAINS}; do for TAG in ${TAGS}; do for SEED in ${SEEDS}; do
  OUT="results/x/${DOM}_${TAG}_seed${SEED}.json"
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  ${PY} scripts/support_calibrated_threshold.py --checkpoint "${CKPT_DIR}/${TAG}_seed${SEED}.pth" --tag "$TAG" --seed "$SEED" --domain "$DOM" --out "$OUT" --gpu "$GPU"
done; done; done
echo "X DONE"
