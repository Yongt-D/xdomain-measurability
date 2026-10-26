#!/usr/bin/env bash
# Appendix F.1: formal re-run of the five mandatory causal ablations of section 4 on the **12 stage-1 C+conf checkpoints**.
# **Committed before it produced any number.**
#
# - criteria and implementation follow scripts/causal_ablations.py unchanged (the thresholds of section 4 / F.2 / G.2 are untouched;
#   ablation 3 was judged negative in 0b and is not re-opened, H.3 U3);
# - threshold = the threshold frozen for that checkpoint on the source-domain WHU validation set in the L1 primary protocol
#   (results/stage1/Cconfidence_seed{s}_fmin0.01_K4_r0.json, field source_val_threshold); target labels are not used;
# - target domain Inria 0.3 m view, K=4, manifest fmin0.01_K4_r0, as in L1;
# - machine: dyt (same machine as L1; inference only).
# - aggregation rule (fixed before running, as strict as the 0b judgement): **all 12 seeds must pass for the item to pass**,
#   judged by stage1_ablations_summarize.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
NW=${NW:-4}
KEY=fmin0.01_K4_r0
mkdir -p results/ablations

for SEED in ${SEEDS}; do
  CKPT="outputs/stage1_Cconfidence_seed${SEED}/best_gaplsegnet_v5.pth"
  L1="results/stage1/Cconfidence_seed${SEED}_${KEY}.json"
  OUT="results/ablations/stage1_Cconfidence_seed${SEED}.json"
  [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
  [ -f "$L1" ]   || { echo "missing L1 result (source of the frozen threshold): $L1"; exit 1; }
  [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
  FT=$(${PY} -c "import json;print(json.load(open('${L1}'))['source_val_threshold'])")
  echo "=== $(date +%F_%T) Cconfidence seed${SEED} threshold=${FT}"
  ${PY} scripts/causal_ablations.py --checkpoint "$CKPT" --threshold "$FT" \
    --manifest-key "$KEY" --out "$OUT" --gpu "$GPU" --num-workers "$NW"
done
echo "STAGE1 ABLATIONS DONE"
