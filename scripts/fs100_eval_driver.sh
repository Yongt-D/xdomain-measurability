#!/usr/bin/env bash
# Fixed-subset (fs100) cross-domain evaluation -- rebuttal preparation, NOT part of the pre-specified protocol.
# Applies the stage-1 evaluation protocol unchanged to the fixed-subset checkpoints trained by scripts/fixedsubset_launch.sh
# (outputs/fs100_{tag}_seed{s}/best_gaplsegnet_v5.pth; --sample-seed 100 for every run, only --seed varies):
#   Inria (L1, as scripts/stage1_eval_all.sh): arm C  K in {1,4,8} x draws r0-r4; arms A/B  K in {1,4,8} x r0
#           -> results/fs100_inria/{tag}_seed{s}_fmin0.01_K{k}_r{r}.json  (threshold frozen on the WHU validation split)
#   Massachusetts (appendix U, via scripts/mass_driver.sh): arm C 15 keys, arms A/B fmin0.01_K4_r0; frozen threshold from the Inria K4_r0 file
#           -> results/fs100_mass/{tag}_seed{s}_{key}.json
# K = 4 runs first (then Massachusetts, then K = 1 and 8) so that the seed-SD comparison with stage 1 (R1: 4.2 / 13.7 / 9.0 on Inria,
# 9.6 / 10.8 / 7.7 on Massachusetts) is available before the remaining keys finish.
# mass_driver.sh expects a flat checkpoint directory, so ckpt/fs100/{tag}_seed{s}.pth are symlinks to the outputs/ checkpoints.
# Usage (dyt, two GPUs):
#   SEEDS="100 101 102 103 104 105" GPU=0 setsid nohup bash scripts/fs100_eval_driver.sh > fs_eval_gpu0.log 2>&1 < /dev/null &
#   SEEDS="106 107 108 109 110 111" GPU=1 setsid nohup bash scripts/fs100_eval_driver.sh > fs_eval_gpu1.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
KS=${KS:-"4 1 8"}
TAGS=${TAGS:-"Aplain Bplain Cconfidence"}   # overridable so that the K=8 arm-C runs can be moved to the 24 GB card
FMIN=0.01
INRIA_OUT=results/fs100_inria
MASS_OUT=results/fs100_mass
mkdir -p "$INRIA_OUT" "$MASS_OUT" ckpt/fs100

for SEED in ${SEEDS}; do
  for TAG in ${TAGS}; do
    SRC="outputs/fs100_${TAG}_seed${SEED}/best_gaplsegnet_v5.pth"
    [ -f "$SRC" ] || { echo "missing checkpoint: $SRC"; exit 1; }
    [ -e "ckpt/fs100/${TAG}_seed${SEED}.pth" ] || ln -s "../../${SRC}" "ckpt/fs100/${TAG}_seed${SEED}.pth"
  done
done

for K in ${KS}; do
  for SEED in ${SEEDS}; do
    for TAG in ${TAGS}; do
      CKPT="ckpt/fs100/${TAG}_seed${SEED}.pth"
      if [ "$TAG" = "Cconfidence" ]; then RS="0 1 2 3 4"; else RS="0"; fi
      for R in ${RS}; do
        KEY="fmin${FMIN}_K${K}_r${R}"
        OUT="${INRIA_OUT}/${TAG}_seed${SEED}_${KEY}.json"
        [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
        echo "=== $(date +%F_%T) inria ${TAG} seed${SEED} ${KEY}"
        ${PY} scripts/eval_crossdomain.py --checkpoint "$CKPT" --manifest-key "$KEY" --out "$OUT" --gpu "$GPU"
      done
    done
  done
  if [ "$K" = "4" ]; then
    echo "=== $(date +%F_%T) massachusetts seeds=[${SEEDS}]"
    PY="$PY" GPU="$GPU" SEEDS="$SEEDS" CKPT_LAYOUT=flat CKPT_DIR=ckpt/fs100 L1_DIR="$INRIA_OUT" OUTDIR="$MASS_OUT" \
      bash scripts/mass_driver.sh
  fi
done
echo "FS100 EVAL DONE seeds=[${SEEDS}]"
