#!/usr/bin/env bash
# Appendix W: training driver for the standard architectures (each architecture entirely on one machine; X1 deeplabv3_r50 on dyt, X2 segformer_b1 on 4090d).
# **Committed before any W number existed.** Hyper-parameters mirror stage1_launch.sh item by item; no tuning.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
ARCH=${ARCH:?"ARCH=deeplabv3_r50|segformer_b1"}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
NW=${NW:-4}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
COMMON="--dataset-path data_view/whu_building --sample-size 1000 --epochs 40 --batch-size 4 --lr 3e-4 --weight-decay 1e-4 --enable-amp --gpu ${GPU} --num-workers ${NW}"
for SEED in ${SEEDS}; do
  OUT="outputs/w_${ARCH}_seed${SEED}"
  if [ -f "${OUT}/results.json" ]; then echo "skip ${OUT}"; continue; fi
  mkdir -p "${OUT}"
  echo "=== $(date +%F_%T) $(hostname) start ARCH=${ARCH} SEED=${SEED} -> ${OUT}"
  ${PY} scripts/train_arch.py --arch "${ARCH}" ${COMMON} --seed "${SEED}" --sample-seed "${SEED}" --output-dir "${OUT}" > "${OUT}.log" 2>&1 || { echo "FAILED ${OUT}"; exit 1; }
  tail -1 "${OUT}.log"
done
echo "W TRAIN DONE $(hostname) arch=${ARCH} seeds=[${SEEDS}]"
