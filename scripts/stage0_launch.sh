#!/usr/bin/env bash
# Stage-0 (variance calibration) launcher -- pre-registration section 3.3 + appendix C.
# **Committed before it produced any number.**
# The effect sizes and point estimates of this stage must not enter any conclusion; they are used only to estimate s_d.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
COMMON="--dataset-path data_view/whu_building --sample-size 100 --epochs 160 \
--batch-size 4 --lr 3e-4 --weight-decay 1e-4 --support-shots 4 --enable-amp --gpu ${GPU}"

for SEED in 0 1 2; do
  for ARM in A C; do
    if [ "$ARM" = "A" ]; then FUSION=plain; else FUSION=confidence; fi
    OUT="outputs/stage0_${ARM}${FUSION}_seed${SEED}"
    if [ -f "${OUT}/results.json" ]; then echo "skip ${OUT} (done)"; continue; fi
    echo "=== $(date +%F_%T) start ARM=${ARM} FUSION=${FUSION} SEED=${SEED} -> ${OUT}"
    ${PY} src/gaplsegnet_v5_ch5.py ${COMMON} \
      --ablation "${ARM}" --fusion-mode "${FUSION}" \
      --seed "${SEED}" --sample-seed "${SEED}" \
      --output-dir "${OUT}" > "${OUT}.log" 2>&1 || { echo "FAILED ${OUT}"; exit 1; }
    echo "=== $(date +%F_%T) done ${OUT}"
  done
done
echo "STAGE0 TRAINING DONE"
