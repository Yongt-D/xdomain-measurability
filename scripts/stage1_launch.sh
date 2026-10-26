#!/usr/bin/env bash
# Stage-1 source-domain training (pre-registration appendix F.3). **Committed before any stage-1 number existed.**
#
# Independent unit = training seed, n = 12 (upper bound of section 3.3). Arms A / B / C+conf (appendix C.2).
# The source budget follows the appendix D.2 revision, WHU-1000 / 40 epochs; all other hyper-parameters unchanged.
#
# **Seeds are 100..111; the seeds 0/1/2 of stages 0/0b are not re-used.**
#   Reason: 0/0b were calibration runs, and the analyst already knows their sign counts (1/3 in stage 0, 3/3 in 0b,
#   see section 5 of the internal stage-0b calibration judgement). Re-using those checkpoints would carry that partial unblinding into stage 1.
#   Fresh seeds cost the same compute as re-use, so fresh seeds are used throughout. Fixed before any stage-1 number existed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
COMMON="--dataset-path data_view/whu_building --sample-size 1000 --epochs 40 \
--batch-size 4 --lr 3e-4 --weight-decay 1e-4 --support-shots 4 --enable-amp --gpu ${GPU}"

for SEED in ${SEEDS}; do
  for ARM in A B C; do
    case "${ARM}" in
      A) FUSION=plain ;;
      B) FUSION=plain ;;
      C) FUSION=confidence ;;
    esac
    OUT="outputs/stage1_${ARM}${FUSION}_seed${SEED}"
    if [ -f "${OUT}/results.json" ]; then echo "skip ${OUT}"; continue; fi
    echo "=== $(date +%F_%T) start ARM=${ARM} SEED=${SEED} -> ${OUT}"
    ${PY} src/gaplsegnet_v5_ch5.py ${COMMON} \
      --ablation "${ARM}" --fusion-mode "${FUSION}" \
      --seed "${SEED}" --sample-seed "${SEED}" \
      --output-dir "${OUT}" > "${OUT}.log" 2>&1 || { echo "FAILED ${OUT}"; exit 1; }
    echo "=== $(date +%F_%T) done ${OUT}"
  done
done
echo "STAGE1 TRAIN DONE"
