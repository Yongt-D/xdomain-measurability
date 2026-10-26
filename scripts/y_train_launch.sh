#!/usr/bin/env bash
# Appendix Y: Inria source-domain training (30141, two GPUs each running half of the seeds). **Committed before any Y number existed.**
# Identical to stage1_launch.sh item by item, only --dataset-path data_view/inria differs; output outputs/y_{ARM}{FUSION}_seed{SEED}/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/root/anaconda3/envs/python38/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
NW=${NW:-6}
ENTRY=${ENTRY:-}          # 30141: ENTRY=scripts/run_fs_sharing.py (file_system sharing-strategy wrapper; the training script is unchanged)
COMMON="--dataset-path data_view/inria --sample-size 1000 --epochs 40 \
--batch-size 4 --lr 3e-4 --weight-decay 1e-4 --support-shots 4 --enable-amp --gpu ${GPU} --num-workers ${NW}"
for SEED in ${SEEDS}; do
  for ARM in A B C; do
    case "${ARM}" in A) FUSION=plain ;; B) FUSION=plain ;; C) FUSION=confidence ;; esac
    OUT="outputs/y_${ARM}${FUSION}_seed${SEED}"
    if [ -f "${OUT}/results.json" ]; then echo "skip ${OUT}"; continue; fi
    mkdir -p "${OUT}"
    echo "=== $(date +%F_%T) $(hostname) GPU${GPU} start ARM=${ARM} SEED=${SEED} -> ${OUT}"
    ${PY} ${ENTRY} src/gaplsegnet_v5_ch5.py ${COMMON} --ablation "${ARM}" --fusion-mode "${FUSION}" \
      --seed "${SEED}" --sample-seed "${SEED}" --output-dir "${OUT}" > "${OUT}.log" 2>&1 || { echo "FAILED ${OUT}"; exit 1; }
    echo "=== $(date +%F_%T) done ${OUT}"
  done
done
echo "Y TRAIN DONE $(hostname) gpu=${GPU} seeds=[${SEEDS}]"
