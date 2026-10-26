#!/usr/bin/env bash
# Fixed-subset seed repetition (rebuttal preparation; NOT part of the pre-specified protocol).
# Same training as scripts/stage1_launch.sh, but --sample-seed is held at 100 for every run, so all twelve runs of an arm
# share the same 1,000-image WHU subset, the same 800/200 split and the same loader order (the loader generators are
# derived from sample_seed); only --seed, i.e. the initialisation (and non-deterministic kernels), varies.
# Comparing the in-domain / cross-domain seed SD of these runs with the stage-1 runs (where the subset varies with the
# seed) separates the subset component from the initialisation component of the run-to-run variability behind R1.
# Usage (local 5080):  PY=/d/Anaconda/envs/beoc/python.exe GPU=0 bash scripts/fixedsubset_launch.sh
#         remote:      setsid nohup bash scripts/fixedsubset_launch.sh > fs_train.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniforge3/envs/beoc/bin/python}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109 110 111"}
SAMPLE_SEED=${SAMPLE_SEED:-100}
ARMS=${ARMS:-"A B C"}
COMMON="--dataset-path data_view/whu_building --sample-size 1000 --epochs 40 \
--batch-size 4 --lr 3e-4 --weight-decay 1e-4 --support-shots 4 --enable-amp --gpu ${GPU}"
for SEED in ${SEEDS}; do
  for ARM in ${ARMS}; do
    case "${ARM}" in
      A) FUSION=plain ;;
      B) FUSION=plain ;;
      C) FUSION=confidence ;;
    esac
    OUT="outputs/fs${SAMPLE_SEED}_${ARM}${FUSION}_seed${SEED}"
    if [ -f "${OUT}/results.json" ]; then echo "skip ${OUT}"; continue; fi
    echo "=== $(date +%F_%T) start ARM=${ARM} SEED=${SEED} SAMPLE_SEED=${SAMPLE_SEED} -> ${OUT}"
    ${PY} src/gaplsegnet_v5_ch5.py ${COMMON} \
      --ablation "${ARM}" --fusion-mode "${FUSION}" \
      --seed "${SEED}" --sample-seed "${SAMPLE_SEED}" \
      --output-dir "${OUT}" > "${OUT}.log" 2>&1 || { echo "FAILED ${OUT}"; exit 1; }
    echo "=== $(date +%F_%T) done ${OUT}"
  done
done
echo "FIXEDSUBSET TRAIN DONE"
