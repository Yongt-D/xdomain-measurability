#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?usage: run_v5_confirm.sh <seed> <gpu>}"
GPU="${2:?usage: run_v5_confirm.sh <seed> <gpu>}"
case "$SEED" in 1|2) ;; *) echo "confirmation seed must be 1 or 2" >&2; exit 2 ;; esac
cd /CSTemp/dyt/GAPLsegnetV2
PYTHON=/CSTemp/fishfield/Anaconda3/envs/dinov3_old_dyt/bin/python
OUTPUT="outputs/v5_whu100_C_confidence_seed${SEED}"
mkdir -p "$OUTPUT" logs
exec "$PYTHON" gaplsegnet_v5.py --dataset-path /CSTemp/dyt/GAPLsegnet/datadir/whu_building --sample-size 100 --support-shots 4 --epochs 160 --batch-size 4 --lr 3e-4 --weight-decay 1e-4 --sample-seed "$SEED" --seed "$SEED" --gpu "$GPU" --enable-amp --pretrained-weights weights/vgg16_bn-6c64b313.pth --ablation C --fusion-mode confidence --output-dir "$OUTPUT"
