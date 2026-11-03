#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: run_v5_seed0.sh <plain|confidence> <gpu>}"
GPU="${2:?usage: run_v5_seed0.sh <plain|confidence> <gpu>}"
case "$MODE" in plain|confidence) ;; *) echo "invalid mode" >&2; exit 2 ;; esac
cd /CSTemp/dyt/GAPLsegnetV2
PYTHON=/CSTemp/fishfield/Anaconda3/envs/dinov3_old_dyt/bin/python
OUTPUT="outputs/v5_whu100_C_${MODE}_seed0"
mkdir -p "$OUTPUT" logs
exec "$PYTHON" gaplsegnet_v5.py --dataset-path /CSTemp/dyt/GAPLsegnet/datadir/whu_building --sample-size 100 --support-shots 4 --epochs 160 --batch-size 4 --lr 3e-4 --weight-decay 1e-4 --sample-seed 0 --seed 0 --gpu "$GPU" --enable-amp --pretrained-weights weights/vgg16_bn-6c64b313.pth --ablation C --fusion-mode "$MODE" --output-dir "$OUTPUT"
