#!/usr/bin/env bash
# Appendix Y evaluation wrapper (shared GPU on 30141): first the Y.3.2 per-GPU calibration (A seed100 K4_r0 WHU -> results/y_calib/gpu{GPU}_*.json),
# then scripts/y_eval_all.sh (skipping completed outputs); when other users' jobs cause a CUDA OOM exit, retry every 5 minutes until Y EVAL DONE.
# Only retry, skip and directory preparation; no evaluation parameter is changed; ENTRY=scripts/run_fs_sharing.py as in training.
cd /CSTemp/dyt/CH5Geo_cal
GPU=${GPU:?}; SEEDS=${SEEDS:?}; STEPS=${STEPS:-"l1 strata thr"}
LOG=outputs/y_eval_gpu${GPU}.log
PY=${PY:-/root/anaconda3/envs/python38/bin/python}
ulimit -n 65536
mkdir -p results/y_calib results/y_whu results/y_whu_strata results/y_mass results/thrdiag_y_whu results/thrdiag_y_mass
CAL=results/y_calib/gpu${GPU}_Aplain_seed100_fmin0.01_K4_r0.json
n=0
while true; do
  n=$((n+1))
  echo "=== $(date +%F_%T) y-eval-wrapper attempt $n gpu=$GPU seeds=[$SEEDS] steps=[$STEPS]" >> "$LOG"
  ok=1
  if [ ! -f "$CAL" ]; then
    echo "=== $(date +%F_%T) Y.3.2 calib gpu=$GPU -> $CAL" >> "$LOG"
    ${PY} scripts/run_fs_sharing.py scripts/eval_crossdomain.py --checkpoint outputs/y_Aplain_seed100/best_gaplsegnet_v5.pth \
      --source-view data_view/inria --target-view data_view/whu_building \
      --manifest results/support_manifests/whu_support_manifests.json --manifest-key fmin0.01_K4_r0 \
      --out "$CAL" --gpu "$GPU" --batch-size 8 --num-workers 8 >> "$LOG" 2>&1 || ok=0
  fi
  if [ "$ok" = 1 ]; then
    GPU=$GPU SEEDS="$SEEDS" STEPS="$STEPS" NW=8 BS=8 ENTRY=scripts/run_fs_sharing.py bash scripts/y_eval_all.sh >> "$LOG" 2>&1 && break
  fi
  echo "=== $(date +%F_%T) non-zero exit; free=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i $GPU); sleep 300" >> "$LOG"
  sleep 300
done
echo "=== $(date +%F_%T) Y EVAL WRAPPER DONE gpu=$GPU after $n attempts" >> "$LOG"
