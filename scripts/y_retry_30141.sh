#!/usr/bin/env bash
# Shared GPU on 30141: when other users' jobs cause a CUDA OOM, y_train_launch.sh exits with FAILED; this wrapper retries every 5 minutes
# (completed seeds are skipped by the launcher via results.json) until Y TRAIN DONE.
cd /CSTemp/dyt/CH5Geo_cal
GPU=${GPU:?}; SEEDS=${SEEDS:?}
LOG=outputs/y_train_gpu${GPU}.log
ulimit -n 65536
n=0
while true; do
  n=$((n+1))
  echo "=== $(date +%F_%T) retry-wrapper attempt $n gpu=$GPU" >> "$LOG"
  GPU=$GPU SEEDS="$SEEDS" NW=8 ENTRY=scripts/run_fs_sharing.py bash scripts/y_train_launch.sh >> "$LOG" 2>&1 && break
  echo "=== $(date +%F_%T) launcher exited non-zero; free=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i $GPU); sleep 300" >> "$LOG"
  sleep 300
done
echo "=== $(date +%F_%T) retry-wrapper finished gpu=$GPU after $n attempts" >> "$LOG"
