#!/usr/bin/env bash
# GPU-window watcher: wait until any GPU has >= FREE_NEED MiB free, then
# launch the competition probe (resume-capable) on that GPU and exit.
# Detached (setsid); polls every 5 min; hard timeout 48h.
set -u
FREE_NEED=3000
TIMEOUT_H=48
DEADLINE=$(( $(date +%s) + TIMEOUT_H * 3600 ))
LOG=/home/wangcheng2021/project/GeoEvi-OVIS/sam3_ovis/logs/probe_run.log

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  LINE=$(nvidia-smi --query-gpu=index,memory.used,memory.total \
        --format=csv,noheader,nounits | \
        awk -F, -v need="$FREE_NEED" \
        '{ free = $3 - $2; if (free >= need) print $1, free }' | head -1)
  if [ -n "$LINE" ]; then
    GPU=$(echo "$LINE" | awk '{print $1}')
    FREE=$(echo "$LINE" | awk '{print $2}')
    echo "[$(date +%T)] GPU$GPU has ${FREE}MiB free -> launching probe" >> "$LOG"
    cd /tmp
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    exec /data/wangcheng/envs/sam3/bin/python \
      /home/wangcheng2021/project/GeoEvi-OVIS/sam3_ovis/eval/competition_probe_source.py \
      --n-images 300 --deltas 0.0 0.05 --lambdas 0.5 1.0 2.0 4.0 \
      >> "$LOG" 2>&1 < /dev/null
  fi
  sleep 300
done
echo "[$(date +%T)] watcher timeout after ${TIMEOUT_H}h" >> "$LOG"
