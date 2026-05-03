#!/usr/bin/env bash
# Queue-style launcher for OGBench offline-to-online sweeps on a multi-GPU box.
#
# Polls nvidia-smi for GPUs whose used memory is below MEM_THRESHOLD_MB, treats
# those as free, and dispatches one pending job onto each. GPUs already busy
# with other runs (e.g. previously launched cube smoke jobs) are left alone;
# this script will simply wait until they finish and reclaim those slots as
# they free up.
#
# 3 yamls x 8 seeds = 24 jobs total. Each run is 1M offline + 1M online,
# continuous in one wandb run. No metrics.csv / no config.yaml / no checkpoints.
#
# Usage:
#   tmux new -s fpot_online
#   bash scripts/launch_online_vast.sh
#   # detach: Ctrl-b d ; reattach: tmux attach -t fpot_online
#
# Env knobs (override by exporting before running):
#   MEM_THRESHOLD_MB   GPU is "free" if used memory < this many MB (default 500)
#   DISPATCH_DELAY     seconds to wait after launching one job before next poll (default 30)
#   POLL_INTERVAL      seconds between nvidia-smi polls when no GPU is free   (default 30)

set -u

MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB:-500}
DISPATCH_DELAY=${DISPATCH_DELAY:-30}
POLL_INTERVAL=${POLL_INTERVAL:-30}

YAMLS=(
  experiments/online/humanoidmaze_medium.yaml
  experiments/online/cube_double.yaml
  experiments/online/puzzle_4x4.yaml
)

queue=()
for yaml in "${YAMLS[@]}"; do
  for idx in 0 1 2 3 4 5 6 7; do
    queue+=("${yaml}|${idx}")
  done
done

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[$(date +%H:%M:%S)] Detected ${NUM_GPUS} GPUs.  Pending jobs: ${#queue[@]}"
echo "[$(date +%H:%M:%S)] free-threshold=${MEM_THRESHOLD_MB} MB  poll=${POLL_INTERVAL}s  dispatch_delay=${DISPATCH_DELAY}s"

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v t="$MEM_THRESHOLD_MB" -F', *' '$2 < t { print $1 }'
}

PIDS=()

while (( ${#queue[@]} > 0 )); do
  mapfile -t free < <(free_gpus)
  if (( ${#free[@]} == 0 )); then
    echo "[$(date +%H:%M:%S)] no free GPU (all >= ${MEM_THRESHOLD_MB} MB used).  sleeping ${POLL_INTERVAL}s."
    sleep "$POLL_INTERVAL"
    continue
  fi

  for gpu in "${free[@]}"; do
    (( ${#queue[@]} > 0 )) || break
    spec="${queue[0]}"
    queue=("${queue[@]:1}")
    yaml="${spec%|*}"
    idx="${spec#*|}"
    name=$(basename "$yaml" .yaml)
    echo "[$(date +%H:%M:%S)] dispatch yaml=${name} idx=${idx} gpu=${gpu}  remaining=${#queue[@]}"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/sweep_runner.py run \
      --sweep "$yaml" --idx "$idx" \
      --override logging.save_csv=false \
      --override logging.save_config=false \
      --override logging.save_checkpoint=false \
      >/dev/null 2>&1 &
    PIDS+=($!)
    # Let the freshly-spawned JAX process claim GPU memory before next nvidia-smi poll,
    # so it shows as busy and we don't double-book this GPU.
    sleep "$DISPATCH_DELAY"
  done
done

echo "[$(date +%H:%M:%S)] all jobs dispatched.  waiting on ${#PIDS[@]} background processes."
wait
echo "[$(date +%H:%M:%S)] All jobs complete."
