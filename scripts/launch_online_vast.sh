#!/usr/bin/env bash
# Launch the OGBench offline-to-online FPOT runs on a single multi-GPU box
# (no Slurm, no per-job stdout logs, no metrics.csv, no checkpoints — for use
# on space-constrained VastAI machines).
#
# 3 yamls x 8 seeds = 24 jobs, fanned out across NUM_GPUS GPUs.
# Each run is 1M offline + 1M online (continuous in one wandb run).
#
# Usage:
#   tmux new -s fpot_online
#   bash scripts/launch_online_vast.sh
#   # detach: Ctrl-b d ; reattach: tmux attach -t fpot_online

set -u

NUM_GPUS=${NUM_GPUS:-8}
PIDS=()
JOBS=()

YAMLS=(
  experiments/online/humanoidmaze_medium.yaml
  experiments/online/cube_double.yaml
  experiments/online/puzzle_4x4.yaml
)

# 8 seeds per yaml -> idx 0..7 (seed: [1..8] is the only sweep axis)
for yaml in "${YAMLS[@]}"; do
  for idx in 0 1 2 3 4 5 6 7; do
    JOBS+=("${yaml}|${idx}")
  done
done
echo "Total jobs: ${#JOBS[@]}  GPUs: ${NUM_GPUS}"

i=0
for spec in "${JOBS[@]}"; do
  while (( ${#PIDS[@]} >= NUM_GPUS )); do
    for j in "${!PIDS[@]}"; do
      kill -0 "${PIDS[$j]}" 2>/dev/null || unset 'PIDS[j]'
    done
    PIDS=("${PIDS[@]}")
    sleep 5
  done
  yaml="${spec%|*}"
  idx="${spec#*|}"
  gpu=$((i % NUM_GPUS))
  name=$(basename "$yaml" .yaml)
  echo "[$(date +%H:%M:%S)] launch yaml=$name idx=$idx gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/sweep_runner.py run --sweep "$yaml" --idx "$idx" --override logging.save_csv=false --override logging.save_config=false --override logging.save_checkpoint=false >/dev/null 2>&1 &
  PIDS+=($!)
  sleep 2
  i=$((i+1))
done
wait
echo "[$(date +%H:%M:%S)] All ${#JOBS[@]} jobs complete"
