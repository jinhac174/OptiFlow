#!/usr/bin/env bash
# Launch the cube-single + cube-double smoke runs on a single multi-GPU box
# (no Slurm, no per-job stdout logs, no metrics.csv, no checkpoints — for use
# on space-constrained VastAI machines).
#
# 5 tasks x 2 seeds (1, 5) x 2 sweeps = 20 jobs, fanned out across 8 GPUs.
# Job indices for seeds {1, 5} in each yaml: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36.
#
# Usage:
#   tmux new -s fpot
#   bash scripts/launch_smoke.sh
#   # detach: Ctrl-b d ; reattach: tmux attach -t fpot

set -u

NUM_GPUS=${NUM_GPUS:-8}
PIDS=()
JOBS=()

YAMLS=(
  experiments/benchmarks/ogbench/manipulation/cube_single.yaml
  experiments/benchmarks/ogbench/manipulation/cube_double.yaml
)

for yaml in "${YAMLS[@]}"; do
  for idx in 0 4 8 12 16 20 24 28 32 36; do
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
  CUDA_VISIBLE_DEVICES=$gpu python scripts/sweep_runner.py run --sweep "$yaml" --idx "$idx" logging.save_csv=false logging.save_config=false logging.save_checkpoint=false >/dev/null 2>&1 &
  PIDS+=($!)
  sleep 2
  i=$((i+1))
done
wait
echo "[$(date +%H:%M:%S)] All ${#JOBS[@]} jobs complete"
