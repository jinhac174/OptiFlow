#!/usr/bin/env bash
# Usage: bash scripts/run_local_sweep.sh <sweep-yaml> [num_gpus=8] [parallel_per_gpu=1]
set -uo pipefail
YAML=${1:?"need yaml"}
NUM_GPUS=${2:-8}
PER_GPU=${3:-1}
MAX=$((NUM_GPUS * PER_GPU))

cd "$(dirname "$0")/.."
mkdir -p logs/local

N=$(python3 scripts/sweep_runner.py count --sweep "$YAML")
NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$YAML'))['name'])")
echo "Sweep=$NAME total=$N gpus=$NUM_GPUS per_gpu=$PER_GPU max_inflight=$MAX"

PIDS=()
for ((idx=0; idx<N; idx++)); do
    while (( ${#PIDS[@]} >= MAX )); do
        for i in "${!PIDS[@]}"; do
            kill -0 "${PIDS[$i]}" 2>/dev/null || unset 'PIDS[i]'
        done
        PIDS=("${PIDS[@]}")
        sleep 5
    done
    GPU=$((idx % NUM_GPUS))
    LOG="logs/local/${NAME}_${idx}.log"
    echo "[$(date +%H:%M:%S)] launch idx=$idx gpu=$GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/sweep_runner.py run --sweep "$YAML" --idx "$idx" >"$LOG" 2>&1 &
    PIDS+=($!)
    sleep 1
done
echo "[$(date +%H:%M:%S)] all queued; waiting..."
wait
echo "[$(date +%H:%M:%S)] DONE"
