#!/usr/bin/env bash
# Chain the three OGBench offline-to-online sweeps. Each sweep is dispatched
# by scripts/run_local_sweep.sh, which assigns each job index deterministically
# to a GPU (idx % NUM_GPUS) and writes per-job logs to logs/local/<sweep>_<idx>.log.
# Sweeps run sequentially: the next one starts only after the previous one's
# 8 jobs all finish.
#
# Usage:
#   bash scripts/launch_online_vast.sh
#   tail -f logs/local/launcher_chain.log
#
# Survives ssh disconnect via nohup + disown.

cd "$(cd "$(dirname "$0")" && pwd)/.."
mkdir -p logs/local

NUM_GPUS=${NUM_GPUS:-8}
PER_GPU=${PER_GPU:-1}

nohup bash -c "
    echo \"[\$(date)] Starting humanoidmaze_medium sweep\"
    bash scripts/run_local_sweep.sh experiments/online/humanoidmaze_medium.yaml ${NUM_GPUS} ${PER_GPU}

    echo \"[\$(date)] Starting cube_double sweep\"
    bash scripts/run_local_sweep.sh experiments/online/cube_double.yaml ${NUM_GPUS} ${PER_GPU}

    echo \"[\$(date)] Starting puzzle_4x4 sweep\"
    bash scripts/run_local_sweep.sh experiments/online/puzzle_4x4.yaml ${NUM_GPUS} ${PER_GPU}

    echo \"[\$(date)] All sweeps done\"
" > logs/local/launcher_chain.log 2>&1 &
disown

echo "Launched. Tail logs/local/launcher_chain.log for chain progress."
echo "Per-job logs land in logs/local/<sweep_name>_<idx>.log."
