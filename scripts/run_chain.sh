#!/usr/bin/env bash
# Chain runner for offline → resolve → online sweeps on a single multi-GPU host.
# Designed for non-preempting environments (e.g. Vast AI). Each phase calls
# run_sweep.sh, which queues at most NUM_GPUS * PER_GPU jobs concurrently.
#
# Usage (foreground, e.g. tmux session):
#   bash scripts/run_chain.sh
#
# Usage (background-survives-disconnect — recommended on Vast):
#   nohup bash scripts/run_chain.sh > logs/local/chain.log 2>&1 &
#   disown
#
# Edit OFFLINE_YAMLS below to change which tasks run; tune NUM_GPUS / PER_GPU
# via env (defaults: 8 / 1).
set -uo pipefail

cd "$(dirname "$0")/.."

NUM_GPUS=${NUM_GPUS:-8}
PER_GPU=${PER_GPU:-1}

OFFLINE_YAMLS=(
    "experiments/online/chain/offline_antsoccer_arena_task4.yaml"
    "experiments/online/chain/offline_scene_play_task2.yaml"
)

ONLINE_YAML_DIR=experiments/online/chain/runtime
mkdir -p "$ONLINE_YAML_DIR" logs/local

# --- Pre-download datasets serially (avoids race when 8 jobs all download at once) ---
for yaml_path in "${OFFLINE_YAMLS[@]}"; do
    env_name=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_path'))['fixed']['env.env_name'])")
    echo "[$(date)] pre-download $env_name"
    python3 -c "from envs.env_utils import make_env_and_datasets; make_env_and_datasets(env_name='$env_name')"
done

# --- Phase 1: offline sweeps (saves final_agent per seed) ---
for yaml_path in "${OFFLINE_YAMLS[@]}"; do
    echo "[$(date)] === offline sweep: $yaml_path ==="
    bash scripts/run_sweep.sh "$yaml_path" "$NUM_GPUS" "$PER_GPU"
done

# --- Resolve offline ckpts → online yamls ---
ONLINE_YAMLS=()
for yaml_path in "${OFFLINE_YAMLS[@]}"; do
    base=$(basename "$yaml_path" .yaml)
    online_yaml="$ONLINE_YAML_DIR/${base/offline_/online_}.yaml"
    echo "[$(date)] resolve ckpts: $yaml_path -> $online_yaml"
    python3 scripts/build_resume_yaml.py \
        --offline-sweep "$yaml_path" \
        --output "$online_yaml"
    ONLINE_YAMLS+=("$online_yaml")
done

# --- Phase 2: online resume sweeps ---
for yaml_path in "${ONLINE_YAMLS[@]}"; do
    echo "[$(date)] === online sweep: $yaml_path ==="
    bash scripts/run_sweep.sh "$yaml_path" "$NUM_GPUS" "$PER_GPU"
done

echo "[$(date)] CHAIN COMPLETE"
