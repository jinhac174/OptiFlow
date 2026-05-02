#!/usr/bin/env bash
# Submit offline FPOTAgent training jobs with 1M-step snapshot saving.
# Each experiment yaml covers one environment (3 seeds = 3 jobs each).
#
# OGBench envs (gfp2 conda):
#   e015 — puzzle-4x4 task2
#   e016 — puzzle-4x4 task5
#   e017 — cube-double task3
#   e018 — cube-double task5
#
# D4RL envs (flowrl-d4rl conda):
#   e019 — door-cloned
#   e020 — hammer-cloned
#
# Usage:
#   bash slurm/submit_offline_snapshot.sh
#
# To submit a single env only:
#   bash scripts/submit.sh experiments/e015_offline_puzzle_4x4_t2.yaml

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== OGBench offline snapshot jobs ==="
bash scripts/submit.sh experiments/e015_offline_puzzle_4x4_t2.yaml
bash scripts/submit.sh experiments/e016_offline_puzzle_4x4_t5.yaml
bash scripts/submit.sh experiments/e017_offline_cube_double_t3.yaml
bash scripts/submit.sh experiments/e018_offline_cube_double_t5.yaml

echo ""
echo "=== D4RL offline snapshot jobs ==="
bash scripts/submit.sh experiments/e019_offline_door_cloned.yaml
bash scripts/submit.sh experiments/e020_offline_hammer_cloned.yaml
