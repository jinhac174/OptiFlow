#!/usr/bin/env bash
# submit.sh — submit a sweep yaml as a Slurm array job.
#
# Usage:
#   ./scripts/submit.sh experiments/eXXX.yaml
#
# Reads env_set, slurm:, wandb_project, and name from the yaml.
# Picks the right runner script, computes N jobs, builds sbatch flags,
# prints the full command, then submits.

set -euo pipefail

YAML=${1:?"Usage: $0 <sweep-yaml> [--array=RANGE] [--qos=NAME]"}
YAML=$(realpath "$YAML")
shift

ARRAY_OVERRIDE=""
QOS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --array=*) ARRAY_OVERRIDE="${1#--array=}" ;;
        --qos=*)   QOS_OVERRIDE="${1#--qos=}" ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- parse yaml fields via PyYAML (no yq dependency) ---
_py() {
    python3 - "$YAML" <<EOF
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
$1
EOF
}

SWEEP_NAME=$(_py "print(cfg['name'])")
ENV_SET=$(_py "print(cfg.get('env_set', 'd4rl'))")

# --- pick runner ---
if [ "$ENV_SET" = "d4rl" ]; then
    RUNNER="slurm/d4rl_array.sh"
else
    RUNNER="slurm/ogbench_array.sh"
fi

# --- compute total jobs ---
N=$(python3 scripts/sweep_runner.py count --sweep "$YAML")
if [ -n "$ARRAY_OVERRIDE" ]; then
    ARRAY_SPEC="$ARRAY_OVERRIDE"
else
    ARRAY_SPEC="0-$((N - 1))"
fi

# --- read slurm block ---
PARTITION=$(_py "
slurm = cfg.get('slurm', {})
parts = slurm.get('partitions', [])
print(','.join(str(p) for p in parts) if parts else '')
")
QOS=$(_py "print(cfg.get('slurm', {}).get('qos', ''))")
if [ -n "$QOS_OVERRIDE" ]; then
    QOS="$QOS_OVERRIDE"
fi
TIME=$(_py "print(cfg.get('slurm', {}).get('time', '24:00:00'))")
MEM=$(_py "print(cfg.get('slurm', {}).get('mem', '16G'))")
CPUS=$(_py "print(cfg.get('slurm', {}).get('cpus_per_task', 4))")
EXCLUDE=$(_py "
slurm = cfg.get('slurm', {})
ex = slurm.get('exclude', [])
print(','.join(str(e) for e in ex) if ex else '')
")

# --- build sbatch flags ---
SBATCH_FLAGS=(
    "--array=${ARRAY_SPEC}"
    "--job-name=${SWEEP_NAME}"
    "--gres=gpu:1"
    "--time=${TIME}"
    "--mem=${MEM}"
    "--cpus-per-task=${CPUS}"
    "--output=slurm/logs/%x_%A_%a.out"
    "--error=slurm/logs/%x_%A_%a.err"
)

[ -n "$PARTITION" ] && SBATCH_FLAGS+=("--partition=${PARTITION}")
[ -n "$QOS"       ] && SBATCH_FLAGS+=("--qos=${QOS}")
[ -n "$EXCLUDE"   ] && SBATCH_FLAGS+=("--exclude=${EXCLUDE}")

# --- ensure log directory exists ---
mkdir -p slurm/logs

# --- print and submit ---
echo "Submitting sweep: $SWEEP_NAME  ($N jobs)"
echo "Runner: $RUNNER"
echo ""
echo "sbatch ${SBATCH_FLAGS[*]} $RUNNER $YAML"
echo ""
sbatch "${SBATCH_FLAGS[@]}" "$RUNNER" "$YAML"
