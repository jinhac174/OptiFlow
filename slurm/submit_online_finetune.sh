#!/usr/bin/env bash
# Discover offline 1M-step snapshots and submit online FPOT fine-tuning array.
#
# Prerequisites: e015 + e016 offline jobs must have completed successfully.
#
# Usage:
#   bash slurm/submit_online_finetune.sh [OPTIONS]
#
# Options (env vars):
#   PARTITION       SLURM partition         (default: base_suma_rtx3090,suma_rtx4090)
#   QOS             SLURM QOS               (default: big_qos)
#   TIME            Wall time               (default: 48:00:00)
#   MEM             Memory per job          (default: 64G)
#   CPUS            CPUs per task           (default: 20)
#   SEEDS           Space-separated seeds   (default: "1 2 3")
#   WANDB_PROJECT   WandB project name      (default: fpot-online-finetune)
#   WANDB_MODE      disabled / online       (default: online)
#   ONLINE_STEPS    Gradient steps          (default: 500000)
#   DRY_RUN         1 = print tasks only    (default: 0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS_ROOT="${REPO_ROOT}/outputs"
GENERATED_DIR="${REPO_ROOT}/slurm/generated/online_finetune"
TASK_FILE="${GENERATED_DIR}/tasks.txt"
LOG_FILE="${GENERATED_DIR}/job_ids.txt"

mkdir -p "${GENERATED_DIR}"
: > "${TASK_FILE}"
: > "${LOG_FILE}"

# ---- Configuration ----
PARTITION="${PARTITION:-base_suma_rtx3090,suma_rtx4090}"
QOS="${QOS:-big_qos}"
TIME="${TIME:-48:00:00}"
MEM="${MEM:-64G}"
CPUS="${CPUS:-20}"
SEEDS="${SEEDS:-1 2 3}"
DRY_RUN="${DRY_RUN:-0}"

# ---- Environment table: env_cfg_name | env_name_str | offline_sweep ----
# env_cfg_name  : name field in configs/env/*.yaml  (= subdir under outputs/)
# env_name_str  : actual env string passed to train_fpot_online.py
# offline_sweep : sweep name = experiment yaml 'name' field (used as the run subdir)
declare -a ENV_ENTRIES=(
    "puzzle_4x4_t2|puzzle-4x4-play-singletask-task2-v0|e015_offline_puzzle_4x4_t2"
    "puzzle_4x4_t5|puzzle-4x4-play-singletask-task5-v0|e016_offline_puzzle_4x4_t5"
    "cube_double_t3|cube-double-play-singletask-task3-v0|e017_offline_cube_double_t3"
    "cube_double_t5|cube-double-play-singletask-task5-v0|e018_offline_cube_double_t5"
    "door_cloned|door-cloned-v1|e019_offline_door_cloned"
    "hammer_cloned|hammer-cloned-v1|e020_offline_hammer_cloned"
)

# ---- Build task file ----
missing=0
for entry in "${ENV_ENTRIES[@]}"; do
    IFS='|' read -r ENV_DIR ENV_NAME OFFLINE_SWEEP <<< "${entry}"
    for seed in ${SEEDS}; do
        # Prefer final_agent, fall back to agent_step_1000000
        ckpt_dir=""
        for pattern in "final_agent" "agent_step_1000000"; do
            found="$(find "${OUTPUTS_ROOT}/${ENV_DIR}/${OFFLINE_SWEEP}" \
                -maxdepth 5 -type f -name "params_0.pkl" \
                -path "*seed_${seed}*/${pattern}/params_0.pkl" \
                2>/dev/null | sort | head -1)"
            if [[ -n "${found}" ]]; then
                ckpt_dir="$(dirname "${found}")"
                break
            fi
        done

        if [[ -z "${ckpt_dir}" ]]; then
            echo "  [MISSING] ${ENV_DIR}  seed=${seed}  (offline snapshot not found under ${OUTPUTS_ROOT}/${ENV_DIR}/${OFFLINE_SWEEP})"
            missing=$((missing + 1))
            continue
        fi

        run_dir="${OUTPUTS_ROOT}/${ENV_DIR}/online_fpot/seed_${seed}"
        printf '%s|%s|%s|%s\n' "${ENV_NAME}" "${ckpt_dir}" "${seed}" "${run_dir}" >> "${TASK_FILE}"
        echo "  [OK] ${ENV_DIR}  seed=${seed}  ckpt=${ckpt_dir}"
    done
done

TASK_COUNT="$(wc -l < "${TASK_FILE}")"

echo ""
echo "Task file : ${TASK_FILE}"
echo "Tasks     : ${TASK_COUNT}  (missing: ${missing})"

if [[ "${TASK_COUNT}" -eq 0 ]]; then
    echo "No tasks to submit. Ensure offline jobs have completed."
    exit 1
fi

if [[ "${missing}" -gt 0 ]]; then
    echo "WARNING: ${missing} snapshot(s) not found — those tasks skipped."
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 — task file written but no sbatch submitted."
    exit 0
fi

# ---- Submit array job ----
JOB_ID="$(
    sbatch \
        --parsable \
        --job-name="fpot-online" \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --time="${TIME}" \
        --mem="${MEM}" \
        --cpus-per-task="${CPUS}" \
        --gres=gpu:1 \
        --array="1-${TASK_COUNT}" \
        --chdir="${REPO_ROOT}" \
        --export=ALL,\
WANDB_PROJECT="${WANDB_PROJECT:-fpot-online-finetune}",\
WANDB_MODE="${WANDB_MODE:-online}",\
ONLINE_STEPS="${ONLINE_STEPS:-500000}",\
BATCH_SIZE="${BATCH_SIZE:-256}",\
REPLAY_CAPACITY="${REPLAY_CAPACITY:-2000000}",\
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}",\
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}",\
GRADIENT_UPDATES="${GRADIENT_UPDATES:-1}",\
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}" \
        slurm/run_online_finetune_array.sh "${TASK_FILE}"
)"

echo "Submitted job array: ${JOB_ID}  (${TASK_COUNT} tasks)"
printf 'job_id=%s\ttasks=%s\ttask_file=%s\n' "${JOB_ID}" "${TASK_COUNT}" "${TASK_FILE}" >> "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
