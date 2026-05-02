#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Per-job logs are written to slurm/logs/online_{ARRAY_JOB_ID}/job_{ARRAY_TASK_ID}.{out,err}

export PYTHONUNBUFFERED=1
export MUJOCO_GL=osmesa

TASK_FILE="${1:?Usage: sbatch ... run_online_finetune_array.sh <task_file>}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

LOG_DIR="/home/danielc174/projects/FPOT/slurm/logs/online_${ARRAY_JOB_ID}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/job_${ARRAY_TASK_ID}.out")
exec 2> >(tee -a "${LOG_DIR}/job_${ARRAY_TASK_ID}.err" >&2)

# Task file is 1-indexed (submitted with --array=1-N)
TASK_LINE="$(sed -n "${ARRAY_TASK_ID}p" "${TASK_FILE}")"
if [[ -z "${TASK_LINE}" ]]; then
    echo "No task found for SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} in ${TASK_FILE}"
    exit 1
fi

IFS='|' read -r ENV_NAME CKPT_DIR SEED RUN_DIR <<< "${TASK_LINE}"
echo "[$(date '+%F %T')] host=$(hostname) job=${ARRAY_JOB_ID} idx=${ARRAY_TASK_ID}"
echo "env=${ENV_NAME}  ckpt=${CKPT_DIR}  seed=${SEED}  run_dir=${RUN_DIR}"

# ---- Select conda env based on env type ----
if [[ "${ENV_NAME}" == *"singletask"* ]]; then
    CONDA_ENV_DIR="/home/danielc174/miniconda3/envs/flowrl-ogbench"
    PYTHON_MM="3.12"
else
    CONDA_ENV_DIR="/home/danielc174/miniconda3/envs/flowrl-d4rl"
    PYTHON_MM="3.10"
    export D4RL_SUPPRESS_IMPORT_ERROR=1
fi

PYTHON="${CONDA_ENV_DIR}/bin/python"
NVIDIA="${CONDA_ENV_DIR}/lib/python${PYTHON_MM}/site-packages/nvidia"

export LD_LIBRARY_PATH="\
${NVIDIA}/cusparse/lib:\
${NVIDIA}/cublas/lib:\
${NVIDIA}/cuda_runtime/lib:\
${NVIDIA}/cudnn/lib:\
${NVIDIA}/cufft/lib:\
${NVIDIA}/cusolver/lib:\
${NVIDIA}/nccl/lib:\
${NVIDIA}/nvjitlink/lib:\
${HOME}/.mujoco/mujoco210/bin:\
/usr/lib/nvidia:\
/opt/ohpc/pub/apps/cuda/12.8/lib64:\
${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export DISABLE_TQDM="${DISABLE_TQDM:-1}"

cd /home/danielc174/projects/FPOT

# ---- Hyperparams (override via env vars before sbatch) ----
WANDB_PROJECT="${WANDB_PROJECT:-fpot-online-finetune}"
WANDB_MODE="${WANDB_MODE:-online}"
ONLINE_STEPS="${ONLINE_STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-2000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
GRADIENT_UPDATES="${GRADIENT_UPDATES:-1}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"

trap 'echo "[$(date +%FT%T)] TERM signal idx=${ARRAY_TASK_ID}"; exit 143' TERM

"${PYTHON}" scripts/train_fpot_online.py \
    --checkpoint_path  "${CKPT_DIR}" \
    --env_name         "${ENV_NAME}" \
    --seed             "${SEED}" \
    --online_steps     "${ONLINE_STEPS}" \
    --batch_size       "${BATCH_SIZE}" \
    --replay_capacity  "${REPLAY_CAPACITY}" \
    --eval_interval    "${EVAL_INTERVAL}" \
    --save_interval    "${SAVE_INTERVAL}" \
    --gradient_updates_per_env_step "${GRADIENT_UPDATES}" \
    --num_eval_episodes "${NUM_EVAL_EPISODES}" \
    --run_dir          "${RUN_DIR}" \
    --wandb_project    "${WANDB_PROJECT}" \
    --wandb_mode       "${WANDB_MODE}"

EXIT_CODE=$?
echo "[$(date '+%F %T')] Done  idx=${ARRAY_TASK_ID}  exit_code=${EXIT_CODE}"
exit "${EXIT_CODE}"
