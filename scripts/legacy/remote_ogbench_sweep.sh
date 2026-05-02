#!/bin/bash
#SBATCH --job-name=og-sweep
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --exclude=node04,node08,node18,node31

ENV=/home/danielc174/miniconda3/envs/flowrl-ogbench
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

cd /home/danielc174/projects/FPOT

ENV_BASES=(cube_single_play cube_double_play scene_play puzzle_3x3_play puzzle_4x4_play)
TASKS=(1 2 3 4 5)
CONFIGS=(
  "1.0 0.01"
  "1.0 0.1"
  "2.0 0.01"
  "2.0 0.1"
)

NUM_TASKS=${#TASKS[@]}
NUM_CFGS=${#CONFIGS[@]}
JOBS_PER_ENV=$((NUM_TASKS * NUM_CFGS))
ENV_IDX=$((SLURM_ARRAY_TASK_ID / JOBS_PER_ENV))
REM=$((SLURM_ARRAY_TASK_ID % JOBS_PER_ENV))
TASK_IDX=$((REM / NUM_CFGS))
CFG_IDX=$((REM % NUM_CFGS))

ENV_BASE=${ENV_BASES[$ENV_IDX]}
TASK_NUM=${TASKS[$TASK_IDX]}
read TEMP E <<< "${CONFIGS[$CFG_IDX]}"

ENV_FULL="${ENV_BASE}_task${TASK_NUM}"
SEED=1

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="task${TASK_NUM}_t${TEMP_TAG}_eta${ETA_TAG}_qmean"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $ENV_FULL seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$ENV_FULL \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=mean \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_mean,algo_gfot,task${TASK_NUM},${ENV_BASE}]"

echo "[$(date)] Done: $EXP_NAME $ENV_FULL seed=$SEED"
