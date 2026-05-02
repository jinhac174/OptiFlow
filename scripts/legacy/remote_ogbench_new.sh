#!/bin/bash
#SBATCH --job-name=og-new
#SBATCH --partition=big_suma_rtx3090,base_suma_rtx3090,suma_rtx4090
#SBATCH --qos=big_qos
#SBATCH --array=0-89
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --exclude=node04,node08,node18,node31

ENV=/home/danielc174/miniconda3/envs/flowrl-ogbench
PYTHON=$ENV/bin/python
NVIDIA=$ENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$NVIDIA/cusparse/lib:$NVIDIA/cublas/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cudnn/lib:$NVIDIA/cufft/lib:$NVIDIA/cusolver/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/opt/ohpc/pub/apps/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

cd /home/danielc174/projects/FPOT

JOB_LIST=()
SEEDS=(2 4)

# humanoidmaze-medium: coworker used (1.0, 0.001)
HM_CONFIGS=("1.0 0.001 mean" "2.0 0.1 mean" "10.0 10.0 mean")
for t in 1 2 3 4 5; do
  for cfg in "${HM_CONFIGS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      JOB_LIST+=("humanoidmaze_medium_navigate_task${t} $cfg $sd")
    done
  done
done

# humanoidmaze-large: same configs as medium
for t in 1 2 3 4 5; do
  for cfg in "${HM_CONFIGS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      JOB_LIST+=("humanoidmaze_large_navigate_task${t} $cfg $sd")
    done
  done
done

# antsoccer: coworker used (1.0, 0.01)
AS_CONFIGS=("1.0 0.01 mean" "2.0 0.1 mean" "10.0 10.0 mean")
for t in 1 2 3 4 5; do
  for cfg in "${AS_CONFIGS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      JOB_LIST+=("antsoccer_arena_navigate_task${t} $cfg $sd")
    done
  done
done

# Total: 30 + 30 + 30 = 90

read ENV_FULL TEMP E QAGG SEED <<< "${JOB_LIST[$SLURM_ARRAY_TASK_ID]}"

ENV_BASE=$(echo $ENV_FULL | sed 's/_task[0-9]//')
TASK_NUM=$(echo $ENV_FULL | grep -o 'task[0-9]' | grep -o '[0-9]')

TEMP_TAG=$(echo $TEMP | tr -d '.')
ETA_TAG=$(echo $E | tr -d '.')
EXP_NAME="task${TASK_NUM}_t${TEMP_TAG}_eta${ETA_TAG}_q${QAGG}"

echo "[$(date)] Job $SLURM_ARRAY_TASK_ID: $EXP_NAME $ENV_FULL seed=$SEED node=$(hostname)"

$PYTHON scripts/train.py \
  env=$ENV_FULL \
  seed=$SEED \
  experiment_name=$EXP_NAME \
  agent.agent_file=agent_gfot \
  agent.w_temperature=$TEMP \
  agent.eta_temperature=$E \
  agent.q_agg=$QAGG \
  +logging.wandb_tags="[wt_${TEMP},eta_${E},qagg_${QAGG},algo_gfot,task${TASK_NUM},${ENV_BASE}]"

echo "[$(date)] Done: $EXP_NAME $ENV_FULL seed=$SEED"
